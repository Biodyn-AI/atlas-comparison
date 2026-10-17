#!/usr/bin/env python
"""Step 4 (job item 3): does the choice of degree-preserving null matter?

Null A (repo): stub-matching configuration model, config_shuffle from hypothesis_trrust2.py:60-69.
  Stubs are shuffled and re-paired; self-loops dropped; duplicate pairs collapsed. So a null graph
  has slightly FEWER distinct pairs than the real graph, and hub genes lose some degree.
Null B (new): Maslov-Sneppen double-edge swaps. Start from the real graph; attempt 10 x E swaps
  (pick two edges (a,b),(c,d) uniformly, pick one of the two re-pairings at random, giving (a,d),(c,b);
  reject if it makes a self-loop or a pair that already exists). Every gene keeps its degree
  exactly and the graph stays simple (no self-loops, no duplicates). Each draw starts from the real
  graph. Implemented with numba; seed = 1000 + draw index.
Per model (7 models, pooled W>=5 graph): NPERM draws of each null; hits, null mean/sd, fold, empirical p.
Cross-model curve (all ten graphs, as hypothesis_trrust2.py:112-146, and the five passing models, as
hypothesis_trrust3.py): each draw rewires every graph independently; precision at >= k models;
NCURVE draws of each null.
Output: OUT/s04_null_choice.json
"""
import json, os, sys, time
from collections import defaultdict
import numpy as np
from numba import njit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200
NCURVE = int(sys.argv[2]) if len(sys.argv) > 2 else 100
SWAPS_PER_EDGE = 10


@njit(cache=False)
def ms_rewire(u0, v0, n_attempts, seed):
    np.random.seed(seed)
    u = u0.copy(); v = v0.copy()
    E = u.shape[0]
    S = 2097152  # 1 << 21
    s = set()
    for i in range(E):
        s.add(min(u[i], v[i]) * S + max(u[i], v[i]))
    acc = 0
    for _ in range(n_attempts):
        i = np.random.randint(E); j = np.random.randint(E)
        if i == j:
            continue
        a = u[i]; b = v[i]; c = u[j]; d = v[j]
        if np.random.random() < 0.5:
            t = c; c = d; d = t
        if a == d or c == b:
            continue
        k1 = min(a, d) * S + max(a, d)
        k2 = min(c, b) * S + max(c, b)
        if k1 == k2 or k1 in s or k2 in s:
            continue
        s.discard(min(a, b) * S + max(a, b)); s.discard(min(c, d) * S + max(c, d))
        s.add(k1); s.add(k2)
        u[i] = a; v[i] = d; u[j] = c; v[j] = b
        acc += 1
    out = np.empty(E, np.int64)
    for i in range(E):
        out[i] = min(u[i], v[i]) * S + max(u[i], v[i])
    return np.sort(out), acc


def ms_keys(keys, seed):
    a, b = decode(keys)
    k, acc = ms_rewire(a.astype(np.int64), b.astype(np.int64), SWAPS_PER_EDGE * len(keys), seed)
    return k, acc


genes = load_genes(); gid = {g: i for i, g in enumerate(genes)}
truth, _, _ = load_truth()
tkey = truth_keys(truth, gid)

# sanity test of the swap on a small graph: degrees preserved, simple graph
rng0 = np.random.default_rng(5)
tk = np.unique(np.sort(rng0.integers(0, 300, (3000, 2)), axis=1) @ np.array([SHIFT, 1]))
tk = tk[(tk // SHIFT) != (tk % SHIFT)]
k2, acc = ms_keys(tk, 7)
deg = lambda k: np.bincount(np.concatenate(decode(k)), minlength=300)
sanity = {"degrees_identical": bool(np.array_equal(deg(tk), deg(k2))), "n_unique": int(len(np.unique(k2))),
          "n_edges": int(len(tk)), "no_self_loops": bool(((k2 // SHIFT) != (k2 % SHIFT)).all()),
          "share_original_edges_kept": round(float(np.isin(k2, tk).mean()), 4), "accepted": int(acc)}
print("sanity", sanity, flush=True)
assert sanity["degrees_identical"] and sanity["n_unique"] == sanity["n_edges"] and sanity["no_self_loops"]

G5 = {}
for m in ORDER:
    feat, lay, L = load_model(m)
    G5[m] = graph_from(feat, lay, None, 5)[0]
    del feat, lay

pub = json.load(open(f"{ORIG}/outputs/atlas/comparative/hypothesis_trrust2.json"))
out = {"n_perm": NPERM, "n_perm_curve": NCURVE, "swaps_attempted_per_edge": SWAPS_PER_EDGE,
       "sanity_small_graph": sanity, "per_model": {}, "curve": {}}
t0 = time.time()
for m in SEVEN:
    keys = G5[m]
    obs = int(np.isin(keys, tkey).sum())
    rng = np.random.default_rng(0)
    stub_h, stub_n = [], []
    for _ in range(NPERM):
        k2 = config_shuffle_keys(keys, rng)
        stub_h.append(int(np.isin(k2, tkey).sum())); stub_n.append(len(k2))
    ms_h, ms_acc, ms_keep = [], [], []
    for d in range(NPERM):
        k2, acc = ms_keys(keys, 1000 + d)
        ms_h.append(int(np.isin(k2, tkey).sum())); ms_acc.append(acc)
        ms_keep.append(float(np.isin(k2, keys).mean()))
    stub_h = np.array(stub_h, float); ms_h = np.array(ms_h, float)
    E = len(keys)
    out["per_model"][m] = {
        "n_pairs": E, "hits": obs,
        "published_stub": {k: pub["per_model"][m][k] for k in ("null_mean", "null_sd", "fold", "p_emp")} | {"n_perm": pub["n_perm"]},
        "stub_matching": summarize(obs, stub_h) | {"null_distinct_pairs_mean": round(float(np.mean(stub_n)), 1),
                                                    "null_pairs_kept_share": round(float(np.mean(stub_n)) / E, 4)},
        "maslov_sneppen": summarize(obs, ms_h) | {"accepted_swaps_mean": round(float(np.mean(ms_acc)), 1),
                                                   "accepted_per_edge": round(float(np.mean(ms_acc)) / E, 2),
                                                   "share_original_pairs_left_mean": round(float(np.mean(ms_keep)), 4)}}
    r = out["per_model"][m]
    print(f"{m}: E={E} hits={obs} | stub null {r['stub_matching']['null_mean']} fold {r['stub_matching']['fold']} p {r['stub_matching']['p_emp']}"
          f" kept {r['stub_matching']['null_pairs_kept_share']} | MS null {r['maslov_sneppen']['null_mean']} fold {r['maslov_sneppen']['fold']}"
          f" p {r['maslov_sneppen']['p_emp']} acc/edge {r['maslov_sneppen']['accepted_per_edge']} orig left {r['maslov_sneppen']['share_original_pairs_left_mean']}"
          f"  [{time.time()-t0:.0f}s]", flush=True)


def curve(models, label):
    allp = defaultdict(int)
    for m in models:
        for k in G5[m].tolist(): allp[k] += 1
    kk = np.fromiter(allp.keys(), np.int64, len(allp)); vv = np.fromiter(allp.values(), np.int32, len(allp))
    ist = np.isin(kk, tkey)
    obs = {k: (int((vv >= k).sum()), int(ist[vv >= k].sum())) for k in (1, 2, 3)}
    res = {}
    for nul in ("stub_matching", "maslov_sneppen"):
        rng = np.random.default_rng(0)
        prec = {k: [] for k in obs}; nh = {k: [] for k in obs}; npairs = {k: [] for k in obs}
        for d in range(NCURVE):
            cnt = defaultdict(int)
            for mi, m in enumerate(models):
                if len(G5[m]) == 0: continue
                k2 = config_shuffle_keys(G5[m], rng) if nul == "stub_matching" else ms_keys(G5[m], 50000 + 100 * d + mi)[0]
                for x in k2.tolist(): cnt[x] += 1
            k3 = np.fromiter(cnt.keys(), np.int64, len(cnt)); v3 = np.fromiter(cnt.values(), np.int32, len(cnt))
            i3 = np.isin(k3, tkey)
            for k in obs:
                sel = v3 >= k
                prec[k].append(float(i3[sel].sum()) / max(int(sel.sum()), 1)); nh[k].append(int(i3[sel].sum())); npairs[k].append(int(sel.sum()))
        res[nul] = {}
        for k, (n, h) in obs.items():
            pc = np.array(prec[k]); p_obs = h / n
            res[nul][k] = {"n_pairs": n, "hits": h, "precision": round(p_obs, 6),
                           "null_precision_mean": round(float(pc.mean()), 8), "null_precision_sd": round(float(pc.std()), 8),
                           "null_hits_mean": round(float(np.mean(nh[k])), 3), "null_pairs_mean": round(float(np.mean(npairs[k])), 1),
                           "null_draws_with_zero_hits": int((np.array(nh[k]) == 0).sum()),
                           "fold": round(p_obs / pc.mean(), 2) if pc.mean() > 0 else None,
                           "p_emp": round((int((pc >= p_obs).sum()) + 1) / (NCURVE + 1), 5), "n_perm": NCURVE}
            print(f"  {label} {nul} >= {k}: {n} pairs {h} hits prec {p_obs:.5f} null {pc.mean():.7f} fold {res[nul][k]['fold']} p {res[nul][k]['p_emp']}"
                  f"  [{time.time()-t0:.0f}s]", flush=True)
    return res


out["curve"]["all_ten"] = curve(ORDER, "all ten")
out["curve"]["five_passing"] = curve(PASSED, "five passing")
json.dump(out, open(f"{OUT}/s04_null_choice.json", "w"), indent=1)
print("done", time.time() - t0)
