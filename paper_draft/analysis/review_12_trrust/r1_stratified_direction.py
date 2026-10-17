#!/usr/bin/env python
"""Review of note 12, check R1: is the TF -> non-TF test in s03_hub_direction.py fair?

Problem found in review: s03 compares the OBSERVED number of TF -> non-TF hits with the number of
TF -> non-TF hits in the repo's stub-matching null of the WHOLE graph. But the observed graph has far
more TF-TF pairs than the null (UCE 738 vs 95.5), so it has FEWER TF - non-TF pairs than the null.
The null is therefore given more TF - non-TF pairs to hit than the model had. That biases the TF ->
non-TF test against the model (and the TF-TF test in its favour).

Fix: a block-stratified configuration null. The graph is split by pair type and each block is
rewired on its own, so each gene keeps its exact number of TF partners and non-TF partners:
  TF-TF block      : stub matching among the TF ends only (repo config_shuffle logic: shuffle,
                     re-pair, drop self-loops, collapse duplicates)
  TF - non-TF block: the TF ends stay, the non-TF ends are permuted (bipartite configuration model;
                     duplicates collapsed; no self-loops possible)
  non-TF - non-TF  : cannot contain a TRRUST edge (every TRRUST edge has a regulator), not rewired.
Graphs, truth, TF set and the hit definition are exactly those of note 12 (its common.py, which
reproduces hypothesis_trrust2.py; graphs verified identical to the repo's .cofire_cache.pkl).
Also reported: number of TF - non-TF pairs in the observed graph vs in the plain stub null, and the
precision-matched version of the plain test (hits per TF - non-TF pair, observed vs null).
NPERM draws per model (argv[1], default 200; the saved JSON was made with `python3 r1_stratified_direction.py 1000`),
seeds 11 (stratified) and 12 (plain stub, pair counts).
Output: r1_stratified_direction.json (this folder)
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "trrust_robustness"))
from common import *  # noqa  (note 12's re-implementation of the repo ruler)

NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200
genes = load_genes(); gid = {g: i for i, g in enumerate(genes)}
truth, directed, tfs = load_truth()
tkey = truth_keys(truth, gid)
is_tf = np.zeros(len(genes), bool)
for t in tfs:
    if t in gid:
        is_tf[gid[t]] = True


def keys_from(lo, hi):
    lo = lo.astype(np.int64); hi = hi.astype(np.int64)
    a = np.minimum(lo, hi); b = np.maximum(lo, hi)
    return np.unique(a * SHIFT + b)


def stub_block(x, y, rng):
    stubs = np.concatenate([x, y]).astype(np.int64)
    rng.shuffle(stubs)
    h, t = stubs[::2], stubs[1::2]
    ok = h != t
    return keys_from(h[ok], t[ok])


def bip_block(tf_end, other_end, rng):
    return keys_from(tf_end, rng.permutation(other_end))


out = {"n_perm": NPERM, "seeds": {"stratified": 11, "plain_stub_pair_counts": 12},
       "null": "block-stratified configuration model (TF-TF block stub-matched among TF ends; TF-nonTF block "
               "bipartite: non-TF ends permuted); duplicates collapsed as in the repo config_shuffle",
       "models": {}}
t0 = time.time()
for m in SEVEN:
    feat, lay, L = load_model(m)
    keys, w = graph_from(feat, lay, None, 5)
    del feat, lay
    a, b = decode(keys)
    ta, tb = is_tf[a], is_tf[b]
    tt = ta & tb
    tn = ta ^ tb
    hit = np.isin(keys, tkey)
    obs_tt, obs_tn = int((hit & tt).sum()), int((hit & tn).sum())
    assert obs_tt + obs_tn == int(hit.sum())
    n_tt, n_tn = int(tt.sum()), int(tn.sum())
    # TF - nonTF block: orient so the first array is the TF end
    tf_end = np.where(ta[tn], a[tn], b[tn]); ot_end = np.where(ta[tn], b[tn], a[tn])
    rng = np.random.default_rng(11)
    s_tt, s_tn, s_tt_pairs, s_tn_pairs = [], [], [], []
    for _ in range(NPERM):
        k1 = stub_block(a[tt], b[tt], rng) if n_tt else np.empty(0, np.int64)
        k2 = bip_block(tf_end, ot_end, rng) if n_tn else np.empty(0, np.int64)
        s_tt.append(int(np.isin(k1, tkey).sum())); s_tn.append(int(np.isin(k2, tkey).sum()))
        s_tt_pairs.append(len(k1)); s_tn_pairs.append(len(k2))
    s_tt = np.array(s_tt, float); s_tn = np.array(s_tn, float)
    # plain stub null (the s03 null): how many TF - nonTF pairs, and hits per such pair
    rng = np.random.default_rng(12)
    p_tn_pairs, p_tn_hits = [], []
    for _ in range(NPERM):
        k = config_shuffle_keys(keys, rng)
        a2, b2 = decode(k)
        x = is_tf[a2] ^ is_tf[b2]
        p_tn_pairs.append(int(x.sum())); p_tn_hits.append(int((x & np.isin(k, tkey)).sum()))
    p_tn_pairs = np.array(p_tn_pairs, float); p_tn_hits = np.array(p_tn_hits, float)
    prec_null = p_tn_hits / np.maximum(p_tn_pairs, 1)
    prec_obs = obs_tn / n_tn if n_tn else 0.0
    out["models"][m] = {
        "n_pairs": int(len(keys)), "hits": int(hit.sum()),
        "observed": {"tf_tf_pairs": n_tt, "tf_nontf_pairs": n_tn, "hits_tf_tf": obs_tt, "hits_tf_nontf": obs_tn},
        "stratified_null": {
            "tf_nontf_block": summarize(obs_tn, s_tn) | {"null_pairs_mean": round(float(np.mean(s_tn_pairs)), 1)},
            "tf_tf_block": summarize(obs_tt, s_tt) | {"null_pairs_mean": round(float(np.mean(s_tt_pairs)), 1)},
            "both_blocks_total": summarize(obs_tt + obs_tn, s_tt + s_tn)},
        "plain_stub_null_tf_nontf": {
            "null_tf_nontf_pairs_mean": round(float(p_tn_pairs.mean()), 1),
            "observed_over_null_pairs": round(n_tn / p_tn_pairs.mean(), 4),
            "hits": summarize(obs_tn, p_tn_hits),
            "precision_obs": round(prec_obs, 6), "precision_null_mean": round(float(prec_null.mean()), 7),
            "precision_fold": round(prec_obs / prec_null.mean(), 3) if prec_null.mean() > 0 else None,
            "precision_p_emp": round((int((prec_null >= prec_obs).sum()) + 1) / (NPERM + 1), 5)}}
    r = out["models"][m]
    print(m, r["observed"], "| strat TFnon", {k: r["stratified_null"]["tf_nontf_block"][k] for k in ("null_mean", "fold", "p_emp")},
          "| strat TFTF", {k: r["stratified_null"]["tf_tf_block"][k] for k in ("null_mean", "fold", "p_emp")},
          "| strat total", {k: r["stratified_null"]["both_blocks_total"][k] for k in ("null_mean", "fold", "p_emp")},
          "| plain pairs", r["plain_stub_null_tf_nontf"]["null_tf_nontf_pairs_mean"], r["plain_stub_null_tf_nontf"]["precision_fold"],
          r["plain_stub_null_tf_nontf"]["precision_p_emp"], f"[{time.time()-t0:.0f}s]", flush=True)

# BH across the seven models for the stratified TF -> non-TF test and the stratified total
for key, lab in (("tf_nontf_block", "stratified_tf_nontf"), ("both_blocks_total", "stratified_total")):
    ps = [out["models"][m]["stratified_null"][key]["p_emp"] for m in SEVEN]
    q = bh(ps)
    out[f"bh_{lab}"] = {m: {"p": p, "q": qq} for m, p, qq in zip(SEVEN, ps, q)}
json.dump(out, open(f"{HERE}/r1_stratified_direction.json", "w"), indent=1)
print("done", time.time() - t0)
