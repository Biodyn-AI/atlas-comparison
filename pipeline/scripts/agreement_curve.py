#!/usr/bin/env python
"""Agreement between models with a well-estimated null: precision of gene pairs predicted by at least
k of the models (k = 1, 2), against the stub-matching configuration null of hypothesis_trrust2.py,
estimated from many draws.

At k >= 2 the null holds well under one known pair per draw, so the 100-draw curve of
hypothesis_trrust2.py gives a noisy ratio. Here each model's pooled W >= 5 co-firing graph (the same
graph as hypothesis_trrust2.py) is rewired independently in every draw, for NDRAW draws (default
2,000), and the script reports the null precision with its Monte Carlo standard error, the ratio to
chance with an approximate 95 % interval, the spread of the ratio over blocks of 100 and 200 draws,
and the null-free ratio of the precision at k >= 2 to the precision at k >= 1. Two sets: all ten
models, and the models that pass the per-model test in hypothesis_trrust2.json (p < 0.05).

    python scripts/agreement_curve.py [NDRAW]  -> outputs/atlas/comparative/agreement_curve.json
"""
from __future__ import annotations

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import glob, itertools, json, sys, time
import numpy as np

BASE = _B
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"; TS = f"{BASE}/outputs/atlas/ts3_out"
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
TOPG, W0, SHIFT, SEED = 10, 5, 1 << 21, 2024
NDRAW = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
COMBOS = np.array(list(itertools.combinations(range(TOPG), 2)), np.int64)

T2 = json.load(open(f"{C}/hypothesis_trrust2.json"))["per_model"]
PASSED = [m for m, v in sorted(T2.items(), key=lambda kv: -kv[1]["fold"]) if v["p_emp"] < 0.05]


def catalog_files(m):
    d = ALLCAT if glob.glob(f"{ALLCAT}/{m}/feature_catalog_L*.json") else TS
    return sorted(p for p in glob.glob(f"{d}/{m}/feature_catalog_L*.json") if not _os.path.basename(p).startswith("._"))


# features as gene ids, with the filter of hypothesis_trrust2.py (top 10, upper case, no Ensembl ids, >= 2 genes)
gid, FEAT = {}, {}
t0 = time.time()
for m in ORDER:
    rows = []
    for p in catalog_files(m):
        for f in json.load(open(p))["features"].values():
            s = sorted({str(x).upper() for x in f.get("top_genes", [])[:TOPG] if not str(x).upper().startswith("ENSG")})
            if len(s) < 2:
                continue
            r = [gid.setdefault(g, len(gid)) for g in s]
            rows.append(r + [-1] * (TOPG - len(r)))
    FEAT[m] = np.array(rows, np.int32)
    print(f"{m}: {len(rows)} features [{time.time() - t0:.0f}s]", flush=True)


def graph(feat):
    """pooled co-firing graph: unique pair keys lo*2^21+hi with weight >= W0"""
    out = []
    for i0 in range(0, len(feat), 50000):
        f = feat[i0:i0 + 50000].astype(np.int64)
        a, b = f[:, COMBOS[:, 0]], f[:, COMBOS[:, 1]]
        ok = (a >= 0) & (b >= 0)
        out.append(np.minimum(a, b)[ok] * SHIFT + np.maximum(a, b)[ok])
    k = np.concatenate(out) if out else np.empty(0, np.int64)
    u, c = np.unique(k, return_counts=True)
    return u[c >= W0]


def config_shuffle_keys(keys, rng):
    """the configuration null of hypothesis_trrust2.py (config_shuffle) on integer pair keys"""
    stubs = np.concatenate([keys // SHIFT, keys % SHIFT]).astype(np.int32)
    rng.shuffle(stubs)
    h, t = stubs[::2], stubs[1::2]
    ok = h != t
    h, t = h[ok], t[ok]
    return np.unique(np.minimum(h, t).astype(np.int64) * SHIFT + np.maximum(h, t).astype(np.int64))


G5 = {m: graph(FEAT[m]) for m in ORDER}
TR = json.load(open(f"{G}/trrust_edges.json")); truth = set()
for a, bs in TR.items():
    for b in bs:
        u, v = sorted((str(a).upper(), str(b).upper()))
        if u != v: truth.add((u, v))
tkey = np.array(sorted({min(gid[a], gid[b]) * SHIFT + max(gid[a], gid[b]) for a, b in truth if a in gid and b in gid}), np.int64)

out = {"n_draws": NDRAW, "seed": SEED, "null": "configuration model of hypothesis_trrust2.py, each model rewired on its own in every draw",
       "passing_models": PASSED, "sets": {}}
for label, models in (("all_ten", ORDER), ("passing", PASSED)):
    allk = np.concatenate([G5[m] for m in models if len(G5[m])])
    kk, vv = np.unique(allk, return_counts=True)
    ist = np.isin(kk, tkey)
    obs = {k: (int((vv >= k).sum()), int(ist[vv >= k].sum())) for k in (1, 2)}
    rng = np.random.default_rng(SEED)
    prec = {1: [], 2: []}; nh = {1: [], 2: []}; npair = {1: [], 2: []}
    for d in range(NDRAW):
        allr = np.concatenate([config_shuffle_keys(G5[m], rng) for m in models if len(G5[m])])
        k3, v3 = np.unique(allr, return_counts=True)
        i3 = np.isin(k3, tkey)
        for k in (1, 2):
            sel = v3 >= k
            s = int(sel.sum()); h = int(i3[sel].sum())
            prec[k].append(h / max(s, 1)); nh[k].append(h); npair[k].append(s)
        if (d + 1) % 500 == 0:
            print(f"  {label} {d + 1}/{NDRAW} [{time.time() - t0:.0f}s]", flush=True)
    res = {}
    for k in (1, 2):
        pc = np.array(prec[k]); n, h = obs[k]; po = h / n
        se = pc.std(ddof=1) / np.sqrt(len(pc))
        blocks = {}
        for B in (100, 200):
            nb = len(pc) // B
            f = [po / pc[i * B:(i + 1) * B].mean() if pc[i * B:(i + 1) * B].mean() > 0 else None for i in range(nb)]
            fv = np.array([x for x in f if x is not None])
            blocks[f"blocks_of_{B}"] = {"n_blocks": nb, "n_blocks_null_zero": int(sum(x is None for x in f)),
                                         "fold_min": round(float(fv.min()), 2), "fold_median": round(float(np.median(fv)), 2),
                                         "fold_max": round(float(fv.max()), 2),
                                         "fold_p05": round(float(np.percentile(fv, 5)), 2), "fold_p95": round(float(np.percentile(fv, 95)), 2)}
        res[k] = {"n_pairs": n, "hits": h, "precision": round(po, 6),
                  "null_precision_mean": round(float(pc.mean()), 8), "null_precision_se": round(float(se), 9),
                  "null_hits_mean": round(float(np.mean(nh[k])), 4), "null_pairs_mean": round(float(np.mean(npair[k])), 1),
                  "null_draws_with_zero_hits": int((np.array(nh[k]) == 0).sum()),
                  "fold": round(po / pc.mean(), 2),
                  "fold_approx_95ci_from_se": [round(po / (pc.mean() + 1.96 * se), 2), round(po / max(pc.mean() - 1.96 * se, 1e-12), 2)],
                  "p_emp": round((int((pc >= po).sum()) + 1) / (len(pc) + 1), 5), **blocks}
    res["null_free_precision_ratio_k2_over_k1"] = round(res[2]["precision"] / res[1]["precision"], 3)
    out["sets"][label] = res
    print(label, json.dumps({k: res[k]["fold"] for k in (1, 2)}), flush=True)
json.dump(out, open(f"{C}/agreement_curve.json", "w"), indent=1)
print(f"\n==> agreement_curve.json [{time.time() - t0:.0f}s]", flush=True)
