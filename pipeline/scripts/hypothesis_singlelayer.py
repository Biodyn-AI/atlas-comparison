#!/usr/bin/env python
"""Does the held-out regulatory recovery survive when the co-firing graph is built from ONE layer?

Section 3.4 pools features over every layer of a model. Design choice (iii) in the Discussion forbids
cross-layer aggregation of *concept counts*, and a reviewer may ask whether the co-firing graph is the
same objection in disguise. This restricts each model to its mid-depth catalogue and repeats the test.

Provenance. The manuscript's "6.1x within a single layer" was produced by a throwaway script in /tmp
that no longer existed; the number therefore had no source a reader could run. This is that script,
recovered from the session log and made a proper part of the pipeline. Two things it did silently are
now explicit:
  * it used a recurrence threshold of W >= 2, not the W >= 5 of the pooled analysis, because one
    layer has a fifth to a thirtieth of the features and W >= 5 leaves almost nothing. Both
    thresholds are computed and stored so the text can say which it quotes and why.
  * it scored the five models that pass at full size, so it is like-for-like with the five-model
    pooled baseline (5.3x, hypothesis_final.json), not with the all-ten headline (4.8x).
Conventions match hypothesis_trrust2.py exactly (top-10 truncated then Ensembl ids dropped, undirected
TRRUST truth, configuration-model null with seed 0).
    python scripts/hypothesis_singlelayer.py [NPERM]  -> outputs/atlas/comparative/hypothesis_singlelayer.json
"""
from __future__ import annotations

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import glob, itertools, json, sys
import numpy as np
from collections import defaultdict

BASE = _B
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"; TS = f"{BASE}/outputs/atlas/ts3_out"
PASSED = json.load(open(f"{C}/hypothesis_final.json"))["models_validated"]
TOPG = 10
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200
THRESHOLDS = (2, 5)          # 5 is the pooled analysis's threshold; 2 is what one layer can support
rng = np.random.default_rng(0)

TRUTH = set()
for a, bs in json.load(open(f"{G}/trrust_edges.json")).items():
    for b in bs:
        u, v = sorted((a.upper(), b.upper()))
        if u != v:
            TRUTH.add((u, v))


def midfile(m):
    d = ALLCAT if glob.glob(f"{ALLCAT}/{m}/feature_catalog_L*.json") else TS
    c = sorted(glob.glob(f"{d}/{m}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])
    return c[len(c) // 2]


def pair_counts(m):
    cnt = defaultdict(int); nfeat = 0
    for f in json.load(open(midfile(m)))["features"].values():
        s = sorted({str(x).upper() for x in f.get("top_genes", [])[:TOPG]
                    if not str(x).upper().startswith("ENSG")})
        if len(s) < 2:
            continue
        nfeat += 1
        for a, b in itertools.combinations(s, 2):
            cnt[(a, b)] += 1
    return cnt, nfeat


COUNTS = {m: pair_counts(m) for m in PASSED}
GENES = sorted({g for m in PASSED for e in COUNTS[m][0] for g in e}); GID = {g: i for i, g in enumerate(GENES)}
TK = np.array(sorted({(lambda i, j: i * (1 << 21) + j)(*sorted((GID[a], GID[b])))
                      for a, b in TRUTH if a in GID and b in GID}), np.int64)


def shuf(E):
    a = np.fromiter((GID[e[0]] for e in E), np.int32, len(E)); b = np.fromiter((GID[e[1]] for e in E), np.int32, len(E))
    s = np.concatenate([a, b]); rng.shuffle(s); h, t = s[::2], s[1::2]; ok = h != t; h, t = h[ok], t[ok]
    return np.unique(np.minimum(h, t).astype(np.int64) * (1 << 21) + np.maximum(h, t).astype(np.int64))


def keys_of(E):
    k = np.empty(len(E), np.int64)
    for n, (a, b) in enumerate(E):
        i, j = sorted((GID[a], GID[b])); k[n] = i * (1 << 21) + j
    return k


out = {"design": "held-out TRRUST recovery from the MID layer only, five models that pass at full size",
       "models": PASSED, "layer": "mid (index n//2 of each model's catalogues)", "top_genes_per_feature": TOPG,
       "n_perm": NPERM, "null": "configuration model, exact co-firing degree preserved, seed 0",
       "features_per_model": {m: COUNTS[m][1] for m in PASSED}, "variants": {}}

for W in THRESHOLDS:
    PRED = {m: {k: c for k, c in COUNTS[m][0].items() if c >= W} for m in PASSED}
    print(f"\n=== W >= {W} ===")
    per = {}
    for m in PASSED:
        E = list(PRED[m])
        if not E:
            per[m] = {"n_pairs": 0, "hits": 0, "null_mean": 0.0, "null_sd": 0.0, "fold": None, "p_emp": 1.0}
            print(f"  {m:<12} no pairs"); continue
        obs = int(np.isin(keys_of(E), TK).sum())
        nul = np.array([int(np.isin(shuf(E), TK).sum()) for _ in range(NPERM)], float)
        per[m] = {"n_pairs": len(E), "hits": obs, "null_mean": round(float(nul.mean()), 2),
                  "null_sd": round(float(nul.std()), 2),
                  "fold": round(obs / nul.mean(), 2) if nul.mean() > 0 else None,
                  "p_emp": round((int((nul >= obs).sum()) + 1) / (NPERM + 1), 5)}
        print(f"  {m:<12} {len(E):>6} pairs {obs:>4} hits vs {nul.mean():>6.2f}±{nul.std():<5.2f} "
              f"fold {per[m]['fold']}  p={per[m]['p_emp']}", flush=True)
    allp = defaultdict(int)
    for m in PASSED:
        for e in PRED[m]:
            allp[e] += 1
    curve = {}
    if allp:
        kk = keys_of(list(allp)); nm = np.fromiter(allp.values(), np.int32, len(allp)); ist = np.isin(kk, TK)
        nulls = {1: [], 2: []}
        for _ in range(NPERM):
            c = defaultdict(int)
            for m in PASSED:
                for x in shuf(list(PRED[m])):
                    c[int(x)] += 1
            a1 = np.fromiter(c.keys(), np.int64, len(c)); v1 = np.fromiter(c.values(), np.int32, len(c)); i1 = np.isin(a1, TK)
            for K in (1, 2):
                s = v1 >= K; nulls[K].append(float(i1[s].sum()) / max(int(s.sum()), 1))
        for K in (1, 2):
            s = nm >= K; n = int(s.sum()); h = int(ist[s].sum()); pr = h / max(n, 1); nc = np.array(nulls[K])
            curve[str(K)] = {"n_pairs": n, "hits": h, "precision": round(pr, 6),
                             "null_precision": round(float(nc.mean()), 7),
                             "fold": round(pr / nc.mean(), 2) if nc.mean() > 0 else None,
                             "p_emp": round((int((nc >= pr).sum()) + 1) / (NPERM + 1), 5)}
            print(f"  k>={K}: {n:>7} pairs, {h:>4} hits, {pr:.5f} vs {nc.mean():.7f} -> "
                  f"{curve[str(K)]['fold']}x  p={curve[str(K)]['p_emp']}", flush=True)
    hits = [per[m]["hits"] for m in PASSED]
    out["variants"][f"W{W}"] = {"min_features_per_pair": W, "per_model": per, "cross_model_curve": curve,
                                "hits_min": min(hits), "hits_max": max(hits),
                                "n_models_significant": sum(1 for m in PASSED if per[m]["p_emp"] <= 0.05)}

json.dump(out, open(f"{C}/hypothesis_singlelayer.json", "w"), indent=1)
print("\n==> hypothesis_singlelayer.json", flush=True)
