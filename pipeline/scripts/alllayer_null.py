#!/usr/bin/env python
"""Reviewer-critical control: the mid-layer random-gene null exists (random_null.py), but the retracted headline
("~2,000 concepts shared by all ten models") is an ALL-LAYER union. This runs the same random-gene null on the
ALL-LAYER core: per model, union the permissive concepts over EVERY layer, intersect across the 10 models, and
compare that real core to a random-gene null built exactly the same way (random genes, per-layer feature counts).
    python scripts/alllayer_null.py   -> outputs/atlas/comparative/alllayer_null.json
"""
from __future__ import annotations
import json, glob, sys
import numpy as np
from scipy.stats import hypergeom
from collections import defaultdict, Counter

BASE = "/Users/annaantipova/Desktop/biomech"
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"; TS = f"{BASE}/outputs/atlas/ts3_out"
MIN, MAX, ALPHA, TOP = 5, 500, 0.05, 5          # permissive annotator, identical to random_null.py
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 20
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
rng = np.random.default_rng(0)

gs = {}
for nm in ("GO_BP", "KEGG", "Reactome"):
    for t, g in json.load(open(f"{G}/{nm}_gene_sets.json")).items():
        s = set(x.upper() for x in g)
        if MIN <= len(s) <= MAX:
            gs[f"{nm}:{t}"] = s
for nm, fn in (("TRRUST", "trrust_edges.json"), ("STRING", "string_edges.json")):
    for t, g in json.load(open(f"{G}/{fn}")).items():
        s = set(x.upper() for x in g)
        if MIN <= len(s) <= MAX:
            gs[f"{nm}:{t}"] = s
bg = set(x.upper() for x in json.load(open(f"{G}/background.json")))
bg |= set().union(*[v for k, v in gs.items() if k.startswith("STRING:")])
bg |= set(k.split(":", 1)[1].upper() for k in gs if k.startswith("STRING:"))
bg_list = np.array(sorted(bg)); bgn = len(bg)
size = {t: len(genes) for t, genes in gs.items()}

# ---- sparse gene x term incidence (same annotator, ~50x faster than per-feature Python loops) ----
import scipy.sparse as sp
GENE_IX = {g: i for i, g in enumerate(bg_list.tolist())}
TERMS = sorted(gs.keys()); TERM_IX = {t: i for i, t in enumerate(TERMS)}
TERM_SIZE = np.array([size[t] for t in TERMS], dtype=np.int64)
_r, _c = [], []
for t, genes in gs.items():
    ti = TERM_IX[t]
    for g in genes:
        gi = GENE_IX.get(g)
        if gi is not None:
            _r.append(gi); _c.append(ti)
M = sp.csr_matrix((np.ones(len(_r), dtype=np.int32), (_r, _c)), shape=(len(bg_list), len(TERMS)))
print(f"incidence matrix: {M.shape[0]} genes x {M.shape[1]} terms, {M.nnz} memberships", flush=True)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.arange(1, n + 1)
    q = np.minimum.accumulate((p[o] * n / r)[::-1])[::-1]; out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


def annotate(feat_lists):
    """Identical annotator to random_null.py (top-TOP genes, overlap>=2, hypergeometric + BH<ALPHA),
    computed as a sparse product instead of per-feature Python loops."""
    rows = []; cols = []; lg = np.zeros(len(feat_lists), dtype=np.int64)
    for i, genes in enumerate(feat_lists):
        g = set(str(x).upper() for x in genes[:TOP]) & bg
        if len(g) < 3:
            continue
        lg[i] = len(g)
        for x in g:
            rows.append(i); cols.append(GENE_IX[x])
    if not rows:
        return set()
    F = sp.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)),
                      shape=(len(feat_lists), len(bg_list)))
    A = (F @ M).tocoo()
    keep = A.data >= 2
    if not keep.any():
        return set()
    a = A.data[keep].astype(np.int64); fi = A.row[keep]; ti = A.col[keep]
    q = bh(hypergeom.sf(a - 1, bgn, TERM_SIZE[ti], lg[fi]))
    return set(TERMS[j] for j in np.unique(ti[q <= ALPHA]))


def orig_dir(m):
    d = f"{ALLCAT}/{m}"
    return d if glob.glob(f"{d}/feature_catalog_L*.json") else f"{TS}/{m}"


# ---- real: per-model union over EVERY layer ----
layer_feats = {}   # model -> list of per-layer feature-gene-lists
real = {}
for m in ORDER:
    cats = sorted(glob.glob(f"{orig_dir(m)}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])
    per_layer = []
    acc = set()
    for cp in cats:
        feats = [f["top_genes"] for f in json.load(open(cp))["features"].values()]
        per_layer.append(len(feats))
        acc |= annotate(feats)
    layer_feats[m] = per_layer
    real[m] = acc
    print(f"  {m}: {len(cats)} layers, {sum(per_layer)} features -> {len(acc)} concepts (all-layer union)", flush=True)

real_core = set.intersection(*real.values())
print(f"\nREAL all-layer core (all {len(ORDER)} models): {len(real_core)}", flush=True)

# ---- random-gene null, built the same way ----
null_cores = []
for it in range(NPERM):
    rsets = {}
    for m in ORDER:
        acc = set()
        for nfeat in layer_feats[m]:
            draws = bg_list[rng.integers(0, bgn, (nfeat, TOP))]
            acc |= annotate(draws.tolist())
        rsets[m] = acc
    nc = len(set.intersection(*rsets.values()))
    null_cores.append(nc)
    print(f"  perm {it+1}/{NPERM}: null all-layer core {nc} (mean so far {np.mean(null_cores):.0f})", flush=True)

nc = np.array(null_cores, float)
z = (len(real_core) - nc.mean()) / (nc.std() or 1)
out = {"n_perm": NPERM, "scope": "all-layer union, permissive annotator (top-5, 5 DBs)", "models": ORDER,
       "layers_per_model": {m: len(layer_feats[m]) for m in ORDER},
       "real_core": len(real_core),
       "null_core_mean": round(float(nc.mean()), 1), "null_core_sd": round(float(nc.std()), 1),
       "null_core_min": int(nc.min()), "null_core_max": int(nc.max()),
       "core_z": round(float(z), 1),
       "core_fold_over_null": round(len(real_core) / max(nc.mean(), 1), 2),
       "per_model_real_concepts": {m: len(real[m]) for m in ORDER}}
json.dump(out, open(f"{C}/alllayer_null.json", "w"), indent=1)
print("\n=== ALL-LAYER RANDOM-GENE NULL ===")
print(f"REAL {out['real_core']} vs NULL {out['null_core_mean']}±{out['null_core_sd']} "
      f"(min {out['null_core_min']}, max {out['null_core_max']}) -> z={out['core_z']}, {out['core_fold_over_null']}x")
print("==> alllayer_null.json", flush=True)
