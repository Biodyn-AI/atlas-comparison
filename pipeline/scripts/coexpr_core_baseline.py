#!/usr/bin/env python
"""Co-expression baseline for the shared backbone (calibrated annotator, mid layer).

Question: does a dictionary built from gene-gene co-expression alone, with no model, recover the
78-concept backbone? Each "feature" is a seed gene plus its 9 most correlated genes (Pearson on
log1p CP10k over the same 6,000 cells, genes detected in >= 10 cells), annotated with the exact
calibrated annotator of stats_final.py.
  seed-matched: one feature per live mid-layer feature of each model, seeded with that feature's
                top gene (keeps each model's gene choice, replaces its grouping by co-expression);
  model-free:   the same number of features per model, seeded with genes drawn uniformly from
                the detected genes (one draw per model, seeded by model index).
For each variant: per-model recall of the 78 backbone terms, and the >= 8/10 core of the ten
dictionaries compared with the models' 78.

    python scripts/coexpr_core_baseline.py -> outputs/atlas/comparative/coexpr_core_baseline.json
"""
from __future__ import annotations

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import glob
import json
from collections import Counter, defaultdict

import anndata as ad
import numpy as np
import scipy.sparse as sp
from scipy.stats import hypergeom

C = f"{_B}/outputs/atlas/comparative"; G = f"{_B}/outputs/atlas/genesets"
ALLCAT = f"{_B}/outputs/atlas/alllayer_cat"; TS = f"{_B}/outputs/atlas/ts3_out"
CORPUS = f"{_B}/external/perturb/tabula_3tissue_6k.h5ad"
ALPHA, TOP, AMIN, MX = 0.05, 10, 3, 200
ALL = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
K_CORE = 8

# ---- calibrated annotator, identical to stats_final.py ----
gs = {}
for src in ("GO_BP", "KEGG", "Reactome"):
    for t, g in json.load(open(f"{G}/{src}_gene_sets.json")).items():
        if 5 <= len(g) <= MX:
            gs[f"{src}:{t}"] = set(x.upper() for x in g)
BG = set(x.upper() for x in json.load(open(f"{G}/background.json"))); BGN = len(BG)
idx = defaultdict(list); size = {}
for t, genes in gs.items():
    size[t] = len(genes)
    for g in genes:
        idx[g].append(t)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.arange(1, n + 1)
    q = np.minimum.accumulate((p[o] * n / r)[::-1])[::-1]; out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


def annotate(feat_lists):
    A = []; TG = []; LG = []; TERM = []
    for genes in feat_lists:
        g = set(str(x).upper() for x in genes[:TOP]) & BG
        if len(g) < 3:
            continue
        cand = Counter()
        for gene in g:
            for k in idx.get(gene, ()):
                cand[k] += 1
        for k, a in cand.items():
            if a < AMIN:
                continue
            A.append(a); TG.append(size[k]); LG.append(len(g)); TERM.append(k)
    if not A:
        return set()
    q = bh(hypergeom.sf(np.array(A) - 1, BGN, np.array(TG), np.array(LG)))
    return set(TERM[i] for i in np.where(q <= ALPHA)[0])


def mid_catalog(m):
    root = ALLCAT if glob.glob(f"{ALLCAT}/{m}/feature_catalog_L*.json") else TS
    cats = sorted(glob.glob(f"{root}/{m}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])
    d = json.load(open(cats[len(cats) // 2]))
    return list(d["features"].values()), d["layer"]


def backbone_keys():
    d = json.load(open(f"{C}/backbone_terms.json"))
    return {f"{t['source']}:{t['term']}" for t in d["tiers"][">=8"]}


def core(sets, k):
    cnt = Counter(t for s in sets.values() for t in s)
    return {t for t, n in cnt.items() if n >= k}


def main():
    a = ad.read_h5ad(CORPUS)
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    X = X.astype(np.float64)
    lib = np.asarray(X.sum(1)).ravel()
    X = sp.diags(1e4 / np.maximum(lib, 1)) @ X
    X.data = np.log1p(X.data)
    det = np.asarray((X > 0).sum(0)).ravel() >= 10
    names = np.array([str(s).upper() for s in a.var["feature_name"].values])[det]
    Xd = X[:, det].toarray().astype(np.float32)
    Z = (Xd - Xd.mean(0)) / (Xd.std(0) + 1e-8); del Xd
    n = Z.shape[0]
    pos = {}
    for i, g in enumerate(names):
        pos.setdefault(g, i)
    print(f"corpus: {n} cells, {len(names)} genes detected in >= 10 cells", flush=True)

    def coexpr_features(seeds):
        cols = np.array([pos[s] for s in seeds])
        out = []
        for c0 in range(0, len(cols), 1000):
            cc = cols[c0:c0 + 1000]
            R = (Z.T @ Z[:, cc]) / n                       # genes x seeds
            R[cc, np.arange(len(cc))] = -np.inf             # the seed itself goes first, not twice
            top = np.argpartition(-R, TOP - 1, axis=0)[:TOP - 1]
            for j, c in enumerate(cc):
                nb = top[:, j][np.argsort(-R[top[:, j], j])]
                out.append([names[c]] + list(names[nb]))
        return out

    BB = backbone_keys()
    rng = np.random.default_rng(0)
    real, seedm, free, info = {}, {}, {}, {}
    for mi, m in enumerate(ALL):
        feats, layer = mid_catalog(m)
        live = [f for f in feats if f.get("max_activation", 0) > 0 and f["top_genes"]]
        real[m] = annotate([f["top_genes"] for f in feats])
        seeds = [str(f["top_genes"][0]).upper() for f in live if str(f["top_genes"][0]).upper() in pos]
        seedm[m] = annotate(coexpr_features(seeds))
        free_seeds = list(names[np.random.default_rng(mi).choice(len(names), len(seeds), replace=True)])
        free[m] = annotate(coexpr_features(free_seeds))
        info[m] = {"layer": layer, "features": len(feats), "live": len(live), "seeded": len(seeds),
                   "terms_model": len(real[m]), "terms_seed_matched": len(seedm[m]), "terms_model_free": len(free[m]),
                   "backbone_recall_model": len(real[m] & BB), "backbone_recall_seed_matched": len(seedm[m] & BB),
                   "backbone_recall_model_free": len(free[m] & BB)}
        print(m, info[m], flush=True)

    out = {"annotator": "top10 genes, >=3 in a curated GO/Reactome/KEGG set <= 200 genes, BH<0.05, mid layer",
           "coexpression": f"Pearson on log1p CP10k, {n} cells, {len(names)} genes detected in >= 10 cells; "
                           "feature = seed + 9 most correlated genes",
           "backbone_size": len(BB), "k": K_CORE, "per_model": info}
    for name, sets in (("model", real), ("seed_matched", seedm), ("model_free", free)):
        cs = core(sets, K_CORE)
        out[f"core_{name}"] = {"size": len(cs), "in_backbone": len(cs & BB),
                               "jaccard_with_backbone": round(len(cs & BB) / max(len(cs | BB), 1), 3),
                               "terms": sorted(cs)}
        print(f"core {name}: {len(cs)} terms, {len(cs & BB)} of the {len(BB)} backbone terms", flush=True)
    bb_cov = Counter()
    for s in seedm.values():
        for t in s & BB:
            bb_cov[t] += 1
    out["backbone_terms_seed_matched_models"] = {t: bb_cov.get(t, 0) for t in sorted(BB)}
    json.dump(out, open(f"{C}/coexpr_core_baseline.json", "w"), indent=1)
    print(f"==> {C}/coexpr_core_baseline.json", flush=True)


if __name__ == "__main__":
    main()
