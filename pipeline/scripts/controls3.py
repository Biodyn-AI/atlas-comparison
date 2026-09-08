#!/usr/bin/env python
"""Round-3 residual controls (local), merged into controls.json.
  #6 p-calibration / false-positive rate: replace every feature's top genes with RANDOM background genes
     (same count) and run the SAME annotator. A calibrated test should then annotate ~nobody (post-BH) and
     give raw p<0.05 at ~the nominal 5%. Reports real vs random-gene annotation rate + raw-p FPR per model.
  #4 capacity normalisation: concepts-per-1000-features and corr(n_concepts, var_explained) — show how much of
     the cross-model concept-count spread is capacity/quality rather than biology.
    python scripts/controls3.py
"""
from __future__ import annotations
import json, glob, os
import numpy as np
from scipy.stats import hypergeom
from collections import defaultdict, Counter

BASE = "/Users/annaantipova/Desktop/biomech"
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"; TS = f"{BASE}/outputs/atlas/ts3_out"
MIN, MAX, ALPHA, TOP = 5, 500, 0.05, 5
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
idx = defaultdict(list); size = {}
for t, genes in gs.items():
    size[t] = len(genes)
    for g in genes:
        idx[g].append(t)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.arange(1, n + 1)
    q = np.minimum.accumulate((p[o] * n / r)[::-1])[::-1]; out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


def annotate(feat_gene_lists):
    """feat_gene_lists: list of top-gene lists. Returns (annot_rate, raw_p_lt_05_fraction)."""
    A = []; TG = []; LG = []; FI = []
    for fi, genes in enumerate(feat_gene_lists):
        g = set(str(x).upper() for x in genes[:TOP]) & bg
        if len(g) < 3:
            continue
        cand = Counter()
        for gene in g:
            for k in idx.get(gene, ()):
                cand[k] += 1
        for k, a in cand.items():
            if a < 2:
                continue
            A.append(a); TG.append(size[k]); LG.append(len(g)); FI.append(fi)
    n_feat = len(feat_gene_lists)
    if not A:
        return 0.0, 0.0
    p = hypergeom.sf(np.array(A) - 1, bgn, np.array(TG), np.array(LG))
    q = bh(p)
    ann_feats = set(FI[i] for i in np.where(q <= ALPHA)[0])
    raw_fpr = float((p < 0.05).mean())          # only meaningful under the random-gene null
    return round(100 * len(ann_feats) / n_feat, 1), round(raw_fpr, 4)


def orig_dir(m):
    d = f"{ALLCAT}/{m}"
    return d if glob.glob(f"{d}/feature_catalog_L*.json") else f"{TS}/{m}"


MODELS = ["AIDO", "Geneformer", "C2S", "UCE", "Tahoe"]   # representative subset (fast)
pcal = {}
for m in MODELS:
    cats = sorted(glob.glob(f"{orig_dir(m)}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])
    cat = json.load(open(cats[len(cats) // 2]))
    feats = [f["top_genes"] for f in cat["features"].values()]
    real_rate, _ = annotate(feats)
    # random-gene null: each feature gets 5 random background genes
    rand = [rng.choice(bg_list, TOP, replace=False).tolist() for _ in feats]
    rand_rate, raw_fpr = annotate(rand)
    pcal[m] = {"real_annot_rate": real_rate, "random_gene_annot_rate": rand_rate, "raw_p_lt05_under_null": raw_fpr}
    print(f"#6 {m}: real {real_rate}% vs random-gene {rand_rate}% (post-BH) · raw p<0.05 under null {raw_fpr} (nominal 0.05)", flush=True)

# #4 capacity
mat = json.load(open(f"{C}/matrix_alllayer.json"))
atlas = json.load(open(f"{C}/atlas_full_notf.json"))
models = atlas["models"]
nconc = {m: len(mat[m]["term_count"]) for m in models}
dsae = json.load(open(f"{C}/controls.json"))["capacity"]["d_sae"]
ve = {m: mat[m].get("var_explained", 0) for m in models}
per1k = {m: round(1000 * nconc[m] / max(dsae.get(m, 1), 1), 1) for m in models}
cap = {"concepts_per_1k_feat": per1k,
       "var_explained": ve,
       "corr_nconcepts_varexplained": round(float(np.corrcoef([nconc[m] for m in models], [ve[m] for m in models])[0, 1]), 3),
       "var_explained_range": [round(min(ve.values()), 3), round(max(ve.values()), 3)]}
print(f"#4 concepts/1k range {min(per1k.values())}-{max(per1k.values())} · VarExpl {cap['var_explained_range']} · corr(nconc,VE) {cap['corr_nconcepts_varexplained']}", flush=True)

ctl = json.load(open(f"{C}/controls.json"))
ctl["pcalib"] = pcal; ctl["capacity_norm"] = cap
json.dump(ctl, open(f"{C}/controls.json", "w"), indent=1)
print("==> controls.json (#6 pcalib + #4 capacity_norm)", flush=True)
