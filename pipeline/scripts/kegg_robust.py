#!/usr/bin/env python
"""KEGG-copyright robustness (reviewer/editorial pre-empt): recompute the calibrated >=k/10 backbone with the full
curated set {GO_BP, KEGG, Reactome} vs a KEGG-DROPPED set {GO_BP, Reactome}, at the main config (top10 a3 max200),
real vs random-gene null (fold). Also count how many of the >=8 backbone concepts are KEGG-sourced. Reuses the exact
machinery of recalibrate_robust.py. -> outputs/atlas/comparative/kegg_robust.json
"""
from __future__ import annotations
import json, glob
import numpy as np
from scipy.stats import hypergeom
from collections import defaultdict, Counter

BASE = "/Users/annaantipova/Desktop/biomech"
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"; TS = f"{BASE}/outputs/atlas/ts3_out"
ALPHA, NPERM = 0.05, 20
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
N = len(ORDER); rng = np.random.default_rng(0)
TOP, AMIN, MX = 10, 3, 200
THRESH = [7, 8, 9, 10]

RAW_ALL = {nm: {t: set(x.upper() for x in g) for t, g in json.load(open(f"{G}/{nm}_gene_sets.json")).items()}
           for nm in ("GO_BP", "KEGG", "Reactome")}
BG = set(x.upper() for x in json.load(open(f"{G}/background.json"))); BGN = len(BG); BG_ARR = np.array(sorted(BG))


def index(sources):
    gs = {}
    for src in sources:
        for t, genes in RAW_ALL[src].items():
            if 5 <= len(genes) <= MX:
                gs[f"{src}:{t}"] = genes
    idx = defaultdict(list); size = {}
    for t, genes in gs.items():
        size[t] = len(genes)
        for g in genes:
            idx[g].append(t)
    return idx, size


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.arange(1, n + 1)
    q = np.minimum.accumulate((p[o] * n / r)[::-1])[::-1]; out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


def annotate(feat_lists, idx, size):
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
    A = np.array(A); TG = np.array(TG); LG = np.array(LG)
    p = hypergeom.sf(A - 1, BGN, TG, LG)
    keep = bh(p) <= ALPHA
    return set(TERM[i] for i in np.where(keep)[0])


def load_feats(m):
    d = f"{ALLCAT}/{m}"
    if not glob.glob(f"{d}/feature_catalog_L*.json"):
        d = f"{TS}/{m}"
    cats = sorted(glob.glob(f"{d}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])
    return [f["top_genes"] for f in json.load(open(cats[len(cats) // 2]))["features"].values()]


FEATS = {m: load_feats(m) for m in ORDER}; NFEAT = {m: len(FEATS[m]) for m in ORDER}


def run(sources):
    idx, size = index(sources)
    real = {m: annotate(FEATS[m], idx, size) for m in ORDER}
    cm = Counter()
    for m in ORDER:
        for t in real[m]:
            cm[t] += 1
    real_ge = {k: sum(1 for v in cm.values() if v >= k) for k in THRESH}
    ge8_terms = sorted([t for t, v in cm.items() if v >= 8])
    null_ge = {k: [] for k in THRESH}
    for _ in range(NPERM):
        cc = Counter()
        for m in ORDER:
            rs = annotate([BG_ARR[rng.integers(0, BGN, TOP)].tolist() for _ in range(NFEAT[m])], idx, size)
            for t in rs:
                cc[t] += 1
        for k in THRESH:
            null_ge[k].append(sum(1 for v in cc.values() if v >= k))
    row = {}
    for k in THRESH:
        nm_ = float(np.mean(null_ge[k]))
        row[f">={k}"] = {"real": real_ge[k], "null": round(nm_, 1), "fold": round(real_ge[k] / max(nm_, 0.3), 1)}
    return row, ge8_terms


row_all, ge8_all = run(("GO_BP", "KEGG", "Reactome"))
row_gr, ge8_gr = run(("GO_BP", "Reactome"))
kegg_in_ge8 = [t for t in ge8_all if t.startswith("KEGG:")]

out = {"config": "top10 a3 max200", "n_perm": NPERM,
       "full_GO_KEGG_Reactome": row_all, "dropKEGG_GO_Reactome": row_gr,
       "ge8_full": len(ge8_all), "ge8_dropKEGG": len(ge8_gr),
       "kegg_sourced_in_ge8": len(kegg_in_ge8),
       "kegg_terms_in_ge8": [t.split(":", 1)[1] for t in kegg_in_ge8]}
json.dump(out, open(f"{C}/kegg_robust.json", "w"), indent=1)
print("FULL {GO,KEGG,Reactome}:", {k: row_all[k] for k in [">=7", ">=8", ">=9", ">=10"]})
print("DROP-KEGG {GO,Reactome}:", {k: row_gr[k] for k in [">=7", ">=8", ">=9", ">=10"]})
print(f">=8 backbone: full={len(ge8_all)}  dropKEGG={len(ge8_gr)}  KEGG-sourced in full's >=8 = {len(kegg_in_ge8)}")
print("==> kegg_robust.json")
