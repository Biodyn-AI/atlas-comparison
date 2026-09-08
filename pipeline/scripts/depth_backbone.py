#!/usr/bin/env python
"""Backbone robustness to LAYER CHOICE (reviewer pre-empt: is the ~78-concept >=8/10 backbone a mid-layer artefact?).
Recompute the calibrated backbone (top10, a>=3, GO/Reactome/KEGG <=200, no PPI, BH<0.05) and its random-gene null at
depth fractions f in {0, .25, .5, .75, 1.0}, matching each model's layer by index round(f*(n-1)). Report >=7/8/9/10
real vs null fold at each depth. Local (catalogs + gene sets), reuses recalibrate_robust machinery.
    -> outputs/atlas/comparative/depth_backbone.json
"""
from __future__ import annotations
import json, glob
import numpy as np
from scipy.stats import hypergeom
from collections import defaultdict, Counter

BASE = "/Users/annaantipova/Desktop/biomech"
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"; TS = f"{BASE}/outputs/atlas/ts3_out"
ALPHA, TOP, AMIN, MX, NPERM = 0.05, 10, 3, 200, 20
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]
THRESH = [7, 8, 9, 10]
rng = np.random.default_rng(0)

RAW = {}
for src in ("GO_BP", "KEGG", "Reactome"):
    for t, g in json.load(open(f"{G}/{src}_gene_sets.json")).items():
        if 5 <= len(g) <= MX:
            RAW[f"{src}:{t}"] = set(x.upper() for x in g)
BG = set(x.upper() for x in json.load(open(f"{G}/background.json"))); BGN = len(BG); BG_ARR = np.array(sorted(BG))
idx = defaultdict(list); size = {}
for t, genes in RAW.items():
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
    p = hypergeom.sf(np.array(A) - 1, BGN, np.array(TG), np.array(LG))
    return set(TERM[i] for i in np.where(bh(p) <= ALPHA)[0])


def cats_of(m):
    d = ALLCAT if glob.glob(f"{ALLCAT}/{m}/feature_catalog_L*.json") else TS
    return sorted(glob.glob(f"{d}/{m}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])

CATS = {m: cats_of(m) for m in ORDER}


def feats_at(m, f):
    cats = CATS[m]; n = len(cats); i = int(round(f * (n - 1)))
    cat = json.load(open(cats[i]))
    return [x["top_genes"] for x in cat["features"].values()], cat["layer"]


out = {"annotator": f"top{TOP} a>={AMIN} GO/Re/KEGG<= {MX} no-PPI (calibrated)", "n_perm": NPERM, "depths": []}
for f in FRACS:
    real = {}; nfeat = {}; used = {}
    for m in ORDER:
        fl, L = feats_at(m, f); nfeat[m] = len(fl); used[m] = L
        real[m] = annotate(fl)
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
            rs = annotate([BG_ARR[rng.integers(0, BGN, TOP)].tolist() for _ in range(nfeat[m])])
            for t in rs:
                cc[t] += 1
        for k in THRESH:
            null_ge[k].append(sum(1 for v in cc.values() if v >= k))
    row = {"frac": f, "layers_used": used, "tiers": {}}
    for k in THRESH:
        nm_ = float(np.mean(null_ge[k]))
        row["tiers"][f">={k}"] = {"real": real_ge[k], "null": round(nm_, 1), "fold": round(real_ge[k] / max(nm_, 0.3), 1)}
    row["ge8_terms"] = ge8_terms
    out["depths"].append(row)
    t = row["tiers"]
    print(f"f={f:<4} " + " ".join(f">={k}:{t[f'>={k}']['real']}v{t[f'>={k}']['null']}({t[f'>={k}']['fold']}x)" for k in THRESH), flush=True)

# stability of the >=8 backbone membership across depths (Jaccard between mid and each other depth)
mid_terms = set(next(r for r in out["depths"] if r["frac"] == 0.5)["ge8_terms"])
for r in out["depths"]:
    s = set(r["ge8_terms"])
    j = len(s & mid_terms) / max(len(s | mid_terms), 1)
    r["jaccard_vs_mid"] = round(j, 3)
    print(f"  f={r['frac']}: >=8 n={len(s)}  Jaccard-vs-mid={r['jaccard_vs_mid']}", flush=True)

json.dump(out, open(f"{C}/depth_backbone.json", "w"), indent=1)
print("==> depth_backbone.json", flush=True)
