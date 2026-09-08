#!/usr/bin/env python
"""Recompute the depth annotation-rate curves under the CALIBRATED annotator (top-10, >=3 in curated
GO/Reactome/KEGG <=200, no PPI), per layer, all 10 models -> depth_calibrated.json. Puts the depth panel on
the same strict annotator as the rest of the main text (no more 'permissive illustrative')."""
from __future__ import annotations
import json, glob
import numpy as np
from scipy.stats import hypergeom
from collections import defaultdict, Counter

BASE = "/Users/annaantipova/Desktop/biomech"
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"; TS = f"{BASE}/outputs/atlas/ts3_out"
ALPHA, TOP, AMIN, MX = 0.05, 10, 3, 200
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]

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


def rate(feats):
    A = []; LG = []; TG = []; FI = []
    for fid, f in feats.items():
        g = set(str(x).upper() for x in f["top_genes"][:TOP]) & BG
        if len(g) < 3:
            continue
        cand = Counter()
        for gene in g:
            for k in idx.get(gene, ()):
                cand[k] += 1
        for k, a in cand.items():
            if a < AMIN:
                continue
            A.append(a); TG.append(size[k]); LG.append(len(g)); FI.append(fid)
    if not A:
        return 0.0
    q = bh(hypergeom.sf(np.array(A) - 1, BGN, np.array(TG), np.array(LG)))
    ann = set(FI[i] for i in np.where(q <= ALPHA)[0])
    return len(ann)


def cats_of(m):
    d = ALLCAT if glob.glob(f"{ALLCAT}/{m}/feature_catalog_L*.json") else TS
    return sorted(glob.glob(f"{d}/{m}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])


out = {"annotator": f"top{TOP} a>={AMIN} GO/Re/KEGG<= {MX} no-PPI (calibrated)", "models": {}}
for m in ORDER:
    cats = cats_of(m)
    if not cats:
        continue
    curve = []
    n = len(cats)
    for i, cp in enumerate(cats):
        cat = json.load(open(cp)); nal = cat.get("n_alive", len(cat["features"]))
        r = rate(cat["features"])
        curve.append({"layer": cat["layer"], "x": round(i / max(n - 1, 1), 3),
                      "rate": round(100 * r / max(nal, 1), 1)})
    out["models"][m] = curve
    rr = [c["rate"] for c in curve]
    print(f"  {m}: {len(curve)} layers, calibrated rate {rr[0]}->{rr[-1]} (min {min(rr)}, max {max(rr)})", flush=True)

json.dump(out, open(f"{C}/depth_calibrated.json", "w"))
print("==> depth_calibrated.json", flush=True)
