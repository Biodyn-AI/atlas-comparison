#!/usr/bin/env python
"""Distinct concepts per model across ALL depth-matched layers (union), using the same 5-DB STRING
annotator as the atlas (top-5, Fisher 'greater' + BH<0.05 vs GO_BP/Reactome/KEGG + TRRUST + STRING).
Complements the mid-layer n_concepts in coverage. Writes alllayer_concepts.json.
    python scripts/alllayer_concepts.py
"""
from __future__ import annotations
import json, glob
import numpy as np
from collections import defaultdict, Counter
from scipy.stats import fisher_exact

BASE = "/Users/annaantipova/Desktop/biomech"
G = f"{BASE}/outputs/atlas/genesets"; TS = f"{BASE}/outputs/atlas/ts3_out"; C = f"{BASE}/outputs/atlas/comparative"
MIN, MAX, ALPHA, TOP = 5, 500, 0.05, 5
import sys
MODELS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT"]
MERGE = "--merge" in sys.argv  # keep existing entries, only (re)compute MODELS


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.arange(1, n + 1)
    q = np.minimum.accumulate((p[o] * n / r)[::-1])[::-1]; out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


print("loading 5 DBs...", flush=True)
gs = {nm: {t: set(g) for t, g in json.load(open(f"{G}/{nm}_gene_sets.json")).items() if MIN <= len(g) <= MAX}
      for nm in ("GO_BP", "KEGG", "Reactome")}
tr = {t: set(v) for t, v in json.load(open(f"{G}/trrust_edges.json")).items() if MIN <= len(v) <= MAX}
st = {t: set(v) for t, v in json.load(open(f"{G}/string_edges.json")).items() if MIN <= len(v) <= MAX}
bg = set(json.load(open(f"{G}/background.json"))); bg |= set().union(*st.values()) | set(st); bgn = len(bg)
idx = defaultdict(list); size = {}
for s, d in list(gs.items()) + [("TRRUST", tr), ("STRING", st)]:
    for t, genes in d.items():
        k = (s, t); size[k] = len(genes)
        for g in genes:
            idx[g].append(k)


def annotate(feats):
    recs = []
    for fid, genes in feats.items():
        g = set(genes) & bg
        if len(g) < 3:
            continue
        cand = Counter()
        for gene in g:
            for k in idx.get(gene, ()):
                cand[k] += 1
        for k, a in cand.items():
            if a < 2:
                continue
            tg = size[k]; b = len(g) - a; c = tg - a; d = bgn - a - b - c
            orr, p = fisher_exact([[a, b], [c, d]], alternative="greater")
            recs.append((fid, f"{k[0]}:{k[1]}", float(p)))
    if not recs:
        return set()
    q = bh([r[2] for r in recs])
    return {recs[i][1] for i in range(len(recs)) if q[i] <= ALPHA}


out = json.load(open(f"{C}/alllayer_concepts.json")) if MERGE and glob.glob(f"{C}/alllayer_concepts.json") else {}
for m in MODELS:
    cats = sorted(glob.glob(f"{TS}/{m}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])
    if len(cats) > 6:  # depth-match to 5 layers (0/25/50/75/100%) for parity with the 5-layer models
        idxs = [round(x * (len(cats) - 1) / 4) for x in range(5)]
        cats = [cats[i] for i in sorted(set(idxs))]
        print(f"{m}: {len(cats)} depth-matched layers from all-layers catalogs", flush=True)
    union, per_layer = set(), {}
    for cp in cats:
        cat = json.load(open(cp)); L = cat["layer"]
        feats = {fid: [str(x).upper() for x in f["top_genes"]][:TOP] for fid, f in cat["features"].items()}
        cons = annotate(feats)
        per_layer[str(L)] = len(cons); union |= cons
    mid = sorted(per_layer, key=lambda x: int(x))[len(per_layer) // 2]
    out[m] = {"per_layer": per_layer, "mid_layer": int(mid), "mid_concepts": per_layer[mid],
              "all_layer_union": len(union)}
    print(f"{m}: mid(L{mid})={per_layer[mid]} | all-layer union={len(union)} | per-layer={per_layer}", flush=True)
json.dump(out, open(f"{C}/alllayer_concepts.json", "w"))
print(f"\n==> {C}/alllayer_concepts.json", flush=True)
