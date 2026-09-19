#!/usr/bin/env python
"""The names of the calibrated backbone concepts, all of them.

The site listed the backbone from controls.recalibration.final.shared_ge8_terms, which was cut
to 40 entries when it was written, so the page titled "78 shared concepts" could only show 40.
This recomputes the real backbone under the headline calibrated annotator (top-10 genes, >=3 in a
curated GO/Reactome/KEGG set of <=200 genes, no PPI, BH<0.05, mid layer) and stores every concept
shared by >=7, >=8, >=9 and 10 models, with its source database. No null is run here; the fold and
p-value come from stats_final / degree_null (250 permutations), which use the same annotator.
    python scripts/backbone_terms.py -> outputs/atlas/comparative/backbone_terms.json
"""
from __future__ import annotations

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import glob, json
import numpy as np
from scipy.stats import hypergeom
from collections import defaultdict, Counter

C = f"{_B}/outputs/atlas/comparative"; G = f"{_B}/outputs/atlas/genesets"
ALLCAT = f"{_B}/outputs/atlas/alllayer_cat"; TS = f"{_B}/outputs/atlas/ts3_out"
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
TOP, AMIN, MAXSET, ALPHA = 10, 3, 200, 0.05

RAW = {nm: {t: set(x.upper() for x in g) for t, g in json.load(open(f"{G}/{nm}_gene_sets.json")).items()}
       for nm in ("GO_BP", "KEGG", "Reactome")}
BG = set(x.upper() for x in json.load(open(f"{G}/background.json"))); BGN = len(BG)
idx, size = defaultdict(list), {}
for src in RAW:
    for t, genes in RAW[src].items():
        if 5 <= len(genes) <= MAXSET:
            k = f"{src}:{t}"; size[k] = len(genes)
            for g in genes:
                idx[g].append(k)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.arange(1, n + 1)
    q = np.minimum.accumulate((p[o] * n / r)[::-1])[::-1]; out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


def annotate(feat_lists):
    A, TG, LG, TERM = [], [], [], []
    for genes in feat_lists:
        g = set(str(x).upper() for x in genes[:TOP]) & BG
        if len(g) < 3:
            continue
        cand = Counter()
        for gene in g:
            for k in idx.get(gene, ()):
                cand[k] += 1
        for k, a in cand.items():
            if a >= AMIN:
                A.append(a); TG.append(size[k]); LG.append(len(g)); TERM.append(k)
    if not A:
        return set()
    p = hypergeom.sf(np.array(A) - 1, BGN, np.array(TG), np.array(LG))
    return set(TERM[i] for i in np.where(bh(p) <= ALPHA)[0])


def mid_feats(m):
    d = ALLCAT if glob.glob(f"{ALLCAT}/{m}/feature_catalog_L*.json") else TS
    cats = sorted(glob.glob(f"{d}/{m}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])
    return [f["top_genes"] for f in json.load(open(cats[len(cats) // 2]))["features"].values()]


cm = Counter()
for m in ORDER:
    for t in annotate(mid_feats(m)):
        cm[t] += 1
    print(f"  {m}: annotated", flush=True)

out = {"annotator": f"top{TOP} genes, >={AMIN} in a curated GO/Reactome/KEGG set <= {MAXSET} genes, no PPI, BH<{ALPHA}, mid layer",
       "n_models": len(ORDER), "tiers": {}}
for k in (7, 8, 9, 10):
    terms = sorted((t for t, v in cm.items() if v >= k), key=lambda t: (-cm[t], t))
    out["tiers"][f">={k}"] = [{"term": t.split(":", 1)[1], "source": t.split(":", 1)[0], "n_models": cm[t]} for t in terms]
    print(f">= {k} models: {len(terms)} concepts")
json.dump(out, open(f"{C}/backbone_terms.json", "w"), indent=1)
print("==> backbone_terms.json")
