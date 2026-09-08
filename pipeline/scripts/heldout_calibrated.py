#!/usr/bin/env python
"""#1 — does the CALIBRATED backbone replicate on held-out cells? Re-annotate original + held-out catalogs with
the calibrated annotator (top-10, >=3 in curated GO/Reactome/KEGG <=200, no PPI), mid layer, the 9 models that
have held-out (scGPT excluded). Build the shared backbone (concepts in >=7 of 9, ~ the >=8/10 tier) on each and
compare: Jaccard, % reproduced, per-model concept Jaccard. Baseline = calibrated random-gene two-draw overlap.
    python scripts/heldout_calibrated.py -> outputs/atlas/comparative/heldout_calibrated.json
"""
from __future__ import annotations
import json, glob, os
import numpy as np
from scipy.stats import hypergeom
from collections import defaultdict, Counter

BASE = "/Users/annaantipova/Desktop/biomech"
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
HO = f"{BASE}/outputs/atlas/heldout_cat/out_heldout"; ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"; TS = f"{BASE}/outputs/atlas/ts3_out"
ALPHA, TOP, AMIN, MX = 0.05, 10, 3, 200
MODELS = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]  # 9 (scGPT no held-out)
KSHARE = 7   # >=7 of 9 ~ the >=8/10 backbone tier
rng = np.random.default_rng(0)

gs = {}
for src in ("GO_BP", "KEGG", "Reactome"):
    for t, g in json.load(open(f"{G}/{src}_gene_sets.json")).items():
        if 5 <= len(g) <= MX:
            gs[f"{src}:{t}"] = set(x.upper() for x in g)
BG = set(x.upper() for x in json.load(open(f"{G}/background.json"))); BGN = len(BG); BG_ARR = np.array(sorted(BG))
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


def ho_mid(m):
    cats = sorted(glob.glob(f"{HO}/{m}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])
    cp = cats[len(cats) // 2]
    return cp, json.load(open(cp))["layer"]


def orig_at(m, L):
    d = ALLCAT if glob.glob(f"{ALLCAT}/{m}/feature_catalog_L*.json") else TS
    p = f"{d}/{m}/feature_catalog_L{L:02d}.json"
    if not os.path.exists(p):  # fall back to nearest
        cats = sorted(glob.glob(f"{d}/{m}/feature_catalog_L*.json"), key=lambda x: json.load(open(x))["layer"])
        p = cats[len(cats) // 2]
    return p


orig = {}; held = {}; nf = {}
for m in MODELS:
    hp, L = ho_mid(m)
    held[m] = annotate([f["top_genes"] for f in json.load(open(hp))["features"].values()])
    op = orig_at(m, L)
    orig[m] = annotate([f["top_genes"] for f in json.load(open(op))["features"].values()])
    nf[m] = len(json.load(open(hp))["features"])


def backbone(sets):
    cm = Counter()
    for m in sets:
        for t in sets[m]:
            cm[t] += 1
    return set(t for t, v in cm.items() if v >= KSHARE)

bb_o = backbone(orig); bb_h = backbone(held)
jac = len(bb_o & bb_h) / max(len(bb_o | bb_h), 1)
pct = 100 * len(bb_o & bb_h) / max(len(bb_o), 1)
permodel = {m: round(len(orig[m] & held[m]) / max(len(orig[m] | held[m]), 1), 3) for m in MODELS}

# null: calibrated random-gene two independent draws -> backbone overlap
null_j = []
for _ in range(20):
    r1 = {m: annotate([BG_ARR[rng.integers(0, BGN, TOP)].tolist() for _ in range(nf[m])]) for m in MODELS}
    r2 = {m: annotate([BG_ARR[rng.integers(0, BGN, TOP)].tolist() for _ in range(nf[m])]) for m in MODELS}
    b1, b2 = backbone(r1), backbone(r2)
    null_j.append(len(b1 & b2) / max(len(b1 | b2), 1))

out = {"annotator": f"top{TOP} a>={AMIN} GO/Re/KEGG<= {MX} no-PPI, mid, {len(MODELS)} models, >={KSHARE}/9 backbone",
       "backbone_orig": len(bb_o), "backbone_heldout": len(bb_h),
       "backbone_jaccard": round(jac, 3), "pct_reproduced": round(pct, 1),
       "mean_per_model_jaccard": round(float(np.mean(list(permodel.values()))), 3), "per_model": permodel,
       "null_two_draw_backbone_jaccard": round(float(np.mean(null_j)), 3),
       "reproduced_terms": sorted(bb_o & bb_h)[:40]}
json.dump(out, open(f"{C}/heldout_calibrated.json", "w"), indent=1)
print(f"CALIBRATED backbone (>={KSHARE}/9): orig {len(bb_o)} | held-out {len(bb_h)}")
print(f"  Jaccard {out['backbone_jaccard']} | % reproduced {out['pct_reproduced']}% | per-model mean Jaccard {out['mean_per_model_jaccard']}")
print(f"  vs random-gene two-draw baseline {out['null_two_draw_backbone_jaccard']}")
print("  reproduced e.g.:", [t.split(':',1)[1][:32] for t in out['reproduced_terms'][:10]])
print("==> heldout_calibrated.json")
