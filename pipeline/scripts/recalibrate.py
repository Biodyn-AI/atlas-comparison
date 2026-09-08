#!/usr/bin/env python
"""Can a STRICTER, calibrated annotator rebuild the universal core so that it BEATS the random-gene null?
Grid over: database subset (drop STRING/TRRUST PPI), max gene-set size, min overlap a, odds-ratio floor. For each
config compute, at the mid layer over all 10 models: real annotation rate, random-gene annotation rate (FPR), the
real universal core, and the random-gene core (mean over perms). A config "wins" if random-gene rate is controlled
AND real core > random core by a clear margin (z>0). Reports the grid so we can pick the calibrated annotator.
    python scripts/recalibrate.py   -> outputs/atlas/comparative/recalibrate.json
"""
from __future__ import annotations
import json, glob, os
import numpy as np
from scipy.stats import hypergeom
from collections import defaultdict, Counter

BASE = "/Users/annaantipova/Desktop/biomech"
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"; TS = f"{BASE}/outputs/atlas/ts3_out"
ALPHA, TOP, NPERM = 0.05, 5, 20
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
rng = np.random.default_rng(0)

RAW = {}   # source -> {term: set(genes)}
for nm in ("GO_BP", "KEGG", "Reactome"):
    RAW[nm] = {t: set(x.upper() for x in g) for t, g in json.load(open(f"{G}/{nm}_gene_sets.json")).items()}
RAW["TRRUST"] = {t: set(x.upper() for x in g) for t, g in json.load(open(f"{G}/trrust_edges.json")).items()}
RAW["STRING"] = {t: set(x.upper() for x in g) for t, g in json.load(open(f"{G}/string_edges.json")).items()}
BGBASE = set(x.upper() for x in json.load(open(f"{G}/background.json")))


def build_index(dbs, mn, mx):
    gs = {}
    for src in dbs:
        for t, genes in RAW[src].items():
            if mn <= len(genes) <= mx:
                gs[f"{src}:{t}"] = genes
    bg = set(BGBASE)
    if "STRING" in dbs:
        bg |= set().union(*[v for k, v in gs.items() if k.startswith("STRING:")]) | \
              set(k.split(":", 1)[1].upper() for k in gs if k.startswith("STRING:"))
    idx = defaultdict(list); size = {}
    for t, genes in gs.items():
        size[t] = len(genes)
        for g in genes:
            idx[g].append(t)
    return idx, size, bg, len(bg)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.arange(1, n + 1)
    q = np.minimum.accumulate((p[o] * n / r)[::-1])[::-1]; out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


def annotate(feat_lists, idx, size, bg, bgn, amin, ormin):
    A = []; TG = []; LG = []; TERM = []; FI = []
    for fi, genes in enumerate(feat_lists):
        g = set(str(x).upper() for x in genes[:TOP]) & bg
        if len(g) < 3:
            continue
        cand = Counter()
        for gene in g:
            for k in idx.get(gene, ()):
                cand[k] += 1
        for k, a in cand.items():
            if a < amin:
                continue
            A.append(a); TG.append(size[k]); LG.append(len(g)); TERM.append(k); FI.append(fi)
    if not A:
        return set(), 0
    A = np.array(A); TG = np.array(TG); LG = np.array(LG)
    p = hypergeom.sf(A - 1, bgn, TG, LG)
    # odds-ratio (2x2): (a/(len_g-a)) / ((tg-a)/(bgn-tg-(len_g-a)))
    a = A.astype(float); lg = LG.astype(float); tg = TG.astype(float)
    orr = (a * (bgn - tg - (lg - a))) / np.maximum((lg - a) * (tg - a), 1e-9)
    q = bh(p)
    keep = (q <= ALPHA) & (orr >= ormin)
    terms = set(TERM[i] for i in np.where(keep)[0])
    ann_feats = set(FI[i] for i in np.where(keep)[0])
    return terms, len(ann_feats)


def load_feats(m):
    d = f"{ALLCAT}/{m}"
    if not glob.glob(f"{d}/feature_catalog_L*.json"):
        d = f"{TS}/{m}"
    cats = sorted(glob.glob(f"{d}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])
    cat = json.load(open(cats[len(cats) // 2]))
    return [f["top_genes"] for f in cat["features"].values()]

FEATS = {m: load_feats(m) for m in ORDER}
NFEAT = {m: len(FEATS[m]) for m in ORDER}

CONFIGS = [
    ("no-PPI max100 a>=3 OR>=10", ["GO_BP", "KEGG", "Reactome"], 5, 100, 3, 10),
    ("no-PPI max100 a>=3", ["GO_BP", "KEGG", "Reactome"], 5, 100, 3, 0),
    ("no-PPI max200 a>=3 OR>=5", ["GO_BP", "KEGG", "Reactome"], 5, 200, 3, 5),
    ("no-PPI max200 a>=3", ["GO_BP", "KEGG", "Reactome"], 5, 200, 3, 0),
    ("no-PPI max200 a>=2", ["GO_BP", "KEGG", "Reactome"], 5, 200, 2, 0),
]

out = {"configs": []}
for name, dbs, mn, mx, amin, ormin in CONFIGS:
    idx, size, bg, bgn = build_index(dbs, mn, mx)
    bg_arr = np.array(sorted(bg))
    real = {}; rates = []
    for m in ORDER:
        terms, nann = annotate(FEATS[m], idx, size, bg, bgn, amin, ormin)
        real[m] = terms; rates.append(100 * nann / max(NFEAT[m], 1))
    real_core = len(set.intersection(*real.values())) if all(real.values()) else 0
    real_rate = float(np.mean(rates))
    rcores = []; rrates = []
    for _ in range(NPERM):
        rs = {}; rr = []
        for m in ORDER:
            rg = [rng.choice(bg_arr, TOP, replace=False).tolist() for _ in range(NFEAT[m])]
            terms, nann = annotate(rg, idx, size, bg, bgn, amin, ormin)
            rs[m] = terms; rr.append(100 * nann / max(NFEAT[m], 1))
        rcores.append(len(set.intersection(*rs.values())) if all(rs.values()) else 0)
        rrates.append(float(np.mean(rr)))
    rc = np.array(rcores); z = (real_core - rc.mean()) / (rc.std() or 1)
    rec = {"config": name, "real_rate": round(real_rate, 1), "rand_rate": round(float(np.mean(rrates)), 1),
           "real_core": real_core, "rand_core_mean": round(float(rc.mean()), 1), "rand_core_sd": round(float(rc.std()), 1),
           "core_z": round(float(z), 1), "beats_null": bool(real_core > rc.mean() + 2 * (rc.std() or 1))}
    out["configs"].append(rec)
    print(f"{name:36} real {rec['real_rate']:5}% / rand {rec['rand_rate']:5}% | core real {real_core:5} vs rand {rec['rand_core_mean']:6}±{rec['rand_core_sd']:<4} z={rec['core_z']:6} {'✓BEATS' if rec['beats_null'] else ''}", flush=True)

json.dump(out, open(f"{C}/recalibrate.json", "w"), indent=1)
print("==> recalibrate.json", flush=True)
