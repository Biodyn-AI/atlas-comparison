#!/usr/bin/env python
"""Second recalibration grid — the two plausible rescues the first grid didn't cover:
  (1) a>=2 but a high odds-ratio floor (keep specific 2-gene hits in SMALL sets, cut generic ones);
  (2) MORE top genes (top-10/20) with a>=3/4 (give real coherent features a chance to land >=3 in one set).
Per-config TOP. A config wins if real rate stays usable AND real core > random-gene core (z>2)."""
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
rng = np.random.default_rng(0)

RAW = {nm: {t: set(x.upper() for x in g) for t, g in json.load(open(f"{G}/{nm}_gene_sets.json")).items()}
       for nm in ("GO_BP", "KEGG", "Reactome")}
BGBASE = set(x.upper() for x in json.load(open(f"{G}/background.json")))


def build_index(mx):
    gs = {}
    for src in ("GO_BP", "KEGG", "Reactome"):
        for t, genes in RAW[src].items():
            if 5 <= len(genes) <= mx:
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


def annotate(feat_lists, idx, size, bgn, top, amin, ormin):
    A = []; TG = []; LG = []; TERM = []; FI = []
    for fi, genes in enumerate(feat_lists):
        g = set(str(x).upper() for x in genes[:top]) & BG
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
    a = A.astype(float); lg = LG.astype(float); tg = TG.astype(float)
    orr = (a * (bgn - tg - (lg - a))) / np.maximum((lg - a) * (tg - a), 1e-9)
    keep = (bh(p) <= ALPHA) & (orr >= ormin)
    return set(TERM[i] for i in np.where(keep)[0]), len(set(FI[i] for i in np.where(keep)[0]))


def load_feats(m):
    d = f"{ALLCAT}/{m}"
    if not glob.glob(f"{d}/feature_catalog_L*.json"):
        d = f"{TS}/{m}"
    cats = sorted(glob.glob(f"{d}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])
    cat = json.load(open(cats[len(cats) // 2]))
    return [f["top_genes"] for f in cat["features"].values()]

FEATS = {m: load_feats(m) for m in ORDER}
NFEAT = {m: len(FEATS[m]) for m in ORDER}
BG = set(BGBASE); BGN = len(BG); BG_ARR = np.array(sorted(BG))

CONFIGS = [
    ("top5 max200 a>=2 OR>=10", 200, 5, 2, 10),
    ("top5 max200 a>=2 OR>=20", 200, 5, 2, 20),
    ("top5 max100 a>=2 OR>=20", 100, 5, 2, 20),
    ("top10 max200 a>=3", 200, 10, 3, 0),
    ("top10 max200 a>=3 OR>=10", 200, 10, 3, 10),
    ("top20 max300 a>=4", 300, 20, 4, 0),
]
out = {"configs": []}
for name, mx, top, amin, ormin in CONFIGS:
    idx, size = build_index(mx)
    real = {m: annotate(FEATS[m], idx, size, BGN, top, amin, ormin) for m in ORDER}
    real_core = len(set.intersection(*[real[m][0] for m in ORDER])) if all(real[m][0] for m in ORDER) else 0
    real_rate = float(np.mean([100 * real[m][1] / max(NFEAT[m], 1) for m in ORDER]))
    rcores = []; rrates = []
    for _ in range(NPERM):
        rs = {}; rr = []
        for m in ORDER:
            rg = [BG_ARR[rng.integers(0, BGN, top)].tolist() for _ in range(NFEAT[m])]
            t, n = annotate(rg, idx, size, BGN, top, amin, ormin); rs[m] = t; rr.append(100 * n / max(NFEAT[m], 1))
        rcores.append(len(set.intersection(*rs.values())) if all(rs.values()) else 0); rrates.append(float(np.mean(rr)))
    rc = np.array(rcores); z = (real_core - rc.mean()) / (rc.std() or 1)
    rec = {"config": name, "real_rate": round(real_rate, 1), "rand_rate": round(float(np.mean(rrates)), 1),
           "real_core": real_core, "rand_core_mean": round(float(rc.mean()), 1), "rand_core_sd": round(float(rc.std()), 1),
           "core_z": round(float(z), 1), "beats_null": bool(real_core > rc.mean() + 2 * (rc.std() or 1) and real_rate >= 10)}
    out["configs"].append(rec)
    print(f"{name:28} real {rec['real_rate']:5}% / rand {rec['rand_rate']:5}% | core real {real_core:5} vs rand {rec['rand_core_mean']:6}±{rec['rand_core_sd']:<4} z={rec['core_z']:6} {'✓BEATS' if rec['beats_null'] else ''}", flush=True)
json.dump(out, open(f"{C}/recalibrate2.json", "w"), indent=1)
print("==> recalibrate2.json", flush=True)
