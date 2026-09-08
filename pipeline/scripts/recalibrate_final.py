#!/usr/bin/env python
"""Under the CALIBRATED annotator that beats the random-gene null (top-10 genes, >=3 in a curated GO/Reactome/KEGG
set of <=200 genes, no PPI), compute the honest universality: per-model concept sets, the shared-by-k spectrum,
the all-10 core (identities), each compared to the random-gene null per tier. This is the recalibrated picture."""
from __future__ import annotations
import json, glob
import numpy as np
from scipy.stats import hypergeom
from collections import defaultdict, Counter

BASE = "/Users/annaantipova/Desktop/biomech"
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"; TS = f"{BASE}/outputs/atlas/ts3_out"
ALPHA, TOP, AMIN, MX, NPERM = 0.05, 10, 3, 200, 30
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
N = len(ORDER); rng = np.random.default_rng(0)

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


def load_feats(m):
    d = f"{ALLCAT}/{m}"
    if not glob.glob(f"{d}/feature_catalog_L*.json"):
        d = f"{TS}/{m}"
    cats = sorted(glob.glob(f"{d}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])
    return [f["top_genes"] for f in json.load(open(cats[len(cats) // 2]))["features"].values()]

FEATS = {m: load_feats(m) for m in ORDER}; NFEAT = {m: len(FEATS[m]) for m in ORDER}

real = {m: annotate(FEATS[m]) for m in ORDER}
cm = Counter()
for m in ORDER:
    for t in real[m]:
        cm[t] += 1
spec = {str(k): sum(1 for v in cm.values() if v == k) for k in range(1, N + 1)}
core = sorted([t for t, v in cm.items() if v == N])
shared8 = sorted([t for t, v in cm.items() if v >= 8])

# random-gene null spectrum
null_spec = defaultdict(list); null_core = []
for _ in range(NPERM):
    rs = {m: annotate([BG_ARR[rng.integers(0, BGN, TOP)].tolist() for _ in range(NFEAT[m])]) for m in ORDER}
    ccm = Counter()
    for m in ORDER:
        for t in rs[m]:
            ccm[t] += 1
    for k in range(1, N + 1):
        null_spec[k].append(sum(1 for v in ccm.values() if v == k))
    null_core.append(sum(1 for v in ccm.values() if v == N))

out = {"annotator": f"top{TOP} genes, a>={AMIN}, GO/Reactome/KEGG size<= {MX}, no PPI",
       "real_spectrum": spec, "real_core": len(core), "core_terms": core,
       "shared_ge8": len(shared8), "shared_ge8_terms": shared8[:40],
       "null_core_mean": round(float(np.mean(null_core)), 1), "null_core_sd": round(float(np.std(null_core)), 1),
       "null_spectrum_mean": {str(k): round(float(np.mean(null_spec[k])), 1) for k in range(1, N + 1)}}
json.dump(out, open(f"{C}/recalibrate_final.json", "w"), indent=1)
print("calibrated annotator:", out["annotator"])
print("real spectrum (by #models):", spec)
print("null   spectrum (mean)   :", out["null_spectrum_mean"])
print(f"all-{N} core: real {len(core)} vs null {out['null_core_mean']}±{out['null_core_sd']}  -> {core}")
print(f"shared by >=8: {len(shared8)} -> {shared8[:15]}")
print("==> recalibrate_final.json")
