#!/usr/bin/env python
"""#1 (calibrated companions for the cheap annotation panels) + CI (bootstrap on the n=10 correlations).
(A) Under the CALIBRATED annotator (top-10, >=3 in curated GO/Reactome/KEGG <=200, no PPI), mid layer, all 10
    models: per-model annotation rate, distinct concepts, and blind spots (exclusive concepts) — the calibrated
    companion to the permissive coverage/blind-spot panels.
(B) Bootstrap 95% CIs for the key cross-model correlations (n=10 is small).
    -> merges 'calibrated_panels' and 'corr_ci' into controls.json
"""
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
rng = np.random.default_rng(0)

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


def annotate(feat_lists):
    A = []; TG = []; LG = []; TERM = []; FI = []
    for fi, genes in enumerate(feat_lists):
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
            A.append(a); TG.append(size[k]); LG.append(len(g)); TERM.append(k); FI.append(fi)
    if not A:
        return set(), 0
    q = bh(hypergeom.sf(np.array(A) - 1, BGN, np.array(TG), np.array(LG)))
    keep = np.where(q <= ALPHA)[0]
    return set(TERM[i] for i in keep), len(set(FI[i] for i in keep))


def load_feats(m):
    d = f"{ALLCAT}/{m}"
    if not glob.glob(f"{d}/feature_catalog_L*.json"):
        d = f"{TS}/{m}"
    cats = sorted(glob.glob(f"{d}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])
    return [f["top_genes"] for f in json.load(open(cats[len(cats) // 2]))["features"].values()]

# (A) calibrated per-model
real = {}; rate = {}; nf = {}
for m in ORDER:
    feats = load_feats(m); nf[m] = len(feats)
    terms, nann = annotate(feats)
    real[m] = terms; rate[m] = round(100 * nann / max(nf[m], 1), 1)
cm = Counter()
for m in ORDER:
    for t in real[m]:
        cm[t] += 1
blind = {m: sum(1 for t in real[m] if cm[t] == 1) for m in ORDER}   # exclusive to this model
calib = {"annotator": f"top{TOP} a>={AMIN} GO/Re/KEGG<= {MX} no-PPI, mid layer",
         "rate": rate, "n_concepts": {m: len(real[m]) for m in ORDER}, "blind_exclusive": blind}
print("(A) calibrated mid rate:", rate)
print("    blind (exclusive):", blind)

# (B) bootstrap CIs for the n=10 correlations
ctl = json.load(open(f"{C}/controls.json"))
cap = ctl["capacity"]; conc = ctl["concentration"]
mid = ORDER
dsae = np.array([cap["d_sae"][m] for m in mid], float)
nconc = np.array([cap["n_concepts"][m] for m in mid], float)
nlay = np.array([ctl["equal_layer"]["n_layers"][m] for m in mid], float)
concv = np.array([conc["conc"][m] for m in mid], float)
tax = json.load(open(f"{C}/atlas_full_notf.json"))["axes"]["tax"]
rank = np.array([1.0 if tax[m]["tok"] == "rank" else 0.0 for m in mid])


def boot_ci(x, y, n=5000):
    r0 = float(np.corrcoef(x, y)[0, 1]); N = len(x); idx0 = np.arange(N); bs = []
    for _ in range(n):
        b = rng.choice(idx0, N, replace=True)
        if len(set(b.tolist())) < 3:
            continue
        with np.errstate(all="ignore"):
            rr = np.corrcoef(x[b], y[b])[0, 1]
        if not np.isnan(rr):
            bs.append(rr)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"r": round(r0, 3), "ci95": [round(float(lo), 2), round(float(hi), 2)], "n": N}

ci = {
    "nconcepts_vs_log_dsae": boot_ci(np.log10(dsae), nconc),
    "nconcepts_vs_nlayers": boot_ci(nlay, nconc),
    "concentration_vs_log_dsae": boot_ci(np.log10(dsae), concv),
    "concentration_vs_rank_tok": boot_ci(rank, concv),
}
print("(B) bootstrap CIs:")
for k, v in ci.items():
    print(f"    {k}: r={v['r']} CI{v['ci95']}")

ctl["calibrated_panels"] = calib; ctl["corr_ci"] = ci
json.dump(ctl, open(f"{C}/controls.json", "w"), indent=1)
print("==> controls.json (calibrated_panels + corr_ci)")
