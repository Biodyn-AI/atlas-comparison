#!/usr/bin/env python
"""Degree-matched random-gene null: the strongest test. Instead of uniform random genes, each real top-gene is
replaced by a random gene with the SAME gene-set membership degree (same bin) — controlling for gene popularity
(high-degree genes are 'hittable' by chance). If the calibrated backbone still beats THIS null, it is not a
gene-abundance artefact. Calibrated annotator (top-10, >=3 in curated GO/Reactome/KEGG <=200, no PPI), mid layer,
all 10 models, N perms -> per-tier real vs degree-matched null, z, empirical p. Also reports the uniform null for
contrast.
    python scripts/degree_null.py -> outputs/atlas/comparative/degree_null.json
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
NB = 250; NBIN = 40
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
BG_ARR = np.array(sorted(BG))
DEG = np.array([len(idx.get(g, ())) for g in BG_ARR])            # gene-set membership degree
# degree bins: degree-0 in its own bin, rest split into NBIN quantile bins
nz = np.where(DEG > 0)[0]; z0 = np.where(DEG == 0)[0]
order = nz[np.argsort(DEG[nz])]
binmembers = [z0] + [b for b in np.array_split(order, NBIN) if len(b)]
gene_bin = np.zeros(len(BG_ARR), int)
for bi, mem in enumerate(binmembers):
    gene_bin[mem] = bi
g2i = {g: i for i, g in enumerate(BG_ARR)}


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


def feats_mid(m):
    d = ALLCAT if glob.glob(f"{ALLCAT}/{m}/feature_catalog_L*.json") else TS
    cats = sorted(glob.glob(f"{d}/{m}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])
    cat = json.load(open(cats[len(cats) // 2]))
    return [f["top_genes"] for f in cat["features"].values()]


# real top-gene lists (bin indices per gene, for degree-matched resampling)
REALGENES = {}
for m in ORDER:
    REALGENES[m] = feats_mid(m)
KS = [7, 8, 9, 10]


def spectrum_ge(sets):
    cm = Counter()
    for s in sets.values():
        for t in s:
            cm[t] += 1
    return {k: sum(1 for v in cm.values() if v >= k) for k in KS}


# feature -> bins of its real top-TOP genes (for degree-matched sampling)
BINS = {m: [[gene_bin[g2i[str(x).upper()]] for x in f[:TOP] if str(x).upper() in g2i] for f in REALGENES[m]] for m in ORDER}

real = {m: annotate(REALGENES[m]) for m in ORDER}
real_ge = spectrum_ge(real)


def sample_bin(b):
    mem = binmembers[b]; return BG_ARR[mem[rng.integers(0, len(mem))]]


def run_null(matched):
    nn = {k: np.empty(NB) for k in KS}
    for it in range(NB):
        rs = {}
        for m in ORDER:
            if matched:
                fl = [[sample_bin(b) for b in bins] for bins in BINS[m]]
            else:
                fl = [BG_ARR[rng.integers(0, BGN, TOP)].tolist() for _ in BINS[m]]
            rs[m] = annotate(fl)
        ge = spectrum_ge(rs)
        for k in KS:
            nn[k][it] = ge[k]
        if (it + 1) % 50 == 0:
            print(f"  {'degree' if matched else 'uniform'} null {it+1}/{NB}", flush=True)
    return nn

print("degree-matched null...", flush=True)
deg = run_null(True)
print("uniform null...", flush=True)
uni = run_null(False)

out = {"annotator": f"top{TOP} a>={AMIN} GO/Re/KEGG<= {MX} no-PPI, mid", "n_perm": NB, "n_bins": len(binmembers), "tiers": {}}
for k in KS:
    d = deg[k]; u = uni[k]
    zd = (real_ge[k] - d.mean()) / (d.std() or 1); pd = (np.sum(d >= real_ge[k]) + 1) / (NB + 1)
    out["tiers"][f">={k}"] = {"real": int(real_ge[k]),
                              "degree_null_mean": round(float(d.mean()), 2), "degree_null_sd": round(float(d.std()), 2),
                              "degree_z": round(float(zd), 1), "degree_p": pd, "degree_fold": round(real_ge[k] / max(d.mean(), 0.1), 1),
                              "uniform_null_mean": round(float(u.mean()), 2)}
    print(f">={k}: real {real_ge[k]} | degree-null {out['tiers'][f'>={k}']['degree_null_mean']}±{out['tiers'][f'>={k}']['degree_null_sd']} z={zd:.1f} p={pd:.2e} fold {out['tiers'][f'>={k}']['degree_fold']} | (uniform-null {out['tiers'][f'>={k}']['uniform_null_mean']})", flush=True)
json.dump(out, open(f"{C}/degree_null.json", "w"), indent=1)
print("==> degree_null.json", flush=True)
