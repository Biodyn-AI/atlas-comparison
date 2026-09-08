#!/usr/bin/env python
"""Publication-grade nulls for the two headline claims (calibrated annotator, mid layer):
  (1) the shared backbone (>=7/8/9 of 10 models) vs a random-gene null, N=1000 perms -> mean, sd, z, empirical p;
  (2) held-out replication of the backbone (>=7/9 of the 9 held-out models) vs a random-gene two-draw null,
      N=500 pairs -> empirical p that the observed Jaccard exceeds the null.
Calibrated annotator = top-10 genes, >=3 in a curated GO/Reactome/KEGG set of <=200 genes, no PPI.
    python scripts/stats_final.py -> outputs/atlas/comparative/stats_final.json
"""
from __future__ import annotations
import json, glob, os
import numpy as np
from scipy.stats import hypergeom
from collections import defaultdict, Counter

BASE = "/Users/annaantipova/Desktop/biomech"
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"; TS = f"{BASE}/outputs/atlas/ts3_out"; HO = f"{BASE}/outputs/atlas/heldout_cat/out_heldout"
ALPHA, TOP, AMIN, MX = 0.05, 10, 3, 200
ALL = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
HOM = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]  # held-out (no scGPT)
NB, NHO = 250, 150
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


def feats_mid(m, root):
    cats = sorted(glob.glob(f"{root}/{m}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])
    if not cats:
        return None, None
    cp = cats[len(cats) // 2]
    return [f["top_genes"] for f in json.load(open(cp))["features"].values()], json.load(open(cp))["layer"]


def orig_root(m):
    return ALLCAT if glob.glob(f"{ALLCAT}/{m}/feature_catalog_L*.json") else TS


def spectrum_ge(sets, ks):
    cm = Counter()
    for s in sets.values():
        for t in s:
            cm[t] += 1
    return {k: sum(1 for v in cm.values() if v >= k) for k in ks}, cm

# ---------- (1) backbone null, N=1000 ----------
real = {}; nf = {}
for m in ALL:
    fl, _ = feats_mid(m, orig_root(m)); nf[m] = len(fl); real[m] = annotate(fl)
KS = [7, 8, 9, 10]
real_ge, _ = spectrum_ge(real, KS)
null = {k: np.empty(NB) for k in KS}
for it in range(NB):
    rs = {m: annotate([BG_ARR[rng.integers(0, BGN, TOP)].tolist() for _ in range(nf[m])]) for m in ALL}
    ge, _ = spectrum_ge(rs, KS)
    for k in KS:
        null[k][it] = ge[k]
    if (it + 1) % 200 == 0:
        print(f"  backbone null {it+1}/{NB}", flush=True)
b1 = {}
for k in KS:
    nu = null[k]; z = (real_ge[k] - nu.mean()) / (nu.std() or 1); p = (np.sum(nu >= real_ge[k]) + 1) / (NB + 1)
    b1[f">={k}"] = {"real": int(real_ge[k]), "null_mean": round(float(nu.mean()), 2), "null_sd": round(float(nu.std()), 2),
                    "z": round(float(z), 1), "p_emp": p, "fold": round(real_ge[k] / max(nu.mean(), 0.1), 1)}
    print(f"backbone >={k}: real {real_ge[k]} vs null {b1[f'>={k}']['null_mean']}±{b1[f'>={k}']['null_sd']} z={b1[f'>={k}']['z']} p={p:.2e} fold {b1[f'>={k}']['fold']}", flush=True)

# ---------- (2) held-out replication null, N=500 ----------
KHO = 7
orig_h = {}; held_h = {}; nfh = {}
for m in HOM:
    fh, L = feats_mid(m, HO); nfh[m] = len(fh); held_h[m] = annotate(fh)
    p = f"{orig_root(m)}/{m}/feature_catalog_L{L:02d}.json"
    if not os.path.exists(p):
        cats = sorted(glob.glob(f"{orig_root(m)}/{m}/feature_catalog_L*.json"), key=lambda x: json.load(open(x))["layer"]); p = cats[len(cats) // 2]
    orig_h[m] = annotate([f["top_genes"] for f in json.load(open(p))["features"].values()])


def bbone(sets, k):
    cm = Counter()
    for s in sets.values():
        for t in s:
            cm[t] += 1
    return set(t for t, v in cm.items() if v >= k)

bo, bh_ = bbone(orig_h, KHO), bbone(held_h, KHO)
obs_j = len(bo & bh_) / max(len(bo | bh_), 1)
nullj = np.empty(NHO)
for it in range(NHO):
    r1 = {m: annotate([BG_ARR[rng.integers(0, BGN, TOP)].tolist() for _ in range(nfh[m])]) for m in HOM}
    r2 = {m: annotate([BG_ARR[rng.integers(0, BGN, TOP)].tolist() for _ in range(nfh[m])]) for m in HOM}
    b1_, b2_ = bbone(r1, KHO), bbone(r2, KHO)
    nullj[it] = len(b1_ & b2_) / max(len(b1_ | b2_), 1)
    if (it + 1) % 100 == 0:
        print(f"  held-out null {it+1}/{NHO}", flush=True)
pj = (np.sum(nullj >= obs_j) + 1) / (NHO + 1)
b2 = {"observed_jaccard": round(obs_j, 3), "backbone_orig": len(bo), "backbone_heldout": len(bh_),
      "null_mean": round(float(nullj.mean()), 4), "null_sd": round(float(nullj.std()), 4),
      "null_max": round(float(nullj.max()), 3), "p_emp": pj, "fold": round(obs_j / max(nullj.mean(), 1e-3), 1)}
print(f"held-out replication: obs Jaccard {obs_j:.3f} vs null {b2['null_mean']}±{b2['null_sd']} (max {b2['null_max']}) p={pj:.2e} fold {b2['fold']}", flush=True)

json.dump({"annotator": f"top{TOP} a>={AMIN} GO/Re/KEGG<= {MX} no-PPI, mid", "n_perm_backbone": NB, "n_perm_heldout": NHO,
           "backbone_null": b1, "heldout_null": b2}, open(f"{C}/stats_final.json", "w"), indent=1)
print("==> stats_final.json", flush=True)
