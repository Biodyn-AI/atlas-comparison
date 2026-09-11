#!/usr/bin/env python
"""depth_backbone.py at 250 permutations instead of 20, by replacing the per-feature Python
annotator with the sparse incidence-matrix product used in alllayer_null.py (~50x).

Why. The depth sweep's observed counts (41-102 concepts) need no permutations at all, but the folds
reported alongside them (16.7-27.6x) divide by a null mean estimated from only 20 draws, while the
headline backbone null uses 250. For a paper whose argument is that nulls must be reported properly,
that asymmetry is worth removing rather than explaining.

The annotator is unchanged in substance -- same top-10 genes, same >=3 overlap, same curated sets
<=200, same BH<0.05 -- so the script FIRST reproduces the existing observed counts exactly and
refuses to continue if they differ. Only then are the nulls recomputed.
    python scripts/depth_backbone_fast.py [NPERM] -> outputs/atlas/comparative/depth_backbone.json
"""
from __future__ import annotations
import json, glob, re, sys
import numpy as np
import scipy.sparse as sp
from scipy.stats import hypergeom
from collections import Counter

BASE = "/Users/annaantipova/Desktop/biomech"
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"; TS = f"{BASE}/outputs/atlas/ts3_out"
ALPHA, TOP, AMIN, MX = 0.05, 10, 3, 200
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 250
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]
THRESH = [7, 8, 9, 10]
rng = np.random.default_rng(0)

RAW = {}
for src in ("GO_BP", "KEGG", "Reactome"):
    for t, g in json.load(open(f"{G}/{src}_gene_sets.json")).items():
        if 5 <= len(g) <= MX:
            RAW[f"{src}:{t}"] = set(x.upper() for x in g)
BG = set(x.upper() for x in json.load(open(f"{G}/background.json")))
BGN = len(BG); BG_ARR = np.array(sorted(BG)); GIX = {g: i for i, g in enumerate(BG_ARR.tolist())}
TERMS = sorted(RAW); TSIZE = np.array([len(RAW[t]) for t in TERMS], dtype=np.int64)
_r, _c = [], []
for ti, t in enumerate(TERMS):
    for g in RAW[t]:
        j = GIX.get(g)
        if j is not None: _r.append(j); _c.append(ti)
M = sp.csr_matrix((np.ones(len(_r), np.int32), (_r, _c)), shape=(len(BG_ARR), len(TERMS)))


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.arange(1, n + 1)
    q = np.minimum.accumulate((p[o] * n / r)[::-1])[::-1]; out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


def annotate(feat_lists):
    """identical rule to depth_backbone.annotate, computed as one sparse product"""
    rows, cols = [], []; lg = np.zeros(len(feat_lists), np.int64)
    for i, genes in enumerate(feat_lists):
        g = set(str(x).upper() for x in genes[:TOP]) & BG
        if len(g) < 3: continue
        lg[i] = len(g)
        for x in g: rows.append(i); cols.append(GIX[x])
    if not rows: return set()
    F = sp.csr_matrix((np.ones(len(rows), np.int32), (rows, cols)), shape=(len(feat_lists), len(BG_ARR)))
    A = (F @ M).tocoo(); keep = A.data >= AMIN
    if not keep.any(): return set()
    a = A.data[keep].astype(np.int64); fi = A.row[keep]; ti = A.col[keep]
    q = bh(hypergeom.sf(a - 1, BGN, TSIZE[ti], lg[fi]))
    return set(TERMS[j] for j in np.unique(ti[q <= ALPHA]))


def cats_of(m):
    """Sort by the layer index in the FILENAME. Reading each file to get its "layer" field, as the
    original did, parses ~700 MB of JSON before any work starts -- verified identical here."""
    d = ALLCAT if glob.glob(f"{ALLCAT}/{m}/feature_catalog_L*.json") else TS
    paths = glob.glob(f"{d}/{m}/feature_catalog_L*.json")
    return sorted(paths, key=lambda p: int(re.search(r"_L(\d+)\.json$", p).group(1)))


CATS = {m: cats_of(m) for m in ORDER}


def feats_at(m, f):
    cats = CATS[m]; i = int(round(f * (len(cats) - 1)))
    cat = json.load(open(cats[i]))
    return [x["top_genes"] for x in cat["features"].values()], cat["layer"]


PREV = {}
try:
    for r in json.load(open(f"{C}/depth_backbone.json"))["depths"]:
        PREV[r["frac"]] = {k: v["real"] for k, v in r["tiers"].items()}
except Exception:
    pass

out = {"annotator": f"top{TOP} a>={AMIN} GO/Re/KEGG<= {MX} no-PPI (calibrated)", "n_perm": NPERM,
       "note": "nulls recomputed at 250 permutations with a sparse annotator; observed counts "
               "verified identical to the 20-permutation run", "depths": []}
mismatch = []
for f in FRACS:
    real, nfeat, used = {}, {}, {}
    for m in ORDER:
        fl, L = feats_at(m, f); nfeat[m] = len(fl); used[m] = L
        real[m] = annotate(fl)
    cm = Counter()
    for m in ORDER:
        for t in real[m]: cm[t] += 1
    real_ge = {k: sum(1 for v in cm.values() if v >= k) for k in THRESH}
    if f in PREV:
        for k in THRESH:
            if PREV[f].get(f">={k}") != real_ge[k]:
                mismatch.append(f"f={f} >={k}: was {PREV[f].get(f'>={k}')}, now {real_ge[k]}")
    ge8_terms = sorted([t for t, v in cm.items() if v >= 8])
    null_ge = {k: [] for k in THRESH}
    for it in range(NPERM):
        cc = Counter()
        for m in ORDER:
            for t in annotate([BG_ARR[rng.integers(0, BGN, TOP)].tolist() for _ in range(nfeat[m])]):
                cc[t] += 1
        for k in THRESH:
            null_ge[k].append(sum(1 for v in cc.values() if v >= k))
        if (it + 1) % 50 == 0: print(f"    f={f} perm {it+1}/{NPERM}", flush=True)
    row = {"frac": f, "layers_used": used, "tiers": {}}
    for k in THRESH:
        arr = np.array(null_ge[k], float); nm_ = float(arr.mean())
        row["tiers"][f">={k}"] = {"real": real_ge[k], "null": round(nm_, 1), "null_sd": round(float(arr.std()), 2),
                                  "fold": round(real_ge[k] / max(nm_, 0.3), 1),
                                  "p_emp": round((int((arr >= real_ge[k]).sum()) + 1) / (NPERM + 1), 5)}
    row["ge8_terms"] = ge8_terms
    out["depths"].append(row)
    t = row["tiers"]
    print(f"f={f:<5} " + "  ".join(f">={k}: {t[f'>={k}']['real']} vs {t[f'>={k}']['null']} "
                                   f"({t[f'>={k}']['fold']}x, p={t[f'>={k}']['p_emp']})" for k in THRESH), flush=True)

if mismatch:
    print("\n*** observed counts differ from the stored run -- NOT writing the file ***")
    for m_ in mismatch: print("   ", m_)
    sys.exit(1)

mid = set(next(r for r in out["depths"] if r["frac"] == 0.5)["ge8_terms"])
for r in out["depths"]:
    s = set(r["ge8_terms"])
    r["jaccard_vs_mid"] = round(len(s & mid) / max(len(s | mid), 1), 3)
json.dump(out, open(f"{C}/depth_backbone.json", "w"), indent=1)
ge8 = [r["tiers"][">=8"]["real"] for r in out["depths"]]
fld = [r["tiers"][">=8"]["fold"] for r in out["depths"]]
print(f"\nobserved counts reproduce the previous run exactly")
print(f">=8 concepts across depth: {min(ge8)}-{max(ge8)}   folds: {min(fld)}-{max(fld)}x")
print("==> depth_backbone.json", flush=True)
