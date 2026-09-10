#!/usr/bin/env python
"""Hypothesis-generator, rebuilt with a calibrated annotator and a per-gene degree-matched null.

Selection : a gene qualifies if it is the top gene of a feature that the CALIBRATED annotator
            (top-10, >=3 genes in curated GO/Reactome/KEGG 5-200, no PPI, BH<=0.05) leaves
            UNannotated, in >=THRESH of the 10 models, at the mid layer.
Null      : within each model, permute the annotated/unannotated label across features, keeping
            the number of unannotated features fixed. This holds every gene's degree EXACTLY
            constant -- the gene still tops the same features in the same models -- so the only
            thing tested is whether it concentrates on unannotated ones. Popularity cannot pass it.
Survivors : empirical p per gene, BH across all genes reaching THRESH, keep q <= 0.05.
            For survivors, co-firing partners get the same per-partner test.
    python scripts/novel_calibrated.py [NPERM]  -> outputs/atlas/comparative/novel_calibrated.json
"""
from __future__ import annotations
import json, glob, sys
import numpy as np
import scipy.sparse as sp
from scipy.stats import hypergeom
from collections import defaultdict

BASE = "/Users/annaantipova/Desktop/biomech"
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"; TS = f"{BASE}/outputs/atlas/ts3_out"
ALPHA, TOP, AMIN, MN, MX = 0.05, 10, 3, 5, 200          # calibrated annotator (== degree_null.py)
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
THRESH = max(4, len(ORDER) // 2)                         # 5 of 10
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
NPART = 5                                                # co-firing partners drawn from top-5
rng = np.random.default_rng(0)

# ---- curated vocabulary, no PPI -------------------------------------------
gs = {}
for src in ("GO_BP", "KEGG", "Reactome"):
    for t, g in json.load(open(f"{G}/{src}_gene_sets.json")).items():
        if MN <= len(g) <= MX:
            gs[f"{src}:{t}"] = set(x.upper() for x in g)
BG = set(x.upper() for x in json.load(open(f"{G}/background.json"))); BGN = len(BG)
BG_ARR = np.array(sorted(BG)); BG_IX = {g: i for i, g in enumerate(BG_ARR.tolist())}
TERMS = sorted(gs); TSIZE = np.array([len(gs[t]) for t in TERMS], dtype=np.int64)
_r, _c = [], []
for ti, t in enumerate(TERMS):
    for g in gs[t]:
        gi = BG_IX.get(g)
        if gi is not None: _r.append(gi); _c.append(ti)
M = sp.csr_matrix((np.ones(len(_r), np.int32), (_r, _c)), shape=(len(BG_ARR), len(TERMS)))


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.arange(1, n + 1)
    q = np.minimum.accumulate((p[o] * n / r)[::-1])[::-1]; out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


def annotated_mask(feat_lists):
    """Per-feature: does the CALIBRATED annotator give it at least one significant term?"""
    rows, cols = [], []; lg = np.zeros(len(feat_lists), np.int64)
    for i, genes in enumerate(feat_lists):
        g = set(str(x).upper() for x in genes[:TOP]) & BG
        if len(g) < 3: continue
        lg[i] = len(g)
        for x in g: rows.append(i); cols.append(BG_IX[x])
    mask = np.zeros(len(feat_lists), bool)
    if not rows: return mask
    F = sp.csr_matrix((np.ones(len(rows), np.int32), (rows, cols)), shape=(len(feat_lists), len(BG_ARR)))
    A = (F @ M).tocoo(); keep = A.data >= AMIN
    if not keep.any(): return mask
    a = A.data[keep].astype(np.int64); fi = A.row[keep]; ti = A.col[keep]
    q = bh(hypergeom.sf(a - 1, BGN, TSIZE[ti], lg[fi]))
    mask[np.unique(fi[q <= ALPHA])] = True
    return mask


def mid_features(m):
    d = ALLCAT if glob.glob(f"{ALLCAT}/{m}/feature_catalog_L*.json") else TS
    cats = sorted(glob.glob(f"{d}/{m}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])
    cat = json.load(open(cats[len(cats) // 2]))
    return json.load(open(cats[len(cats) // 2]))["layer"], [f["top_genes"] for f in cat["features"].values()]


def sym(genes, n):
    out = []
    for x in genes[:n + 3]:
        x = str(x).upper()
        if not x.startswith("ENSG"): out.append(x)
        if len(out) == n: break
    return out


# ---- per model: top gene, partners, unannotated mask -----------------------
VOC = {}                                                 # gene symbol -> index
def vix(g):
    i = VOC.get(g)
    if i is None: i = VOC[g] = len(VOC)
    return i

TGI, UN, PART, LAYER = {}, {}, {}, {}
for m in ORDER:
    layer, lists = mid_features(m)
    ann = annotated_mask(lists)
    tg, un, part = [], [], []
    for genes, a in zip(lists, ann):
        s = sym(genes, NPART)
        if not s: continue
        tg.append(vix(s[0])); un.append(not a); part.append([vix(x) for x in s[1:]])
    TGI[m] = np.array(tg, np.int32); UN[m] = np.array(un, bool); PART[m] = part; LAYER[m] = layer
    print(f"  {m}: L{layer}, {len(lists)} features, {int(ann.sum())} annotated (calibrated), "
          f"{int((~ann).sum())} unannotated", flush=True)

NG = len(VOC); INV = np.empty(NG, object)
for g, i in VOC.items(): INV[i] = g


def k_per_gene(masks):
    """models in which a gene tops a feature carrying mask=True"""
    k = np.zeros(NG, np.int32)
    for m in ORDER:
        hit = np.zeros(NG, bool); hit[TGI[m][masks[m]]] = True
        k += hit
    return k


k_obs = k_per_gene(UN)
cand = np.where(k_obs >= THRESH)[0]
print(f"\nCandidates at >={THRESH}/{len(ORDER)} models (calibrated, before the null): {len(cand)}", flush=True)

# ---- degree-matched null: shuffle the annotated label within each model -----
ge = np.zeros(len(cand), np.int64); glob_null = np.zeros(NPERM, np.int32)
for it in range(NPERM):
    masks = {m: rng.permutation(UN[m]) for m in ORDER}
    kn = k_per_gene(masks)
    ge += (kn[cand] >= k_obs[cand])
    glob_null[it] = int((kn >= THRESH).sum())
    if (it + 1) % 500 == 0: print(f"  perm {it+1}/{NPERM}", flush=True)

p = (ge + 1) / (NPERM + 1)
q = bh(p)
surv = cand[q <= ALPHA]
gz = (len(cand) - glob_null.mean()) / (glob_null.std() or 1)
print(f"\nGlobal: real {len(cand)} vs label-shuffle null {glob_null.mean():.1f}±{glob_null.std():.1f} "
      f"-> z={gz:.2f}, {len(cand)/max(glob_null.mean(),1e-9):.2f}x", flush=True)
print(f"Survivors at BH q<={ALPHA}: {len(surv)}", flush=True)

# ---- co-firing partners of survivors, same null ----------------------------
out_genes = []
if len(surv):
    sset = set(surv.tolist())
    def partner_counts(masks):
        cnt = defaultdict(lambda: np.zeros(NG, np.int32))
        for m in ORDER:
            seen = defaultdict(set)
            mk = masks[m]
            for fi in np.where(mk)[0]:
                g = int(TGI[m][fi])
                if g in sset: seen[g].update(PART[m][fi])
            for g, ps in seen.items():
                for pg in ps: cnt[g][pg] += 1
        return cnt
    pc_obs = partner_counts(UN)
    pc_ge = {g: np.zeros(NG, np.int64) for g in sset}
    for it in range(NPERM):
        masks = {m: rng.permutation(UN[m]) for m in ORDER}
        pc = partner_counts(masks)
        for g in sset:
            pc_ge[g] += (pc[g] >= pc_obs[g])
    for gi in surv:
        gi = int(gi)
        pp = (pc_ge[gi] + 1) / (NPERM + 1)
        cnts = pc_obs[gi]
        cand_p = np.where(cnts >= 2)[0]
        if len(cand_p):
            pq = bh(pp[cand_p])
            keep = cand_p[pq <= ALPHA]
            keep = keep[np.argsort(-cnts[keep])][:6]
        else:
            keep = np.array([], int)
        out_genes.append({"gene": str(INV[gi]), "n_models": int(k_obs[gi]),
                          "p": round(float(p[list(cand).index(gi)]), 5),
                          "q": round(float(q[list(cand).index(gi)]), 5),
                          "co": [str(INV[j]) for j in keep]})
    out_genes.sort(key=lambda d: (-d["n_models"], d["q"]))

res = {"annotator": "calibrated: top-10, >=3 genes in GO_BP/KEGG/Reactome 5-200, no PPI, BH<=0.05",
       "null": "per-model permutation of the annotated/unannotated label (gene degree held exactly fixed)",
       "threshold_models": THRESH, "n_models": len(ORDER), "n_perm": NPERM, "layers": LAYER,
       "n_candidates_before_null": int(len(cand)),
       "global_null_mean": round(float(glob_null.mean()), 1), "global_null_sd": round(float(glob_null.std()), 1),
       "global_z": round(float(gz), 2), "global_fold": round(len(cand) / max(float(glob_null.mean()), 1e-9), 2),
       "n_survivors": int(len(surv)), "alpha": ALPHA, "candidates": out_genes}
json.dump(res, open(f"{C}/novel_calibrated.json", "w"), indent=1)
print(f"==> novel_calibrated.json  ({len(out_genes)} survivors)", flush=True)
