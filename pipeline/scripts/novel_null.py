#!/usr/bin/env python
"""Is the "new biology" shortlist (genes topping UNannotated features in >=THRESH models)
more than what any equally-sized set of features would give?

Real:  per model, take the features that got NO annotation (permissive annotator, the same
       one the site uses), take each one's top gene, and count genes reaching >=THRESH models.
Null:  per model, draw the SAME NUMBER of features uniformly from ALL features and repeat.
       If the unannotated subset carries no special signal, real ~= null.
    python scripts/novel_null.py [NPERM]  -> outputs/atlas/comparative/novel_null.json
"""
from __future__ import annotations
import json, glob, sys
import numpy as np
import scipy.sparse as sp
from scipy.stats import hypergeom
from collections import defaultdict, Counter

BASE = "/Users/annaantipova/Desktop/biomech"
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"; TS = f"{BASE}/outputs/atlas/ts3_out"
MIN, MAX, ALPHA, TOP = 5, 500, 0.05, 5          # permissive annotator (as on the site)
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
THRESH = max(4, len(ORDER) // 2)                # 5, same rule as atlas_assemble.py
rng = np.random.default_rng(0)

gs = {}
for nm in ("GO_BP", "KEGG", "Reactome"):
    for t, g in json.load(open(f"{G}/{nm}_gene_sets.json")).items():
        s = set(x.upper() for x in g)
        if MIN <= len(s) <= MAX: gs[f"{nm}:{t}"] = s
for nm, fn in (("TRRUST", "trrust_edges.json"), ("STRING", "string_edges.json")):
    for t, g in json.load(open(f"{G}/{fn}")).items():
        s = set(x.upper() for x in g)
        if MIN <= len(s) <= MAX: gs[f"{nm}:{t}"] = s
bg = set(x.upper() for x in json.load(open(f"{G}/background.json")))
bg |= set().union(*[v for k, v in gs.items() if k.startswith("STRING:")])
bg |= set(k.split(":", 1)[1].upper() for k in gs if k.startswith("STRING:"))
bg_list = np.array(sorted(bg)); bgn = len(bg)
GENE_IX = {g: i for i, g in enumerate(bg_list.tolist())}
TERMS = sorted(gs); TERM_SIZE = np.array([len(gs[t]) for t in TERMS], dtype=np.int64)
_r, _c = [], []
for ti, t in enumerate(TERMS):
    for g in gs[t]:
        gi = GENE_IX.get(g)
        if gi is not None: _r.append(gi); _c.append(ti)
M = sp.csr_matrix((np.ones(len(_r), np.int32), (_r, _c)), shape=(len(bg_list), len(TERMS)))


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.arange(1, n + 1)
    q = np.minimum.accumulate((p[o] * n / r)[::-1])[::-1]; out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


def annotated_mask(feat_lists):
    """Which features get >=1 term under the permissive annotator."""
    rows, cols = [], []; lg = np.zeros(len(feat_lists), np.int64)
    for i, genes in enumerate(feat_lists):
        g = set(str(x).upper() for x in genes[:TOP]) & bg
        if len(g) < 3: continue
        lg[i] = len(g)
        for x in g: rows.append(i); cols.append(GENE_IX[x])
    mask = np.zeros(len(feat_lists), bool)
    if not rows: return mask
    F = sp.csr_matrix((np.ones(len(rows), np.int32), (rows, cols)), shape=(len(feat_lists), len(bg_list)))
    A = (F @ M).tocoo(); keep = A.data >= 2
    if not keep.any(): return mask
    a = A.data[keep].astype(np.int64); fi = A.row[keep]; ti = A.col[keep]
    q = bh(hypergeom.sf(a - 1, bgn, TERM_SIZE[ti], lg[fi]))
    mask[np.unique(fi[q <= ALPHA])] = True
    return mask


def cats_of(m):
    d = ALLCAT if glob.glob(f"{ALLCAT}/{m}/feature_catalog_L*.json") else TS
    return sorted(glob.glob(f"{d}/{m}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])


def top_gene(f):
    for x in f.get("top_genes", [])[:6]:
        x = str(x).upper()
        if not x.startswith("ENSG"): return x
    return None


TOPG, N_UNANN = {}, {}
real_gene_models = defaultdict(set)
for m in ORDER:
    cats = cats_of(m)
    cat = json.load(open(cats[len(cats) // 2]))
    feats = list(cat["features"].values())
    lists = [f["top_genes"] for f in feats]
    ann = annotated_mask(lists)
    tg = [top_gene(f) for f in feats]
    TOPG[m] = [g for g in tg if g]                       # every feature's top gene (for the null)
    unann_genes = [g for g, a in zip(tg, ann) if g and not a]
    N_UNANN[m] = len(unann_genes)
    for g in set(unann_genes): real_gene_models[g].add(m)
    print(f"  {m}: {len(feats)} features, {ann.sum()} annotated, {len(unann_genes)} unannotated", flush=True)

real_hits = sorted([g for g, ms in real_gene_models.items() if len(ms) >= THRESH])
print(f"\nREAL: {len(real_hits)} genes top an UNannotated feature in >={THRESH}/{len(ORDER)} models", flush=True)

null_counts = []
for it in range(NPERM):
    gm = defaultdict(set)
    for m in ORDER:
        pool = TOPG[m]; k = min(N_UNANN[m], len(pool))
        idx = rng.choice(len(pool), k, replace=False)
        for i in idx: gm[pool[i]].add(m)
    null_counts.append(sum(1 for ms in gm.values() if len(ms) >= THRESH))
    if (it + 1) % 50 == 0: print(f"  perm {it+1}/{NPERM}: null mean {np.mean(null_counts):.1f}", flush=True)

nc = np.array(null_counts, float)
z = (len(real_hits) - nc.mean()) / (nc.std() or 1)
out = {"threshold_models": THRESH, "n_perm": NPERM, "annotator": "permissive top-5, 5 DBs (as on the site)",
       "real_n_genes": len(real_hits),
       "null_mean": round(float(nc.mean()), 1), "null_sd": round(float(nc.std()), 1),
       "null_min": int(nc.min()), "null_max": int(nc.max()),
       "z": round(float(z), 2), "fold": round(len(real_hits) / max(nc.mean(), 1e-9), 2),
       "p_emp": round((int((nc >= len(real_hits)).sum()) + 1) / (NPERM + 1), 4),
       "real_genes": real_hits[:40], "n_unannotated_per_model": N_UNANN}
json.dump(out, open(f"{C}/novel_null.json", "w"), indent=1)
print(f"\n=== NEW-BIOLOGY SHORTLIST vs FEATURE-SUBSET NULL ===")
print(f"REAL {out['real_n_genes']}  vs  NULL {out['null_mean']}±{out['null_sd']} "
      f"(range {out['null_min']}–{out['null_max']})  ->  z={out['z']}, {out['fold']}x, p={out['p_emp']}")
print("==> novel_null.json", flush=True)
