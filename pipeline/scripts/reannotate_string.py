#!/usr/bin/env python
"""Re-annotate the TS-3-tissue catalogs at top-5 with STRING added (5 databases, matching Igor:
GO_BP/Reactome/KEGG + TRRUST + STRING). Rebuilds matrix_ts3 (mid layer) + depth_ts3 (all layers)
+ atlas_data_ts3 from the STRING-augmented annotations. Inverted gene->term index for speed.
    python scripts/reannotate_string.py
"""
from __future__ import annotations
import json, glob, os
import numpy as np
from collections import defaultdict, Counter
from scipy.stats import fisher_exact

BASE = "/Users/annaantipova/Desktop/biomech"
G = f"{BASE}/outputs/atlas/genesets"; TS = f"{BASE}/outputs/atlas/ts3_out"; C = f"{BASE}/outputs/atlas/comparative"
MIN, MAX, ALPHA, TOP = 5, 500, 0.05, 5
AXIS = {"AIDO": "expression (all genes)", "UCE": "ESM protein-token", "tGPT": "autoregressive rank",
        "Geneformer": "rank-MLM", "scGPT": "expr-MLM", "C2S": "cell-sentence LLM (Gemma-2-2B)", "MaxToki": "temporal Llama",
        "Tahoe": "expression MLM (MosaicX 3B)", "scFoundation": "read-depth MAE (100M)", "GeneCompass": "knowledge/GRN-prior BERT (104M)"}
PARAMS = {"AIDO": "10M", "UCE": "650M", "tGPT": "~50M", "Geneformer": "316M", "scGPT": "~50M", "C2S": "2B", "MaxToki": "217M", "Tahoe": "3B", "scFoundation": "100M", "GeneCompass": "104M"}


def clean(t):
    return (t.replace("GO_BP:", "").replace("Reactome:", "").replace("KEGG:", "")
            .replace("TRRUST:", "TF ").replace("STRING:", "PPI ").split(" (GO:")[0]
            .split(" (R-HSA")[0].split(" (hsa")[0].replace("Homo sapiens ", "").strip())


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.arange(1, n + 1)
    q = np.minimum.accumulate((p[o] * n / r)[::-1])[::-1]; out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


print("loading gene sets (5 DBs incl STRING)...", flush=True)
gs = {nm: {t: set(g) for t, g in json.load(open(f"{G}/{nm}_gene_sets.json")).items() if MIN <= len(g) <= MAX}
      for nm in ("GO_BP", "KEGG", "Reactome")}
tr = {t: set(v) for t, v in json.load(open(f"{G}/trrust_edges.json")).items() if MIN <= len(v) <= MAX}
st = {t: set(v) for t, v in json.load(open(f"{G}/string_edges.json")).items() if MIN <= len(v) <= MAX}
bg = set(json.load(open(f"{G}/background.json"))); bg |= set().union(*st.values()) | set(st); bgn = len(bg)
idx = defaultdict(list); size = {}
for s, d in list(gs.items()) + [("TRRUST", tr), ("STRING", st)]:
    for t, genes in d.items():
        k = (s, t); size[k] = len(genes)
        for g in genes:
            idx[g].append(k)
print(f"index: {len(idx)} genes, {len(size)} terms (STRING {len(st)}); bg {bgn}", flush=True)


def annotate(feats):
    recs = []
    for fid, genes in feats.items():
        g = set(genes) & bg
        if len(g) < 3:
            continue
        cand = Counter()
        for gene in g:
            for k in idx.get(gene, ()):
                cand[k] += 1
        for k, a in cand.items():
            if a < 2:
                continue
            tg = size[k]; b = len(g) - a; c = tg - a; d = bgn - a - b - c
            orr, p = fisher_exact([[a, b], [c, d]], alternative="greater")
            recs.append((fid, k[0], k[1], float(orr), float(p)))
    if not recs:
        return {}
    q = bh([r[4] for r in recs]); ann = defaultdict(list)
    for i, (fid, s, t, orr, p) in enumerate(recs):
        if q[i] <= ALPHA:
            ann[fid].append((s, t, orr, float(q[i])))
    return ann


models = sorted(AXIS)
matrix, depth = {}, {}
for m in models:
    cats = sorted(glob.glob(f"{TS}/{m}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])
    layers = [json.load(open(p))["layer"] for p in cats]
    mid = layers[len(layers) // 2]
    prof = []
    for cp in cats:
        cat = json.load(open(cp)); L = cat["layer"]; feats_in = cat["features"]
        feats = {fid: [str(x).upper() for x in f["top_genes"]][:TOP] for fid, f in feats_in.items()}
        ann = annotate(feats)
        na = len(ann); n = len(feats)
        tot = sum(len(v) for v in ann.values())
        prof.append({"layer": L, "rate": round(100 * na / max(n, 1), 1), "rich": round(tot / max(na, 1), 1),
                     "var_explained": round(1 - cat.get("fvu", 0), 3), "dead": round(cat.get("dead_frac", 0), 3)})
        if L == mid:
            tc = Counter(); ft = {}
            for fid, lst in ann.items():
                terms = [f"{s}:{t}" for s, t, _, _ in lst]; ft[fid] = terms
                for t in set(terms):
                    tc[t] += 1
            matrix[m] = {"n_feat": cat["n_alive"], "n_annot": na, "layer": mid, "axis": AXIS[m], "params": PARAMS[m],
                         "var_explained": round(1 - cat.get("fvu", 0), 3), "term_count": dict(tc), "feat_terms": ft}
        print(f"  {m} L{L}: {na}/{n} ({100*na/max(n,1):.0f}%)", flush=True)
    depth[m] = prof
json.dump(matrix, open(f"{C}/matrix_ts3_string.json", "w"))
json.dump(depth, open(f"{C}/depth_ts3_string.json", "w"))
# headline
cm = defaultdict(set)
for m in models:
    for t in matrix[m]["term_count"]:
        cm[t].add(m)
regs = set(t.split(":", 1)[1] for t in cm if t.startswith("TRRUST:"))
print(f"\nSTRING re-annotation done. concepts={len(cm)} TF-regulons={len(regs)}={100*len(regs)/len(cm):.2f}% "
      f"core(all7)={sum(1 for v in cm.values() if len(v)==len(models))}", flush=True)
print("wrote matrix_ts3_string.json + depth_ts3_string.json", flush=True)
