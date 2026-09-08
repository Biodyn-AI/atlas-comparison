#!/usr/bin/env python
"""Does the top-N gene cutoff change annotation? Re-annotate every model's mid layer at top-5/10/15/20 genes
with the same 5-DB Fisher+BH annotator, and record annotation rate + distinct concepts + median odds-ratio.
    python scripts/topn_sweep.py   -> outputs/atlas/comparative/topn_sweep.json
"""
from __future__ import annotations
import json, glob
import numpy as np
from collections import defaultdict, Counter
from scipy.stats import fisher_exact

BASE = "/Users/annaantipova/Desktop/biomech"
G = f"{BASE}/outputs/atlas/genesets"; TS = f"{BASE}/outputs/atlas/ts3_out"; C = f"{BASE}/outputs/atlas/comparative"
MIN, MAX, ALPHA = 5, 500, 0.05
CUTOFFS = [5, 10, 15, 20]
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]

print("loading gene sets...", flush=True)
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


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.arange(1, n + 1)
    q = np.minimum.accumulate((p[o] * n / r)[::-1])[::-1]; out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


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
            recs.append((fid, f"{k[0]}:{k[1]}", float(orr), float(p)))
    if not recs:
        return {}, []
    q = bh([r[3] for r in recs]); ann = defaultdict(list); ors = []
    for i, (fid, term, orr, p) in enumerate(recs):
        if q[i] <= ALPHA:
            ann[fid].append(term); ors.append(orr)
    return ann, ors


out = {"cutoffs": CUTOFFS, "models": {}}
for m in ORDER:
    cats = sorted(glob.glob(f"{TS}/{m}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])
    if not cats:
        continue
    cat = json.load(open(cats[len(cats) // 2])); feats_in = cat["features"]; nfeat = cat.get("n_alive", len(feats_in))
    per = []
    for TOP in CUTOFFS:
        feats = {fid: [str(x).upper() for x in f["top_genes"]][:TOP] for fid, f in feats_in.items()}
        ann, ors = annotate(feats)
        terms = set(t for v in ann.values() for t in v)
        per.append({"top": TOP, "annot_rate": round(100 * len(ann) / max(nfeat, 1), 1),
                    "n_concepts": len(terms), "med_or": round(float(np.median(ors)), 2) if ors else 0})
        print(f"  {m} top{TOP}: {per[-1]['annot_rate']}% · {per[-1]['n_concepts']} concepts · medOR {per[-1]['med_or']}", flush=True)
    out["models"][m] = per
json.dump(out, open(f"{C}/topn_sweep.json", "w"))
print("==> topn_sweep.json", flush=True)
