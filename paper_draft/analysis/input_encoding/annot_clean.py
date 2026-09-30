#!/usr/bin/env python
"""Side check for notes/11_input_encoding.md: does the tie-tail artefact change the calibrated
annotator's per-model concepts or the headline shared backbone (78 concepts in >= 8 of 10 models)?

Annotator, gene sets, background, mid-layer choice and backbone counting are copied verbatim from
pipeline/scripts/stats_final.py (gs / BG / bh / annotate / feats_mid / spectrum_ge): top-10 genes,
>= 3 genes in a GO_BP / KEGG / Reactome set of 5..200 genes, hypergeometric test against background.json,
BH q <= 0.05 over all (feature, term) tests of the model.
Variants: raw = published top_genes; clean = top_genes with the tie-tail slots removed
(ie_common.tie_tail_with_firing), then the first 10 of what is left.
The random-gene null of stats_final.py does not depend on the real gene lists (only on the number
of features, unchanged here), so the clean backbone is compared with the stored null
(stats_final.json:backbone_null).
    ATLAS_BASE=<copy> python annot_clean.py -> annot_clean.json
"""
from __future__ import annotations

import json
from collections import defaultdict, Counter
import numpy as np
from scipy.stats import hypergeom
from ie_common import *

ALPHA, TOP, AMIN, MX = 0.05, 10, 3, 200
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
    """stats_final.annotate, plus the number of features with >= 1 significant term."""
    A = []; TG = []; LG = []; TERM = []; FID = []
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
            A.append(a); TG.append(size[k]); LG.append(len(g)); TERM.append(k); FID.append(fi)
    if not A:
        return set(), 0
    q = bh(hypergeom.sf(np.array(A) - 1, BGN, np.array(TG), np.array(LG)))
    sig = np.where(q <= ALPHA)[0]
    return set(TERM[i] for i in sig), len(set(FID[i] for i in sig))


def spectrum_ge(sets, ks):
    cm = Counter()
    for s in sets.values():
        for t in s:
            cm[t] += 1
    return {k: sum(1 for v in cm.values() if v >= k) for k in ks}, cm


real, clean, per = {}, {}, {}
for m in ORDER:
    d = load_model(m); names = d["names"]; rank = universe_ranks(names)
    tail, _, _ = tie_tail_with_firing(d["T20"], rank, d["nfire"])
    mid = mid_layer(m); rows = np.where(d["layer"] == mid)[0]
    raw_lists = [[names[k] for k in d["T20"][i] if k >= 0] for i in rows]
    cln_lists = [[names[k] for k in d["T20"][i, :20 - tail[i]] if k >= 0] for i in rows]
    real[m], nr = annotate(raw_lists); clean[m], nc = annotate(cln_lists)
    per[m] = {"mid_layer": mid, "n_features": len(rows),
              "features_tail_into_top10": int((tail[rows] >= 11).sum()),
              "annotated_features_raw": nr, "annotated_features_clean": nc,
              "terms_raw": len(real[m]), "terms_clean": len(clean[m]),
              "terms_lost": sorted(real[m] - clean[m])[:25], "n_terms_lost": len(real[m] - clean[m]),
              "n_terms_gained": len(clean[m] - real[m])}
    print(m, per[m]["n_features"], "tie->top10", per[m]["features_tail_into_top10"], "annotated", nr, "->", nc,
          "terms", len(real[m]), "->", len(clean[m]), flush=True)
KS = [7, 8, 9, 10]
rg, rc = spectrum_ge(real, KS); cg, cc = spectrum_ge(clean, KS)
b8r = {t for t, v in rc.items() if v >= 8}; b8c = {t for t, v in cc.items() if v >= 8}
stored = json.load(open(f"{C}/stats_final.json"))["backbone_null"]
out = {"annotator": "stats_final.py (top10, >=3 genes, GO/KEGG/Reactome 5..200, BH q<=0.05), mid layer",
       "per_model": per, "backbone_raw": {f">={k}": rg[k] for k in KS}, "backbone_clean": {f">={k}": cg[k] for k in KS},
       "backbone_ge8_lost": sorted(b8r - b8c), "backbone_ge8_gained": sorted(b8c - b8r),
       "stored_null": {k: {"null_mean": v["null_mean"], "null_sd": v["null_sd"], "real_published": v["real"]} for k, v in stored.items()}}
json.dump(out, open(f"{OUTDIR}/annot_clean.json", "w"), indent=1)
print("backbone raw", out["backbone_raw"], "clean", out["backbone_clean"], "lost", out["backbone_ge8_lost"])
