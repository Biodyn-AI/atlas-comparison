#!/usr/bin/env python
"""Build the FINAL comparative SAE atlas (v2) — uniform, panel-fair.

9 models, ONE representative layer each, top-5 genes, ONE annotator (Fisher 'greater'
+ BH<0.05 vs GO_BP/KEGG/Reactome and TRRUST regulons, fixed background). Cluster-extracted
AIDO/UCE/tGPT (H100, 2000-cell corpus) + our scPRINT + the ingested prior Geneformer/scGPT/
C2S/MaxToki/Novae. Fast via inverted gene->term index (only real-overlap terms tested).

    python scripts/atlas_build_v2.py
-> outputs/atlas/comparative/matrix_v2.json  {model:{n_feat,n_annot,layer,term_count,feat_terms}}
"""
from __future__ import annotations

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import json, os, sys, glob
import numpy as np
from collections import defaultdict, Counter
from scipy.stats import fisher_exact

BASE = _B
CL = f"{BASE}/outputs/atlas/cluster/out"
GS_DIR = f"{BASE}/outputs/atlas/genesets"
OUT = f"{BASE}/outputs/atlas/comparative/matrix_v2.json"
TOP, MIN_TERM, MAX_TERM, ALPHA = 5, 5, 500, 0.05

ROSTER = [  # (model, layer, path, axis)
    ("scPRINT", 4, f"{BASE}/outputs/atlas/scPRINT/feature_catalog_L4.json", "ESM-augmented expr"),
    ("AIDO", 8, f"{CL}/AIDO/feature_catalog_L08.json", "expression (all genes)"),
    ("UCE", 3, f"{CL}/UCE/feature_catalog_L03.json", "ESM protein-token"),
    ("tGPT", 8, f"{CL}/tGPT/feature_catalog_L08.json", "autoregressive rank"),
    ("Geneformer", 4, f"{BASE}/outputs/atlas/Geneformer/feature_catalog_L4.json", "rank-MLM"),
    ("scGPT", 6, f"{BASE}/outputs/atlas/scGPT/feature_catalog_L6.json", "expr-MLM"),
    ("C2S", 6, f"{BASE}/outputs/atlas/C2S/feature_catalog_L6.json", "cell-sentence LLM"),
    ("MaxToki", 6, f"{BASE}/outputs/atlas/MaxToki/feature_catalog_L6.json", "tokeniser MLM"),
    ("Novae", 0, f"{BASE}/outputs/atlas/Novae/feature_catalog_L0.json", "spatial"),
]


def bh_fdr(pvals, alpha=ALPHA):
    p = np.asarray(pvals, float); n = len(p)
    order = np.argsort(p); ranks = np.arange(1, n + 1)
    q_sorted = np.minimum.accumulate((p[order] * n / ranks)[::-1])[::-1]
    q = np.empty(n); q[order] = np.clip(q_sorted, 0, 1)
    return q <= alpha, q


def load_genesets():
    gs = {}
    for name in ("GO_BP", "KEGG", "Reactome"):
        raw = json.load(open(f"{GS_DIR}/{name}_gene_sets.json"))
        gs[name] = {t: set(g) for t, g in raw.items() if MIN_TERM <= len(g) <= MAX_TERM}
    tr = {t: set(v) for t, v in json.load(open(f"{GS_DIR}/trrust_edges.json")).items()
          if MIN_TERM <= len(v) <= MAX_TERM}
    bg = set(json.load(open(f"{GS_DIR}/background.json")))
    return gs, tr, bg


def build_index(gs, tr):
    """gene -> list of (source, term). Includes TRRUST as source 'TRRUST'."""
    idx = defaultdict(list)
    sizes = {}
    for src, terms in gs.items():
        for term, genes in terms.items():
            key = (src, term); sizes[key] = len(genes)
            for g in genes:
                idx[g].append(key)
    for tf, tgts in tr.items():
        key = ("TRRUST", tf); sizes[key] = len(tgts)
        for g in tgts:
            idx[g].append(key)
    return idx, sizes


def annotate(feats, idx, sizes, bg):
    """Return {fid: [(source, term, OR, q)]} using inverted index; BH across all tested pairs."""
    bgn = len(bg)
    recs = []  # (fid, src, term, OR, p)
    for fid, genes in feats.items():
        g = set(genes) & bg
        if len(g) < 3:
            continue
        cand = Counter()
        for gene in g:
            for key in idx.get(gene, ()):
                cand[key] += 1
        for key, a in cand.items():
            if a < 2:
                continue
            tgN = sizes[key]
            b = len(g) - a; c = tgN - a; d = bgn - a - b - c
            orr, p = fisher_exact([[a, b], [c, d]], alternative="greater")
            recs.append((fid, key[0], key[1], float(orr), float(p)))
    if not recs:
        return {}
    keep, q = bh_fdr(np.array([r[4] for r in recs]))
    ann = defaultdict(list)
    for i, (fid, src, term, orr, p) in enumerate(recs):
        if keep[i]:
            ann[fid].append((src, term, orr, float(q[i])))
    for fid in ann:
        ann[fid].sort(key=lambda t: t[3])
    return ann


def main():
    gs, tr, bg = load_genesets()
    idx, sizes = build_index(gs, tr)
    print(f"index: {len(idx)} genes -> {len(sizes)} terms "
          f"(GO_BP {len(gs['GO_BP'])} KEGG {len(gs['KEGG'])} Reactome {len(gs['Reactome'])} TRRUST {len(tr)}); bg {len(bg)}",
          flush=True)
    matrix = {}
    for model, layer, path, axis in ROSTER:
        d = json.load(open(path))
        feats_in = d["features"] if "features" in d else d
        feats = {fid: [str(x).upper() for x in f["top_genes"]][:TOP] for fid, f in feats_in.items()}
        ann = annotate(feats, idx, sizes, bg)
        # term_count: # distinct features detecting each concept
        tc = Counter()
        feat_terms = {}
        for fid, lst in ann.items():
            terms = [f"{s}:{t}" for s, t, _, _ in lst]
            feat_terms[fid] = terms
            for term in set(terms):
                tc[term] += 1
        matrix[model] = {"n_feat": len(feats), "n_annot": len(ann), "layer": layer,
                         "axis": axis, "term_count": dict(tc), "feat_terms": feat_terms}
        print(f"  {model:11s} L{layer:<2} n_feat={len(feats):5d} annotated={len(ann):5d} "
              f"({100*len(ann)/max(len(feats),1):4.0f}%) concepts={len(tc)}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(matrix, open(OUT, "w"))
    print("==> wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
