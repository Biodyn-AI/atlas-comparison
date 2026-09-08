#!/usr/bin/env python
"""Cross-model gene-search index for the atlas (genes-data block). For every gene: which of the N
models encode a feature with that gene among its mid-layer top decoder genes, how many features,
the dominant (non-TF) concept, and an example feature id. Structural only — TF-regulon terms are
excluded from the concept table. Rebuilds genes_ts3.json for all models in matrix_ts3_string.json.
    python scripts/genes_search_ts3.py
"""
from __future__ import annotations
import json, glob, os
from collections import defaultdict, Counter

BASE = "/Users/annaantipova/Desktop/biomech"
TS = f"{BASE}/outputs/atlas/ts3_out"; C = f"{BASE}/outputs/atlas/comparative"
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
TOPG, CAP = 10, 24


def clean(t):
    return (t.replace("GO_BP:", "").replace("Reactome:", "").replace("KEGG:", "")
            .replace("STRING:", "PPI ").split(" (GO:")[0].split(" (R-HSA")[0]
            .split(" (hsa")[0].replace("Homo sapiens ", "").strip())


def main():
    matrix = json.load(open(f"{C}/matrix_ts3_string.json"))
    models = [m for m in ORDER if m in matrix]
    midx = {m: i for i, m in enumerate(models)}
    concepts, cidx = [], {}

    def cid(term):
        if term not in cidx:
            cidx[term] = len(concepts); concepts.append(term)
        return cidx[term]

    # gene -> model -> [count, Counter(concepts), example_fid]
    gm = defaultdict(lambda: defaultdict(lambda: [0, Counter(), None]))
    for m in models:
        mid = matrix[m]["layer"]; ft = matrix[m]["feat_terms"]
        cat = None
        for cp in glob.glob(f"{TS}/{m}/feature_catalog_L*.json"):
            j = json.load(open(cp))
            if j.get("layer") == mid:
                cat = j; break
        if cat is None:
            print(f"  ! {m}: no catalog at L{mid} — skipped"); continue
        feats = cat["features"]
        for fid, f in feats.items():
            genes = [str(g).upper() for g in f.get("top_genes", [])[:TOPG]]
            terms = [t for t in ft.get(fid, []) if not t.startswith("TRRUST:")]  # drop TF regulons
            concept = clean(terms[0]) if terms else (genes[0] + " (top gene)" if genes else "?")
            for g in genes:
                rec = gm[g][m]; rec[0] += 1; rec[1][concept] += 1
                if rec[2] is None:
                    rec[2] = int(fid) if str(fid).isdigit() else fid

    index = {}
    for g, ms in gm.items():
        rows = []
        for m, (n, cc, ex) in ms.items():
            dom = cc.most_common(1)[0][0] if cc else "?"
            rows.append([midx[m], n, cid(dom), ex])
        rows.sort(key=lambda r: -r[1])
        index[g] = rows[:CAP]

    out = {"models": models, "concepts": concepts, "index": index, "n_genes": len(index)}
    json.dump(out, open(f"{C}/genes_ts3.json", "w"))
    tf = sum(1 for c in concepts if c.startswith("TF "))
    print(f"==> genes_ts3.json ({len(models)} models, {len(index)} genes, {len(concepts)} concepts, TF={tf})")


if __name__ == "__main__":
    main()
