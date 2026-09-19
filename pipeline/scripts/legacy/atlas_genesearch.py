#!/usr/bin/env python
"""Cross-model gene index (the prior work's per-model Gene Search -> comparative). For every gene,
which of the 9 models encode a feature with that gene among its top decoder genes, which
feature, and under what dominant concept. Answers "is gene X represented everywhere or only
in some architectures?" — a view no single-model atlas can give.

    python scripts/atlas_genesearch.py -> outputs/atlas/comparative/genes_index.json
"""
from __future__ import annotations

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import json, os
from collections import defaultdict

BASE = _B
CL = f"{BASE}/outputs/atlas/cluster/out"
M = json.load(open(f"{BASE}/outputs/atlas/comparative/matrix_v2.json"))
OUT = f"{BASE}/outputs/atlas/comparative/genes_index.json"
TOPG = 10          # genes per feature indexed
CAP = 24           # max features listed per gene

ROSTER = {
    "scPRINT": f"{BASE}/outputs/atlas/scPRINT/feature_catalog_L4.json",
    "AIDO": f"{CL}/AIDO/feature_catalog_L08.json",
    "UCE": f"{CL}/UCE/feature_catalog_L03.json",
    "tGPT": f"{CL}/tGPT/feature_catalog_L08.json",
    "Geneformer": f"{BASE}/outputs/atlas/Geneformer/feature_catalog_L4.json",
    "scGPT": f"{BASE}/outputs/atlas/scGPT/feature_catalog_L6.json",
    "C2S": f"{BASE}/outputs/atlas/C2S/feature_catalog_L6.json",
    "MaxToki": f"{BASE}/outputs/atlas/MaxToki/feature_catalog_L6.json",
    "Novae": f"{BASE}/outputs/atlas/Novae/feature_catalog_L0.json",
}
MODELS = list(ROSTER)
MIDX = {m: i for i, m in enumerate(MODELS)}


def clean(t):
    return (t.replace("GO_BP:", "").replace("Reactome:", "").replace("KEGG:", "")
            .replace("TRRUST:", "TF ").split(" (GO:")[0].split(" (R-HSA")[0]
            .split(" (hsa")[0].replace("Homo sapiens ", "").strip())


concepts = []          # concept string table
cidx = {}
def cid(term):
    if term not in cidx:
        cidx[term] = len(concepts); concepts.append(term)
    return cidx[term]


# gene -> model -> {count, concepts Counter, example feat id}
from collections import Counter
gm = defaultdict(lambda: defaultdict(lambda: [0, Counter(), None]))
per_model_genes = {m: set() for m in MODELS}

for m, path in ROSTER.items():
    d = json.load(open(path))
    feats = d["features"] if "features" in d else d
    ft = M.get(m, {}).get("feat_terms", {})
    for fid, f in feats.items():
        genes = [str(g).upper() for g in f.get("top_genes", [])[:TOPG]]
        terms = ft.get(fid, [])
        if terms:
            concept = clean(terms[0])
        elif genes:                          # fallback: no enrichment -> the feature's strongest gene
            concept = f"{genes[0]} (top gene)"
        else:
            concept = "(unannotated)"
        for g in genes:
            per_model_genes[m].add(g)
            rec = gm[g][m]
            rec[0] += 1; rec[1][concept] += 1
            if rec[2] is None:
                rec[2] = int(fid) if str(fid).isdigit() else fid

# index: gene -> [[modelIdx, nFeatures, topConceptIdx, exampleFeatId], ...] (one row per model)
index = {}
for g, per in gm.items():
    rows = []
    for m, (cnt, ccounter, ex) in per.items():
        top_concept = ccounter.most_common(1)[0][0]
        rows.append([MIDX[m], cnt, cid(top_concept), ex])
    rows.sort(key=lambda r: -r[1])
    index[g] = rows

coverage = {g: len(per) for g, per in gm.items()}
out = {"models": MODELS, "concepts": concepts, "index": index,
       "n_genes": len(index),
       "model_gene_counts": {m: len(s) for m, s in per_model_genes.items()}}
json.dump(out, open(OUT, "w"))
print(f"genes indexed: {len(index)} | concepts: {len(concepts)} | size {os.path.getsize(OUT)//1024} KB")
print("max models per gene:", max(coverage.values()),
      "| genes in >=5 models:", sum(1 for v in coverage.values() if v >= 5))
print("per-model vocab sizes:", out["model_gene_counts"])
for g in ["CD3D", "GATA1", "ACTB", "MT-ND4", "HBB", "EPCAM"]:
    if g in index:
        ms = sorted(MODELS[r[0]] for r in index[g])
        print(f"  {g}: {coverage[g]}/9 models -> {ms}")
