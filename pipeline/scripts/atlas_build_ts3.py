#!/usr/bin/env python
"""Build the updated comparative matrix from the TS-3-tissue re-run (review feedback applied:
3 Tabula Sapiens tissues, depth-matched layers 0/25/50/75/100%, top-5 annotation, no perturbation,
no Novae). Uses the MID (50%-depth) layer of each model for a depth-matched cross-model comparison,
straight from the cluster's precomputed top-5 annotations. Also emits a per-layer depth profile.

    python scripts/atlas_build_ts3.py
-> outputs/atlas/comparative/matrix_ts3.json  + depth_ts3.json
"""
from __future__ import annotations

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import json, os, glob
import numpy as np
from collections import Counter

BASE = _B
TS = f"{BASE}/outputs/atlas/ts3_out"
OUT = f"{BASE}/outputs/atlas/comparative"

AXIS = {
    "AIDO": "expression (all genes)", "UCE": "ESM protein-token",
    "tGPT": "autoregressive rank", "Geneformer": "rank-MLM", "scGPT": "expr-MLM",
    "C2S": "cell-sentence LLM (Gemma-2-2B)", "MaxToki": "rank autoregressive (Llama)",
}
PARAMS = {"AIDO": "10M", "UCE": "650M", "tGPT": "~120M", "Geneformer": "316M",
          "scGPT": "~50M", "C2S": "2B", "MaxToki": "217M"}


def clean(t):
    return (t.replace("GO_BP:", "").replace("Reactome:", "").replace("KEGG:", "")
            .replace("TRRUST:", "TF ").split(" (GO:")[0].split(" (R-HSA")[0]
            .split(" (hsa")[0].replace("Homo sapiens ", "").strip())


def load_ann(model, layer):
    p = f"{TS}/annotations/{model}_L{layer:02d}_annotations.json"
    return json.load(open(p))


def model_layers(model):
    cats = sorted(glob.glob(f"{TS}/{model}/feature_catalog_L*.json"),
                  key=lambda p: json.load(open(p))["layer"])
    return [json.load(open(p)) for p in cats]


matrix, depth = {}, {}
models = sorted(AXIS)
for m in models:
    cats = model_layers(m)
    layers = [c["layer"] for c in cats]
    mid = layers[len(layers) // 2]                      # 50%-depth layer
    ann = load_ann(m, mid)["annotations"]
    tc, feat_terms = Counter(), {}
    for fid, lst in ann.items():
        terms = [f"{a['source']}:{a['term']}" for a in lst]
        feat_terms[fid] = terms
        for t in set(terms):
            tc[t] += 1
    cat_mid = next(c for c in cats if c["layer"] == mid)
    matrix[m] = {"n_feat": cat_mid["n_alive"], "n_annot": len(ann), "layer": mid,
                 "axis": AXIS[m], "params": PARAMS[m],
                 "var_explained": round(1 - cat_mid.get("fvu", 0), 3),
                 "term_count": dict(tc), "feat_terms": feat_terms}
    # depth profile across all 5 layers
    prof = []
    for c in cats:
        a = load_ann(m, c["layer"])
        na, n = a["n_annotated"], a["n_features"]
        tot = sum(len(v) for v in a["annotations"].values())
        prof.append({"layer": c["layer"], "rate": round(100 * na / max(n, 1), 1),
                     "rich": round(tot / max(na, 1), 1),
                     "var_explained": round(1 - c.get("fvu", 0), 3),
                     "dead": round(c.get("dead_frac", 0), 3)})
    depth[m] = prof
    print(f"{m:11s} mid-L{mid:<2} feat={matrix[m]['n_feat']:5d} annot={matrix[m]['n_annot']:5d} "
          f"({100*matrix[m]['n_annot']/max(matrix[m]['n_feat'],1):4.0f}%) concepts={len(tc):4d} "
          f"VarExpl={matrix[m]['var_explained']}")

os.makedirs(OUT, exist_ok=True)
json.dump(matrix, open(f"{OUT}/matrix_ts3.json", "w"))
json.dump(depth, open(f"{OUT}/depth_ts3.json", "w"))
print(f"\nwrote matrix_ts3.json ({len(models)} models, depth-matched mid layer) + depth_ts3.json")

# quick headline check
from collections import defaultdict
cm = defaultdict(set)
for m in models:
    for t in matrix[m]["term_count"]:
        cm[t].add(m)
regs = set(t.split(":", 1)[1] for t in cm if t.startswith("TRRUST:"))
print(f"distinct concepts={len(cm)} | TF regulons={len(regs)} = {100*len(regs)/len(cm):.2f}% | "
      f"universal core (all {len(models)})={sum(1 for v in cm.values() if len(v)==len(models))}")
