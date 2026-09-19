#!/usr/bin/env python
"""Analyses on the uniform v2 comparative matrix (top-5, 9 models). Produces the data
JSON that the interactive atlas artifact renders. Cross-MODEL analogues of the prior
cross-LAYER atlas sections.

Analyses:
  A. per-model coverage (n_feat, n_annot, %annot, #concepts, source composition)
  B. universality spectrum — how many models share each concept (1..9)
  C. concept x model heatmap matrix (top shared + top model-specific concepts)
  D. TF-regulon coverage — the null headline (are TRRUST regulons encoded? how many, universal?)
  E. model-specific concepts — what each model uniquely encodes (blind spots of the rest)
  F. feature-orthologs — pairwise model similarity in concept space (Jaccard on concept sets)

    python scripts/atlas_analyze_v2.py  -> outputs/atlas/comparative/atlas_data.json
"""
from __future__ import annotations

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import json, os
import numpy as np
from collections import Counter, defaultdict

BASE = _B
M = json.load(open(f"{BASE}/outputs/atlas/comparative/matrix_v2.json"))
OUT = f"{BASE}/outputs/atlas/comparative/atlas_data.json"
MODELS = list(M)


def src_of(term):
    return term.split(":", 1)[0]


# ---- A. per-model coverage ----
coverage = {}
for m in MODELS:
    tc = M[m]["term_count"]
    by_src = Counter(src_of(t) for t in tc)
    coverage[m] = {
        "axis": M[m].get("axis", ""), "layer": M[m]["layer"],
        "n_feat": M[m]["n_feat"], "n_annot": M[m]["n_annot"],
        "annot_rate": round(100 * M[m]["n_annot"] / max(M[m]["n_feat"], 1), 1),
        "n_concepts": len(tc),
        "by_source": {s: by_src.get(s, 0) for s in ["GO_BP", "Reactome", "KEGG", "TRRUST"]},
    }

# ---- B. universality spectrum ----
concept_models = defaultdict(set)
for m in MODELS:
    for t in M[m]["term_count"]:
        concept_models[t].add(m)
share = Counter(len(v) for v in concept_models.values())
universality = {str(k): share.get(k, 0) for k in range(1, len(MODELS) + 1)}
# the universal core (present in ALL models)
universal_core = sorted([t for t, v in concept_models.items() if len(v) == len(MODELS)])

# ---- C. concept x model heatmap ----
# rows = union of each model's top-8 concepts by feature-count, deduped, capped
top_terms = []
for m in MODELS:
    top_terms += [t for t, _ in Counter(M[m]["term_count"]).most_common(8)]
heat_terms = list(dict.fromkeys(top_terms))
Z = [[M[m]["term_count"].get(t, 0) for m in MODELS] for t in heat_terms]
heatmap = {"terms": heat_terms, "models": MODELS, "Z": Z}

# ---- D. TF-regulon coverage (the null) ----
regulon = {}
all_regulons = set()
for m in MODELS:
    tfs = sorted([t.split(":", 1)[1] for t in M[m]["term_count"] if t.startswith("TRRUST:")],
                 key=lambda tf: -M[m]["term_count"][f"TRRUST:{tf}"])
    all_regulons |= set(tfs)
    regulon[m] = {"n_regulons": len(tfs), "top": tfs[:12],
                  "pct_of_concepts": round(100 * len(tfs) / max(len(M[m]["term_count"]), 1), 1),
                  "n_regulon_features": sum(M[m]["term_count"][f"TRRUST:{tf}"] for tf in tfs)}
# regulon universality
reg_models = defaultdict(set)
for m in MODELS:
    for t in M[m]["term_count"]:
        if t.startswith("TRRUST:"):
            reg_models[t.split(":", 1)[1]].add(m)
reg_share = Counter(len(v) for v in reg_models.values())
regulon_universality = {str(k): reg_share.get(k, 0) for k in range(1, len(MODELS) + 1)}
regulon_shared_by = {tf: sorted(v) for tf, v in sorted(reg_models.items(), key=lambda kv: -len(kv[1]))}

# ---- E. model-specific concepts (uniquely encoded) ----
specific = {}
for m in MODELS:
    uniq = [(t, M[m]["term_count"][t]) for t in M[m]["term_count"] if concept_models[t] == {m}]
    uniq.sort(key=lambda x: -x[1])
    specific[m] = {"n_unique": len(uniq),
                   "top": [{"term": t, "count": c} for t, c in uniq[:12]]}

# ---- F. feature-orthologs — pairwise concept-set Jaccard ----
sets = {m: set(M[m]["term_count"]) for m in MODELS}
J = [[round(len(sets[a] & sets[b]) / max(len(sets[a] | sets[b]), 1), 3) for b in MODELS] for a in MODELS]
similarity = {"models": MODELS, "J": J}

data = {
    "models": MODELS,
    "n_models": len(MODELS),
    "totals": {
        "total_features": sum(M[m]["n_feat"] for m in MODELS),
        "total_annotated": sum(M[m]["n_annot"] for m in MODELS),
        "total_concepts": len(concept_models),
        "universal_core": len(universal_core),
        "total_regulons": len(all_regulons),
        "regulon_pct": round(100 * len(all_regulons) / max(len(concept_models), 1), 2),
    },
    "coverage": coverage,
    "universality": universality,
    "universal_core_terms": universal_core[:60],
    "heatmap": heatmap,
    "regulon": regulon,
    "regulon_universality": regulon_universality,
    "regulon_shared_by": dict(list(regulon_shared_by.items())[:40]),
    "specific": specific,
    "similarity": similarity,
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(data, open(OUT, "w"), indent=1)

# console summary
print(f"== Comparative SAE atlas — {len(MODELS)} models ==")
print(f"total features {data['totals']['total_features']:,} | annotated {data['totals']['total_annotated']:,} | "
      f"distinct concepts {data['totals']['total_concepts']:,}")
print(f"universal core (in ALL {len(MODELS)}): {data['totals']['universal_core']} concepts")
print(f"TF-regulons detected (any model): {data['totals']['total_regulons']} = "
      f"{data['totals']['regulon_pct']}% of concepts  <-- the null headline")
print("\nper-model:")
for m in MODELS:
    c = coverage[m]; r = regulon[m]
    print(f"  {m:11s} L{c['layer']:<2} feat={c['n_feat']:5d} annot={c['annot_rate']:4.0f}% "
          f"concepts={c['n_concepts']:4d}  TF-regulons={r['n_regulons']:3d} ({r['pct_of_concepts']:.1f}%)  "
          f"axis={c['axis']}")
print("\nuniversality spectrum (concepts shared by k models):")
for k in range(len(MODELS), 0, -1):
    print(f"  {k} models: {universality[str(k)]:5d} concepts")
print("\nregulon universality (TFs shared by k models):")
for k in range(len(MODELS), 0, -1):
    if regulon_universality[str(k)]:
        print(f"  {k} models: {regulon_universality[str(k)]} TFs")
print("\nwrote", OUT)
