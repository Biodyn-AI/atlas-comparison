#!/usr/bin/env python
"""Regenerate the supplementary tables that depend on computed results, from those results.

Table S8 had drifted: it still carried the 20-permutation folds after the manuscript moved to 250,
so the supplement contradicted the main text. Tables S9 and S10 were cited by Section 3.4 but had
never been written at all -- S9 is the paper's actual deliverable, the released prediction list.
Generating them from the JSONs means they cannot drift again.
    python scripts/make_supp_tables.py   (rewrites the S8/S9/S10 blocks in SUPPLEMENTARY.md)
"""
from __future__ import annotations
import json, re

C = "/Users/annaantipova/Desktop/biomech/outputs/atlas/comparative"
SUP = f"{C}/SUPPLEMENTARY.md"
DISP = {"AIDO": "AIDO.Cell", "C2S": "C2S-Scale", "Geneformer": "Geneformer-V2", "Tahoe": "Tahoe-x1"}
nm = lambda m: DISP.get(m, m)


def load(n): return json.load(open(f"{C}/{n}"))


DPB, SM = load("depth_backbone.json"), load("hypothesis_sizematched.json")
FIN, TR2 = load("hypothesis_final.json"), load("hypothesis_trrust2.json")
PAIRS = load("hypothesis_pairs_full.json")

# ---- S8: depth robustness, now at 250 permutations -------------------------
s8 = [f"""## Table S8 — Backbone robustness to layer choice *(reviewer pre-empt)*

Calibrated backbone (top-10, a≥3, GO/Reactome/KEGG ≤200, no PPI, BH<0.05) recomputed at five depth fractions, matching
each model's layer by index round(f·(n−1)); real vs random-gene null (N={DPB['n_perm']}). The ≥8/10 backbone is large and
highly significant at **every** depth — it is not a mid-layer artefact — while its exact concept membership drifts with
depth (Jaccard vs the mid set), as expected for a layered representation. Observed counts are identical to the earlier
20-permutation run; only the null estimates changed. (`depth_backbone_fast.py` → `depth_backbone.json`.)

| Depth f | ≥7 (fold) | ≥8 (fold) | ≥9 (fold) | ≥8 p | ≥8 Jaccard vs mid |
|---|---|---|---|---|---|"""]
for r in DPB["depths"]:
    t, lab = r["tiers"], f"{r['frac']:.2f}" + (" (mid)" if r["frac"] == 0.5 else "")
    s8.append(f"| {lab} | {t['>=7']['real']} ({t['>=7']['fold']}×) | {t['>=8']['real']} ({t['>=8']['fold']}×) "
              f"| {t['>=9']['real']} ({t['>=9']['fold']}×) | {t['>=8']['p_emp']} | {r['jaccard_vs_mid']:.2f} |")

# ---- S9: the released prediction list --------------------------------------
pairs = PAIRS["pairs"]
s9 = [f"""## Table S9 — Released predictions: corroborated gene pairs absent from every database

The {len(pairs):,} pairs of Section 3.4: predicted by ≥2 of the models, and present in neither TRRUST nor STRING nor any
curated GO/Reactome/KEGG set of ≤200 genes. **These are prioritised hypotheses, not results.** The calibration that
gives them their expected value is the enrichment in Fig 5B: at ≥2 models the precision against held-out TRRUST is
{100*FIN['cross_model_curve']['2']['precision']:.2f} %, i.e. roughly one correct regulatory edge per
{round(1/FIN['cross_model_curve']['2']['precision'])} pairs proposed — {TR2['cross_model_curve']['2']['fold']}× chance, but
sparse in absolute terms. The full list is `data/hypothesis_pairs_full.json`; the twenty highest-weight pairs follow.
Weight is the number of features in which the pair co-occurs, summed over the models that predict it.

| Gene A | Gene B | Models | Weight | Predicted by |
|---|---|---|---|---|"""]
for c in pairs[:20]:
    s9.append(f"| {c['pair'][0]} | {c['pair'][1]} | {c['n_models']} | {c['weight']} | "
              f"{', '.join(nm(m) for m in c['models'])} |")

# ---- S10: equal-budget re-scoring ------------------------------------------
B = SM["budget"]
s10 = [f"""## Table S10 — Per-model recovery at an equal prediction budget *(capability vs capacity)*

How many pairs a model can offer is set by its SAE dictionary (4 × d_model) and its layer count, not by anything under
our control, and spans four orders of magnitude. Recovered-edge counts track graph size at r = 0.90, so the full-size
ranking of Fig 5A partly ranks capacity. Here every model is re-scored on its **{B:,} highest-weight pairs** — the
smallest graph among the models that pass at full size — each with its own configuration-model null. *rand-N* repeats
the same with {SM['n_draws']} random draws of {B:,} pairs instead of the top-weighted ones, isolating size from weighting.
Models whose whole graph is smaller than the budget are untestable at it, not failures. Rank correlation with the
full-size ordering is {SM['spearman_full_vs_matched']}; the ranking is **not** preserved. (`hypothesis_sizematched.py`.)

| Model | Graph (pairs) | Full-size fold | Equal-budget fold | p | rand-N fold | Verdict |
|---|---|---|---|---|---|---|"""]
def fold(x):
    """a fold is undefined when nothing was recovered -- say so rather than printing 0 or None"""
    return f"{x}×" if x else "—"


order = sorted(SM["models"], key=lambda m: -((SM["models"][m].get("topN") or {}).get("fold") or -1))
for m in order:
    d = SM["models"][m]
    full = TR2["per_model"][m]["fold"] or None
    if not d.get("testable"):
        s10.append(f"| {nm(m)} | {d['graph']:,} | {fold(full)} | — | — | — | "
                   f"untestable — graph smaller than the budget |")
        continue
    t, r_ = d["topN"], d["randN"]
    if not t["fold"]:
        v = "recovers nothing at either size"
    elif t["p_emp"] > 0.05:
        v = "**loses significance** at equal budget" if (full or 0) > 5 else "not significant either way"
    elif (full or 0) <= 5 and TR2["per_model"][m]["p_emp"] > 0.05:
        v = "**gains significance** at equal budget"
    else:
        v = "robust"
    eb = f"**{t['fold']}×**" if t["fold"] else "—"
    rn = f"{r_['fold_mean']} ± {r_['fold_sd']}×" if r_["fold_mean"] else "—"
    s10.append(f"| {nm(m)} | {d['graph']:,} | {fold(full)} | {eb} | {t['p_emp']} | {rn} | {v} |")

# Rebuild by section, so running this twice cannot duplicate anything: drop whatever S8/S9/S10 are
# already there, add the freshly generated ones, and sort the numbered tables back into sequence.
# (The file had also drifted out of order -- S7 preceded S6, and S6 sat after the figures.)
GENERATED = {8: "\n".join(s8), 9: "\n".join(s9), 10: "\n".join(s10)}
head, *rest = re.split(r"\n(?=## )", open(SUP, encoding="utf-8").read())
tables, other = dict(GENERATED), []
for sec in rest:
    m = re.match(r"## Table S(\d+)", sec)
    if not m:
        other.append(sec)
    elif int(m.group(1)) not in GENERATED:
        tables[int(m.group(1))] = sec
ordered = [tables[k] for k in sorted(tables)] + other
open(SUP, "w", encoding="utf-8").write(
    head.rstrip() + "\n\n" + "\n\n".join(x.rstrip() for x in ordered) + "\n")
print(f"S8: {len(DPB['depths'])} depths at N={DPB['n_perm']}")
print(f"S9: {len(pairs):,} pairs, top 20 tabulated")
print(f"S10: {sum(1 for m in SM['models'] if SM['models'][m].get('testable'))} models at budget {B:,}, "
      f"{sum(1 for m in SM['models'] if not SM['models'][m].get('testable'))} untestable")
print("==> SUPPLEMENTARY.md")
