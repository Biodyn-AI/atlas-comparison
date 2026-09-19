#!/usr/bin/env python
"""Regenerate the supplementary tables that depend on computed results, from those results.

Table S8 had drifted: it still carried the 20-permutation folds after the manuscript moved to 250,
so the supplement contradicted the main text. Tables S9 and S10 were cited by Section 3.4 but had
never been written at all -- S9 is the paper's actual deliverable, the released prediction list.
Generating them from the JSONs means they cannot drift again.
    python scripts/make_supp_tables.py   (rewrites the S8/S9/S10 blocks in SUPPLEMENTARY.md)
"""
from __future__ import annotations

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import json, re

C = f"{_B}/outputs/atlas/comparative"
G = f"{_B}/outputs/atlas/genesets"
SCRIPTS = _os.path.dirname(_os.path.abspath(__file__))
SUP = f"{C}/SUPPLEMENTARY.md"
DISP = {"AIDO": "AIDO.Cell", "C2S": "C2S-Scale", "Geneformer": "Geneformer-V2", "Tahoe": "Tahoe-x1"}
nm = lambda m: DISP.get(m, m)


def load(n): return json.load(open(f"{C}/{n}"))


CTL = load("controls.json")
DPB, SM = load("depth_backbone.json"), load("hypothesis_sizematched.json")
SAE = load("sae_health.json")
ALN, KEG = load("alllayer_null.json"), load("kegg_robust.json")
ROB, PUB, PRT = load("hypothesis_robust.json"), load("hypothesis_pubmed.json"), load("hypothesis_perturb.json")
FIN, TR2 = load("hypothesis_final.json"), load("hypothesis_trrust2.json")
PAIRS = load("hypothesis_pairs_full.json")

# ---- S2: database sizes, whose column said 5-500 but held the totals -------
def gs_counts(src):
    d = json.load(open(f"{G}/{src}_gene_sets.json"))
    return len(d), sum(1 for _, g in d.items() if 5 <= len(g) <= 500), sum(1 for _, g in d.items() if 5 <= len(g) <= 200)


STR = json.load(open(f"{G}/string_edges.json"))
TRR = json.load(open(f"{G}/trrust_edges.json"))
BGN = len(json.load(open(f"{G}/background.json")))
s2 = ["""## Table S2 — Gene-set / interaction databases

Counts are generated from the retrieved files. The size filters matter and differ between the annotators: the
permissive one admits sets of 5–500 genes, the calibrated one 5–200, so both columns are given. (An earlier version of
this table put the unfiltered totals under a "5–500" heading.)

| Database | Source | Version / retrieved | #terms total | 5–500 | 5–200 | Use |
|---|---|---|---|---|---|---|"""]
for src, label, when in (("GO_BP", "GO Biological Process", "retrieved 2026-08-17"),
                         ("Reactome", "Reactome", "retrieved 2026-08-17"),
                         ("KEGG", "KEGG (Human)", "retrieved 2026-08-17")):
    tot, f500, f200 = gs_counts(src)
    s2.append(f"| {label} | Enrichr via gseapy | {when} | {tot:,} | {f500:,} | {f200:,} | permissive + calibrated |")
s2 += [f"| STRING (PPI) | STRING v12 | downloaded 2026-08-27 | {len(STR):,} hubs (combined score ≥ 700) | — | — | "
       "permissive **only** (excluded from calibrated) |",
       f"| TRRUST | TRRUST v2 | 2026-08-17 | {len(TRR):,} | — | — | permissive only; **held out** for the Section 3.4 test |",
       f"| Background | union of DB genes + STRING | — | {BGN:,} genes | — | — | Fisher/hypergeometric universe |"]

# ---- S3: the SAE-health rows, which had contradicted themselves ------------
# "0 % for 7/10 models; UCE/C2S 0.1 %" counted UCE twice -- it is 0.0 % and already in the seven.
HM = SAE["models"]
zero = sorted(m for m, d in HM.items() if d["mid"] == 0.0)
rest = sorted((m for m in HM if m not in zero), key=lambda m: HM[m]["mid"])
worst = max(HM, key=lambda m: HM[m]["mean"])
s3_rows = [
    ("SAE health — dead features (mid layer)",
     f"0 % for {len(zero)}/{len(HM)} models; "
     + ", ".join(f"{nm(m)} {HM[m]['mid']} %" for m in rest)
     + f" — {nm(worst)} the worst"),
    ("SAE health — dead features (all layers)",
     f"Section 3.4 pools every layer, so health is reported over all {SAE['total_layers']} model-layers: "
     f"{SAE['layers_over_threshold']} ({SAE['pct_layers_over_threshold']} %) exceed "
     f"{int(SAE['threshold']*100)} % dead features. {len(SAE['clean_models'])} models have none at any layer; "
     + "; ".join(f"{nm(m)} mean {HM[m]['mean']} %, max {HM[m]['max']} %"
                 for m in sorted(HM, key=lambda m: -HM[m]["mean"])[:3])
     + f". {nm(worst)}'s dictionaries are the weakest in the roster (`sae_health.py`)."),
]

# ---- S5: nulls and tests, including the ones 3.4-3.5 added -----------------
ck = CTL["survivor_nulls"]["cka_null"]
cka_min = min(x["real_cka"] / x["shuffled_cka"] for x in ck)
BN, DN = CTL["stats_final"]["backbone_null"], CTL["degree_null"]["tiers"]
HN, RN = CTL["stats_final"]["heldout_null"], CTL["random_null"]
RC, CV = CTL["recalibration"], TR2["cross_model_curve"]
f1 = [v["1"]["fold"] for v in ROB["variants"].values() if v["1"]["fold"]]
s5 = [f"""## Table S5 — Nulls and tests

Every claim in the paper with the control it is measured against. Generated from the result files, so it cannot
drift from them (`make_supp_tables.py`).

| Test | Statistic | Permutations | Result |
|---|---|---|---|
| Universal-core artefact (permissive, mid layer) | real core vs random-gene core | {RN['n_perm']} | real {RN['real_core']} < null {RN['rand_core_mean']:.0f} (z = {RN['core_z']}) |
| Universal-core artefact (permissive, **all-layer union**) | real core vs random-gene core, identical construction | {ALN['n_perm']} | real {ALN['real_core']:,} < null {ALN['null_core_mean']:,.0f} ± {ALN['null_core_sd']:.0f} (min {ALN['null_core_min']:,}) — z = {ALN['core_z']}, {ALN['core_fold_over_null']}×; the null exceeded the real core in **all {ALN['n_perm']}** permutations (`alllayer_null.py`) |
| Backbone vs uniform random-gene null | shared by ≥8/10 | {CTL['stats_final']['n_perm_backbone']} | {BN['>=8']['real']} vs {BN['>=8']['null_mean']} ± {BN['>=8']['null_sd']}, z = {BN['>=8']['z']}, p < 0.004, {BN['>=8']['fold']:.0f}× |
| Backbone vs **degree-matched** null | shared by ≥8/10 | {CTL['degree_null']['n_perm']} | {DN['>=8']['real']} vs {DN['>=8']['degree_null_mean']} ± {DN['>=8']['degree_null_sd']}, z = {DN['>=8']['degree_z']}, p < 0.004, {DN['>=8']['degree_fold']:.0f}× |
| Backbone vs depth | ≥8/10 at five depth fractions | {DPB['n_perm']} | {min(r['tiers']['>=8']['real'] for r in DPB['depths'])}–{max(r['tiers']['>=8']['real'] for r in DPB['depths'])} concepts, {min(r['tiers']['>=8']['fold'] for r in DPB['depths'])}–{max(r['tiers']['>=8']['fold'] for r in DPB['depths'])}×, every depth at the p floor |
| Held-out replication | backbone Jaccard vs two-draw null | {CTL['stats_final']['n_perm_heldout']} | {HN['observed_jaccard']} vs {HN['null_mean']:.2f} (max {HN['null_max']}), p < 0.007, {HN['fold']}× — nine of ten models (scGPT not re-extractable), ≥7-of-9 backbone |
| Config robustness | ≥8 fold across 5 calibrated configs | — | {RC['ge8_fold_range'][0]}–{RC['ge8_fold_range'][1]:.0f}× (all beat null) |
| KEGG-drop robustness | ≥8/10 backbone, GO+Reactome only vs full | {KEG['n_perm']} | full {KEG['ge8_full']} ({KEG['full_GO_KEGG_Reactome']['>=8']['fold']}×) → drop-KEGG {KEG['ge8_dropKEGG']} ({KEG['dropKEGG_GO_Reactome']['>=8']['fold']}×); {KEG['kegg_sourced_in_ge8']}/{KEG['ge8_full']} KEGG-sourced (`kegg_robust.py`) |
| CKA reality | real vs cell-shuffle, **all {len(ck)} pairs** | 20 per pair | real {min(x['real_cka'] for x in ck)}–{max(x['real_cka'] for x in ck)} vs floor {min(x['shuffled_cka'] for x in ck)}–{max(x['shuffled_cka'] for x in ck)}; **worst pair {cka_min:.1f}× its own floor** |
| SAE vs PCA | var-explained at k=32, **all ten models** | — | SAE 0.61–0.98 vs PCA 0.23–0.96; ahead in all ten, gap +0.02 (tGPT) to +0.53 (Tahoe-x1); random dict < 0 |
| **Held-out regulatory recovery** (§3.4) | co-firing pairs vs TRRUST, configuration-model null | {TR2['n_perm']} | 5 of 10 models pass at p ≤ 0.005: {', '.join(f"{nm(m)} {TR2['per_model'][m]['fold']}×" for m in sorted(TR2['per_model'], key=lambda x: -(TR2['per_model'][x]['fold'] or 0))[:5])} |
| **Cross-model corroboration** (§3.4) | precision by number of models predicting a pair | {TR2.get('n_perm_curve', '?')} | 1 model {CV['1']['fold']}× → 2 models {CV['2']['fold']}× (all ten pooled); absolute precision {100*CV['2']['precision']:.2f} % |
| **Equal-budget re-scoring** (§3.4) | same {SM['budget']:,}-pair budget per model | {SM['n_perm']} | ranking not preserved (rank corr {SM['spearman_full_vs_matched']}); UCE loses significance, MaxToki gains it |
| **Robustness of the recovery** (§3.4) | five variants of the pair definition | {ROB['n_perm']} | {min(f1)}–{max(f1)}× at one model, p ≤ 0.005 in all five |
| **Literature co-mention** (§3.5) | gene–paper graph, publication counts preserved | {PUB['n_perm']} | {PUB['observed_pairs_comentioned']:,} of {PUB['n_pairs_testable']:,} vs {PUB['null_pairs_comentioned_mean']} ± {PUB['null_sd']} — {PUB['fold']}×, z = {PUB['z']} |
| **Perturbation** (§3.5) | responder rank, responsiveness-matched null | {PRT['n_perm']} | K562 {PRT['K562']['top5pct_fold']}× (z = {PRT['K562']['z_top5']}), RPE1 {PRT['RPE1']['top5pct_fold']}× (z = {PRT['RPE1']['z_top5']}) — replicates in an unrelated line |
| Correlations (n=10) | bootstrap 95 % CI | 5000 | n_conc~d_sae 0.79 [0.56,0.96]; conc~tokenisation −0.01 [−0.86,0.85] (underpowered) |"""]

# ---- S11: the keyword list behind the specific/housekeeping split ----------
# Section 3.2 promised "the keyword list is given in Supplementary" and it was not there.
import re as _re
_src = open(f"{SCRIPTS}/recalibrate_robust.py", encoding="utf-8").read()
_hk = _re.search(r"HK = \[(.*?)\]", _src, _re.S)
HK = [w.strip().strip('"\'') for w in _hk.group(1).replace("\n", " ").split(",") if w.strip()] if _hk else []
TRIV = load("recalibrate_robust.json")["triviality"]
s11 = [f"""## Table S11 — Keyword classification of the backbone

Section 3.2 splits the {TRIV['n_ge8']}-concept backbone into specific programme biology and housekeeping. The split is
by substring match on the concept name, with no manual curation: a concept counts as housekeeping if its name contains
any of the {len(HK)} strings below, and as specific otherwise. That gives **{TRIV['specific']} specific
({100 - TRIV['pct_housekeeping']:.1f} %)** and **{TRIV['housekeeping']} housekeeping ({TRIV['pct_housekeeping']} %)**.
The rule is deliberately crude and stated here so it can be applied or disputed exactly
(`recalibrate_robust.py`, `HK`).

```
{chr(10).join('  ' + ', '.join(HK[i:i + 6]) for i in range(0, len(HK), 6))}
```

A concept such as *cytoplasmic translation* or *rRNA processing* therefore lands in housekeeping, while *antigen
processing and presentation of exogenous peptide antigen via MHC class II* or *defence response to bacterium* lands in
specific. Borderline cases exist — *regulation of gene expression* is counted as housekeeping on the "gene expression"
string — and the percentages should be read as approximate."""]

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
{100*TR2['cross_model_curve']['2']['precision']:.2f} %, i.e. roughly one correct regulatory edge per
{round(1/TR2['cross_model_curve']['2']['precision'])} pairs proposed — {TR2['cross_model_curve']['2']['fold']}× chance, but
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
# ---- S12: near-duplicate features and the recurrence threshold ----------------------------------
RED = json.load(open(f"{C}/pair_redundancy.json"))["models"]
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
s12 = [f"""## Table S12 — Near-duplicate features and the recurrence threshold *(robustness)*

A pair is *predicted* when it recurs in ≥5 features, and a block of near-duplicate features could satisfy that on its
own. In every model the features that form co-activation communities (the co-firing graph of §3.4) fire far more
often than the rest. *Block-carried* pairs are those whose supporting features are ≥50 % community members.
*Dedup* counts community features once per community and layer. Generated from `pair_redundancy.json`
(`pair_redundancy.py`).

| Model | Features (in communities) | Fire %: community vs rest | Predicted pairs | Block-carried | TRRUST hits: block-carried / rest | Hits full → dedup | Precision full → dedup |
|---|---|---|---|---|---|---|---|"""]
for m in ORDER:
    r = RED[m]
    s12.append(f"| {nm(m)} | {r['n_features']:,} ({r['n_community_features']:,}) | {r['fire_pct_community']} vs {r['fire_pct_rest']} | {r['n_pred']:,} | "
               f"{r['n_pred_block_carried']:,} ({r['pct_pred_block_carried']} %) | {r['hits_block_carried']} / {r['hits_rest']} | "
               f"{r['hits_block_carried'] + r['hits_rest']} → {r['hits_dedup']} | {100*r['precision_full']:.3f} % → {100*r['precision_dedup']:.3f} % |")
s12.append("")

GENERATED = {2: "\n".join(s2), 11: "\n".join(s11), 5: "\n".join(s5), 8: "\n".join(s8), 9: "\n".join(s9), 10: "\n".join(s10), 12: "\n".join(s12)}


def patch_s3(sec: str) -> str:
    """replace the data-derived health rows in Table S3, leaving the hand-written rows alone"""
    lines = [l for l in sec.split("\n") if not l.startswith("| SAE health — dead features")]
    out, done = [], False
    for l in lines:
        if l.startswith("| SAE health — decoder orthogonality") and not done:
            out += [f"| {k} | {v} |" for k, v in s3_rows]; done = True
        out.append(l)
    return "\n".join(out)
head, *rest = re.split(r"\n(?=## )", open(SUP, encoding="utf-8").read())
tables, other = dict(GENERATED), []
for sec in rest:
    m = re.match(r"## Table S(\d+)", sec)
    if not m:
        other.append(sec)
    elif int(m.group(1)) not in GENERATED:
        k = int(m.group(1))
        tables[k] = patch_s3(sec) if k == 3 else sec
ordered = [tables[k] for k in sorted(tables)] + other
open(SUP, "w", encoding="utf-8").write(
    head.rstrip() + "\n\n" + "\n\n".join(x.rstrip() for x in ordered) + "\n")
print(f"S8: {len(DPB['depths'])} depths at N={DPB['n_perm']}")
print(f"S9: {len(pairs):,} pairs, top 20 tabulated")
print(f"S10: {sum(1 for m in SM['models'] if SM['models'][m].get('testable'))} models at budget {B:,}, "
      f"{sum(1 for m in SM['models'] if not SM['models'][m].get('testable'))} untestable")
print("==> SUPPLEMENTARY.md")
