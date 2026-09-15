#!/usr/bin/env python
"""Check every headline number in MANUSCRIPT.md against the file that produced it.

Written after three separate bugs in one session, each of which looked like a result rather than a
mistake (a null taken from the wrong population, unmapped gene symbols folded into "zero
publications", and an Ensembl filter applied after truncation instead of before). Reasoning about
whether a number is right is what failed each time; this re-reads it from source instead.

Matching is numeric, not textual: the manuscript rounds (13.29 -> "13.3x"), so each check pulls the
number out of the sentence it lives in and compares it to the source value within the tolerance
implied by how many decimals the text shows. A check fails only when the values genuinely differ.
    python scripts/audit_manuscript.py
"""
from __future__ import annotations

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import json, re, sys

C = f"{_B}/outputs/atlas/comparative"
MD = open(f"{C}/MANUSCRIPT.md", encoding="utf-8").read()
FLAT = re.sub(r"\s+", " ", MD)


def load(n):
    try:
        return json.load(open(f"{C}/{n}"))
    except Exception as e:
        return {"__error__": str(e)}


TR2, FIN, ROB = load("hypothesis_trrust2.json"), load("hypothesis_final.json"), load("hypothesis_robust.json")
PUB, PRT, SBI = load("hypothesis_pubmed.json"), load("hypothesis_perturb.json"), load("hypothesis_studybias.json")

rows = []


def dig(d, *path, default=None):
    for p in path:
        if isinstance(d, dict) and p in d:
            d = d[p]
        else:
            return default
    return d


def check(claim, pattern, expected, source):
    """pattern: a regex over the manuscript with ONE capture group holding the number."""
    if expected is None:
        rows.append((claim, source, "-", "source missing", False)); return
    m = re.search(pattern, FLAT)
    if not m:
        rows.append((claim, source, str(expected), "no such sentence", False)); return
    txt = m.group(1).replace(",", "").replace("−", "-")
    got = float(WORDS[txt.lower()]) if txt.lower() in WORDS else float(txt)
    if txt.lower() in WORDS:
        txt = str(int(got))
    dec = len(txt.split(".")[1]) if "." in txt else 0
    tol = 0.5 * 10 ** (-dec) + 1e-9                      # the text's own rounding precision
    ok = abs(got - float(expected)) <= tol
    rows.append((claim, source, f"{expected} vs {got}", "ok" if ok else "MISMATCH", ok))


N = r"([-−\d][\d,]*\.?\d*)"
W = r"([A-Za-z]+)"             # a count the text spells out ("two of the five models")
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

# ---- 3.4 per-model TRRUST --------------------------------------------------
pm = dig(TR2, "per_model", default={})
for key, disp in (("Tahoe", "Tahoe-x1"), ("scGPT", "scGPT"), ("UCE", "UCE"),
                  ("C2S", "C2S-Scale"), ("Geneformer", "Geneformer-V2")):
    d = pm.get(key, {})
    check(f"3.4 {disp} fold", rf"{re.escape(disp)} {N}×", d.get("fold"), "hypothesis_trrust2:per_model")
    check(f"3.4 {disp} edges", rf"{re.escape(disp)} [\d.]+× \({N}", d.get("hits"), "hypothesis_trrust2:per_model")
check("3.4 MaxToki fold", rf"MaxToki reaches {N}×", dig(pm, "MaxToki", "fold"), "hypothesis_trrust2")
check("3.4 MaxToki p", rf"MaxToki reaches [\d.]+× but p = {N}", dig(pm, "MaxToki", "p_emp"), "hypothesis_trrust2")
check("3.4 tGPT fold", rf"tGPT {N}×", dig(pm, "tGPT", "fold"), "hypothesis_trrust2")

# ---- 3.4 cross-model curve -------------------------------------------------
cv = dig(TR2, "cross_model_curve", default={})   # all ten models: the unselected headline
check("3.4 k>=1 fold", rf"enriched {N}-fold over the degree-matched null", dig(cv, "1", "fold"), "hypothesis_trrust2 (all 10)")
check("3.4 k>=1 edges", rf"degree-matched null \({N} recovered edges\)", dig(cv, "1", "hits"), "hypothesis_trrust2 (all 10)")
check("3.4 k>=2 fold", rf"enriched \*\*{N}-fold\*\*", dig(cv, "2", "fold"), "hypothesis_trrust2 (all 10)")
check("3.4 k>=2 edges", rf"\*\*[\d.]+-fold\*\* \({N} edges\)", dig(cv, "2", "hits"), "hypothesis_trrust2 (all 10)")
FINC = dig(FIN, "cross_model_curve", default={})
check("3.4 selected-5 k>=1", rf"raises the same figures to {N}-fold", dig(FINC, "1", "fold"), "hypothesis_final (selected 5)")
check("3.4 selected-5 k>=2", rf"to [\d.]+-fold and {N}-fold", dig(FINC, "2", "fold"), "hypothesis_final (selected 5)")
check("3.4 absolute precision", rf"a\s*\n?precision of \*\*{N} %", 100*dig(cv, "2", "precision", default=0), "hypothesis_trrust2 (all 10)")
SM = load("hypothesis_sizematched.json")
for mm, pat in (("Tahoe",      rf"on this evidence \({N}× and"),
                ("scGPT",      rf"on this evidence \([\d.]+× and {N}× at equal budget"),
                ("C2S",        rf"also\s*hold \({N}× and"),
                ("Geneformer", rf"also\s*hold \([\d.]+× and {N}×"),
                ("UCE",        rf"falls to {N}× \(p = 0\.12"),
                ("MaxToki",    rf"at matched budget \({N}×, p = 0\.04")):
    check(f"3.4 matched {mm}", pat, dig(SM, "models", mm, "topN", "fold"), "hypothesis_sizematched")
check("3.4 matched budget", rf"the {N} highest-weight pairs", dig(SM, "budget"), "hypothesis_sizematched")
check("3.4 rank correlation", rf"rank correlation {N}\)", dig(SM, "spearman_full_vs_matched"), "hypothesis_sizematched")
check("3.4 candidate links", rf"leaves {N} candidate links", dig(FIN, "n_hypotheses"), "hypothesis_final")

# ---- 3.4 robustness --------------------------------------------------------
var = dig(ROB, "variants", default={})
f1 = [dig(v, "1", "fold") for v in var.values() if dig(v, "1", "fold")]
check("3.4 robustness min", rf"five variants give\s*{N}–", min(f1) if f1 else None, "hypothesis_robust")
check("3.4 robustness max", rf"five variants give\s*[\d.]+–{N}×", max(f1) if f1 else None, "hypothesis_robust")
check("3.4 firing-only before", rf"\*raises\* the enrichment \({N}× →", dig(var, "R0 baseline (W>=5)", "1", "fold"),
      "hypothesis_robust:R0")
check("3.4 firing-only after", rf"enrichment \([\d.]+× → {N}×\)", dig(var, "R2 firing features only", "1", "fold"),
      "hypothesis_robust:R2")

# ---- 3.5 literature --------------------------------------------------------
check("3.5 co-mentioned", rf"{N} of the 4,111", dig(PUB, "observed_pairs_comentioned"), "hypothesis_pubmed")
check("3.5 testable pairs", rf"of the {N} mappable pairs", dig(PUB, "n_pairs_testable"), "hypothesis_pubmed")
check("3.5 lit null mean", rf"against {N} ± ", dig(PUB, "null_pairs_comentioned_mean"), "hypothesis_pubmed")
check("3.5 lit null sd", rf"against [\d.]+ ± {N}", dig(PUB, "null_sd"), "hypothesis_pubmed")
check("3.5 lit fold", rf"publication count exactly \({N}×", dig(PUB, "fold"), "hypothesis_pubmed")
check("3.5 lit z", rf"publication count exactly \([\d.]+×, z = {N}", dig(PUB, "z"), "hypothesis_pubmed")
check("3.5 never co-mentioned", rf"Only {N} of the", dig(PUB, "n_never_comentioned"), "hypothesis_pubmed")

# ---- 3.5 perturbation ------------------------------------------------------
for line in ("K562", "RPE1"):
    d = dig(PRT, line, default={})
    pre = r"In K562, [\d,]+ ordered pairs are testable and B lands in the top 5 % of moved genes for " \
        if line == "K562" else r"unrelated cell line: "
    check(f"3.5 {line} rate", pre + rf"{N} %", round(100 * d.get("top5pct_rate", 0), 2), f"hypothesis_perturb:{line}")
    check(f"3.5 {line} null", pre + rf"[\d.]+ % (?:against|of them against) {N} %",
          round(100 * d.get("null_top5pct_rate", 0), 2), f"hypothesis_perturb:{line}")
    check(f"3.5 {line} fold", pre + rf"[\d.]+ %[^()]*\({N}×", d.get("top5pct_fold"), f"hypothesis_perturb:{line}")
    check(f"3.5 {line} z", pre + rf"[\d.]+ %[^()]*\([\d.]+×, z = {N}", d.get("z_top5"), f"hypothesis_perturb:{line}")
check("3.5 K562 n pairs", rf"In K562, {N} ordered pairs", dig(PRT, "K562", "n_ordered_pairs"), "hypothesis_perturb")

# ---- Discussion / Limitations ---------------------------------------------
k = dig(SBI, "K562", "by_papers", default={}); r = dig(SBI, "RPE1", "by_papers", default={})
kf = [v["fold"] for lab, v in k.items() if lab != "unmapped*"]
rf_ = [v["fold"] for lab, v in r.items() if lab != "unmapped*"]
check("Disc K562 strata min", rf"enrichments of {N}–", min(kf) if kf else None, "hypothesis_studybias:K562")
check("Disc K562 strata max", rf"enrichments of [\d.]+–{N}× in K562", max(kf) if kf else None, "hypothesis_studybias:K562")
check("Disc RPE1 strata min", rf"in K562 and {N}–", min(rf_) if rf_ else None, "hypothesis_studybias:RPE1")
check("Disc RPE1 strata max", rf"in K562 and [\d.]+–{N}× in RPE1", max(rf_) if rf_ else None, "hypothesis_studybias:RPE1")
check("Disc understudied at W>=2", rf"from 141 to {N} while", dig(SBI, "n_understudied_genes"), "hypothesis_studybias")
check("Disc W>=2 K562 fold", rf"falls from [\d.]+× to {N}× \(K562\)", dig(SBI, "K562", "overall", "fold"),
      "hypothesis_studybias:K562")
check("Disc W>=2 RPE1 fold", rf"and [\d.]+× to {N}× \(RPE1\)", dig(SBI, "RPE1", "overall", "fold"),
      "hypothesis_studybias:RPE1")
check("Limit lncRNA K562 fold", rf"stratum \({N}×, z = ", dig(k, "unmapped*", "fold"), "hypothesis_studybias:K562")
check("Limit lncRNA K562 z", rf"stratum \([\d.]+×, z = {N}", dig(k, "unmapped*", "z"), "hypothesis_studybias:K562")
check("Limit lncRNA RPE1", rf"replicate in RPE1 \({N}×", dig(r, "unmapped*", "fold"), "hypothesis_studybias:RPE1")


# ---- Discussion: the blind-spot paragraph, previously computed ad hoc --------
SBC = load("study_bias_coverage.json")
R, SB = dig(SBC, "retention", default={}), dig(SBC, "screen_bias", default={})
check("Disc featured understudied", rf"place {N} genes with fewer than five", dig(SBC, "understudied_featured"), "study_bias_coverage")
check("Disc protein-coding of them", rf"genes — {N} of them protein-coding", dig(SBC, "understudied_protein_coding"), "study_bias_coverage")
check("Disc filter ratio", rf"confidence filter is {N}\s*\n?times harsher", R.get("ratio"), "study_bias_coverage:retention")
check("Disc understudied kept %", rf"\({N} % of understudied genes survive", R.get("understudied_pct"), "study_bias_coverage:retention")
check("Disc well-studied kept %", rf"against {N} % of genes with ≥50 papers", R.get("well_studied_pct"), "study_bias_coverage:retention")
check("Disc absent from screen", rf"evidence runs out: {N} of the 793", dig(SBC, "protein_coding_understudied_absent_from_screen"), "study_bias_coverage")
check("Disc screen % understudied", rf"perturbed set contains {N} %", SB.get("screen_pct_understudied"), "study_bias_coverage:screen_bias")
check("Disc all-PC % understudied", rf"understudied genes against {N} % of all protein-coding", SB.get("all_protein_coding_pct_understudied"), "study_bias_coverage:screen_bias")
check("Limit non-coding share", rf"{N} of the 4,003\s*\n?understudied featured genes are non-coding", dig(SBC, "understudied_noncoding_or_pseudo"), "study_bias_coverage")

# ---- permutation counts: the text said 200 for tests that ran 500 and 100 -----------------
# Each test's N is read from its own result file, so a script default that changes shows up
# here rather than in a reviewer's recomputation of the p-value floor.
check("Meth N per-model TRRUST", rf"N = {N} for the per-model held-out regulatory recovery", TR2.get("n_perm"), "trrust2:n_perm")
check("Meth N corroboration curve", rf"N = {N} for the cross-model corroboration curve", TR2.get("n_perm_curve"), "trrust2:n_perm_curve")
check("Meth N equal budget", rf"N = {N} for the cross-model corroboration curve \(each draw rewires all ten graphs\) and for the equal-budget",
      load("hypothesis_sizematched.json").get("n_perm"), "sizematched:n_perm")
check("Meth N robustness", rf"N = {N} for the robustness variants", ROB.get("n_perm"), "robust:n_perm")
check("Meth N literature", rf"N = {N} for the robustness variants and for the literature", PUB.get("n_perm"), "pubmed:n_perm")
check("Meth N perturbation", rf"N = {N} for the robustness variants and for the literature, perturbation", PRT.get("n_perm"), "perturb:n_perm")
check("Meth N strata", rf"N = {N} for the robustness variants and for the literature, perturbation and\s*publication-stratum", SBI.get("n_perm"), "studybias:n_perm")
check("Meth N held-out rewirings", rf"empirical p over N = {N} rewirings per model", TR2.get("n_perm"), "trrust2:n_perm")
check("Meth N curve rewirings", rf"over N = {N} rewirings for precision as a function", TR2.get("n_perm_curve"), "trrust2:n_perm_curve")
check("3.4 rewirings per model", rf"significant at p ≤ 0\.005 \({N} rewirings\)", TR2.get("n_perm"), "trrust2:n_perm")
check("3.4 curve no-edge rewirings", rf"no edge at all in {N}/[\d]+\s*rewirings", TR2.get("n_perm_curve"), "trrust2:n_perm_curve")
check("Fig5A rewirings", rf"p ≤ 0\.05 over {N} rewirings", TR2.get("n_perm"), "trrust2:n_perm")
check("Fig5B rewirings", rf"no edge in any of the {N} rewirings", TR2.get("n_perm_curve"), "trrust2:n_perm_curve")

# ---- Discussion: the single-layer check, previously a /tmp script with no saved output --------
SL = load("hypothesis_singlelayer.json")
W2, W5 = dig(SL, "variants", "W2", default={}), dig(SL, "variants", "W5", default={})
check("Disc single-layer fold", rf"the pooled enrichment is {N}× \(p = ", dig(W2, "cross_model_curve", "1", "fold"), "singlelayer:W2.curve.1")
check("Disc single-layer p", rf"the pooled enrichment is [\d.]+× \(p = {N};", dig(W2, "cross_model_curve", "1", "p_emp"), "singlelayer:W2.curve.1")
check("Disc single-layer hits", rf"\(p = [\d.]+; {N} recovered edges\)", dig(W2, "cross_model_curve", "1", "hits"), "singlelayer:W2.curve.1")
check("Disc pooled five-model fold", rf"against {N}× when layers are\s*pooled on the same five", dig(FIN, "cross_model_curve", "1", "fold"), "hypothesis_final:curve.1")
check("Disc single-layer W5 hits", rf"threshold of ≥5 features only {N} edges are recovered in total", dig(W5, "cross_model_curve", "1", "hits"), "singlelayer:W5.curve.1")
check("Disc single-layer hits min", rf"one layer yields only {N}–", W2.get("hits_min"), "singlelayer:W2")
check("Disc single-layer hits max", rf"one layer yields only \d+–{N} recovered edges", W2.get("hits_max"), "singlelayer:W2")
check("Disc single-layer n significant", rf"only {W} of the five models are individually significant", W2.get("n_models_significant"), "singlelayer:W2")

# ---- Limitations: the non-coding stratum that does not replicate ----------------------------
UK, UR = dig(SBI, "K562", "by_papers", "unmapped*", default={}), dig(SBI, "RPE1", "by_papers", "unmapped*", default={})
check("Lim K562 unmapped fold", rf"strongest causal enrichment of any stratum \({N}×, z = ", UK.get("fold"), "studybias:K562.unmapped*")
check("Lim K562 unmapped z", rf"strongest causal enrichment of any stratum \([\d.]+×, z = {N}\)", UK.get("z"), "studybias:K562.unmapped*")
check("Lim RPE1 unmapped fold", rf"replicate in RPE1 \({N}×, not significant\)", UR.get("fold"), "studybias:RPE1.unmapped*")
_ratio = (max(UK.get("null", 0), UR.get("null", 0)) / min(UK.get("null", 1), UR.get("null", 1))) if UK and UR else None
_ok = bool(_ratio and _ratio > 2 and re.search(r"null rate for that stratum differs more than twofold", FLAT))
rows.append(("Lim unmapped null ratio >2", "studybias:null rates", f"{_ratio:.2f}" if _ratio else "-", "ok" if _ok else "MISMATCH", _ok))

# ---- report ----------------------------------------------------------------
bad = [x for x in rows if not x[4]]
w = max(len(x[0]) for x in rows)
print(f"{'claim':<{w}}  {'source':<32}  {'source vs text':>22}  status")
print("-" * (w + 62))
for claim, src, val, status, ok in rows:
    print(f"{claim:<{w}}  {src:<32}  {val:>22}  {status if ok else '*** ' + status + ' ***'}")
print(f"\n{len(rows)-len(bad)}/{len(rows)} numbers verified against source")
if bad:
    print("\nneeds a look:")
    for claim, src, val, status, _ in bad:
        print(f"   {claim:<28} {status:<18} {val}   [{src}]")
# A check that could not find its sentence did not run at all, which is worse than a mismatch:
# rewording a sentence would otherwise switch its check off in silence. Any failure exits 1.
sys.exit(1 if bad else 0)
