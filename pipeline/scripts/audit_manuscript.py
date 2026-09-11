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
import json, re, sys

C = "/Users/annaantipova/Desktop/biomech/outputs/atlas/comparative"
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
    got = float(m.group(1).replace(",", "").replace("−", "-"))
    dec = len(m.group(1).split(".")[1]) if "." in m.group(1) else 0
    tol = 0.5 * 10 ** (-dec) + 1e-9                      # the text's own rounding precision
    ok = abs(got - float(expected)) <= tol
    rows.append((claim, source, f"{expected} vs {got}", "ok" if ok else "MISMATCH", ok))


N = r"([-−\d][\d,]*\.?\d*)"

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
cv = dig(FIN, "cross_model_curve", default={})
check("3.4 k>=1 fold", rf"enriched {N}-fold over the degree-matched null", dig(cv, "1", "fold"), "hypothesis_final")
check("3.4 k>=1 edges", rf"degree-matched null \({N} recovered edges\)", dig(cv, "1", "hits"), "hypothesis_final")
check("3.4 k>=2 fold", rf"enriched \*\*{N}-fold\*\*", dig(cv, "2", "fold"), "hypothesis_final")
check("3.4 k>=2 edges", rf"\*\*[\d.]+-fold\*\* \({N} edges\)", dig(cv, "2", "hits"), "hypothesis_final")
check("3.4 candidate links", rf"leaves {N} candidate links", dig(FIN, "n_hypotheses"), "hypothesis_final")

# ---- 3.4 robustness --------------------------------------------------------
var = dig(ROB, "variants", default={})
f1 = [dig(v, "1", "fold") for v in var.values() if dig(v, "1", "fold")]
check("3.4 robustness min", rf"variant we ran \({N}–", min(f1) if f1 else None, "hypothesis_robust")
check("3.4 robustness max", rf"variant we ran \([\d.]+–{N}×", max(f1) if f1 else None, "hypothesis_robust")
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
sys.exit(1 if any(x[3] == "MISMATCH" for x in rows) else 0)
