#!/usr/bin/env python
"""Third audit module: the numbers the manuscript *derives* rather than reads off a result file.

The other two modules check numbers that appear verbatim in a JSON. A coverage measurement over the
Results and Discussion found 33 that do not: minima and medians over a list, ratios of two stored
figures, a correlation across models, a reciprocal. Nothing verified those, so a stale one would have
survived every re-run -- which is exactly how the 28.4x selection artefact lasted as long as it did.

Each check here recomputes the quantity from the stored primitives instead of comparing it to a
second stored copy of itself. That is deliberate: a copy can go stale in step with the text, a
recomputation cannot. Two numbers had no primitives to recompute from at all and are now written by
their own scripts (hypothesis_pubmed's uncapped co-mention rate; see also study_bias_coverage.py).
    python scripts/audit_manuscript_derived.py
"""
from __future__ import annotations

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import json, re, sys
import numpy as np

C = f"{_B}/outputs/atlas/comparative"
FLAT = re.sub(r"\s+", " ", open(f"{C}/MANUSCRIPT.md", encoding="utf-8").read())


def load(n):
    try:
        return json.load(open(f"{C}/{n}"))
    except Exception as e:
        return {"__error__": str(e)}


T2, SM = load("hypothesis_trrust2.json"), load("hypothesis_sizematched.json")
CKA, RR = load("cka_svd_null.json"), load("recalibrate_robust.json")
PUB, PT = load("hypothesis_pubmed.json"), load("hypothesis_perturb.json")
SBC, FIN = load("study_bias_coverage.json"), load("hypothesis_final.json")
rows = []

WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
         "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def check(claim, pattern, expected, source, tol_rel=None):
    """Compare one number in the text against a value recomputed from source data."""
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
    tol = 0.5 * 10 ** (-dec) + 1e-9          # the precision the text itself shows
    if tol_rel:
        tol = max(tol, tol_rel * abs(float(expected)))
    ok = abs(got - float(expected)) <= tol
    rows.append((claim, source, f"{round(float(expected), 4)} vs {got}", "ok" if ok else "MISMATCH", ok))


N = r"([-−]?[\d,]+\.?\d*)"
W = r"([A-Za-z]+)"

# ---- 3.4 graph sizes, and the correlation that explains the ranking ---------
PM = T2.get("per_model", {})
check("graph C2S", rf"AIDO\.Cell against {N} for C2S-Scale", PM.get("C2S", {}).get("n_pairs"), "trrust2:C2S.n_pairs")
check("graph AIDO", rf"four orders of magnitude, {N} pairs for AIDO\.Cell", PM.get("AIDO", {}).get("n_pairs"),
      "trrust2:AIDO.n_pairs")
check("graph UCE", rf"rests on having {N} pairs", PM.get("UCE", {}).get("n_pairs"), "trrust2:UCE.n_pairs")
check("graph scFoundation", rf"scFoundation {N}, both below the budget",
      PM.get("scFoundation", {}).get("n_pairs"), "trrust2:scFoundation.n_pairs")

# The claim is that recovery tracks graph size; recompute the correlation over all ten models.
if PM:
    sz = np.array([PM[m]["n_pairs"] for m in sorted(PM)], float)
    hi = np.array([PM[m]["hits"] for m in sorted(PM)], float)
    r_size_hits = float(np.corrcoef(sz, hi)[0, 1])
else:
    r_size_hits = None
check("r(graph size, hits)", rf"track graph size at r = {N}", r_size_hits, "trrust2:per_model (recomputed)")

# ---- 3.3 CKA: a span, a median and a minimum over 45 pairs ------------------
CN = CKA.get("cka_null") if isinstance(CKA.get("cka_null"), list) else None
if CN:
    real = np.array([x["real_cka"] for x in CN], float)
    shuf = np.array([x["shuffled_cka"] for x in CN], float)
    ratio = real / shuf
else:
    real = shuf = ratio = None
check("CKA n pairs", rf"CKA with a cell-shuffle\s*\n?null for all {N} model pairs",
      len(CN) if CN else None, "cka_svd_null (recomputed)")
check("CKA min", rf"Real CKA spans {N}[-–—]", None if real is None else real.min(), "cka_svd_null (recomputed)")
check("CKA max", rf"Real CKA spans [\d.]+[-–—]{N} \(median", None if real is None else real.max(), "cka_svd_null (recomputed)")
check("CKA median", rf"\(median {N}\) against a shuffle floor", None if real is None else float(np.median(real)),
      "cka_svd_null (recomputed)")
check("CKA floor min", rf"shuffle floor of {N}[-–—]", None if shuf is None else shuf.min(), "cka_svd_null (recomputed)")
check("CKA floor max", rf"shuffle floor of [\d.]+[-–—]{N}", None if shuf is None else shuf.max(), "cka_svd_null (recomputed)")
check("CKA min ratio", rf"\(minimum ratio {N};", None if ratio is None else ratio.min(), "cka_svd_null (recomputed)")
# The Fig 4 caption states a bound, not a value; it read ">10-fold" for weeks while the text said fortyfold.
_m = re.search(r"every pair exceeds its floor >(\d+)-fold", FLAT)
_ok = bool(_m and ratio is not None and int(_m.group(1)) <= ratio.min() < int(_m.group(1)) + 10)
rows.append(("Fig4 caption floor bound", "cka_svd_null (bound)", f"min {ratio.min():.1f} vs >{_m.group(1) if _m else '?'}",
             "ok" if _ok else "MISMATCH", _ok))

# ---- 3.2 how much of the backbone is ordinary housekeeping ------------------
TRV = RR.get("triviality", {})
n8, hk = TRV.get("n_ge8"), TRV.get("housekeeping")
check("backbone specific %", rf"About {N} % of the backbone is specific",
      None if n8 in (None, 0) else 100 * TRV["specific"] / n8, "recalibrate_robust:triviality (recomputed)")
check("backbone housekeeping %", rf"and ~{N} % is housekeeping", TRV.get("pct_housekeeping"), "recalibrate_robust:triviality")
check("backbone n concepts", rf"rather than the ≥8-of-10 set of {N}", n8, "recalibrate_robust:triviality.n_ge8")

# ---- 3.5 literature co-mention: why the mega-paper cap is there -------------
check("mega-paper cap", rf"exclude papers annotating more than {N} genes", PUB.get("mega_paper_cap"), "pubmed:mega_paper_cap")
check("uncapped co-mention %", rf"without that cap {N} % of all pairs are nominally",
      PUB.get("observed_pct_comentioned_uncapped"), "pubmed:observed_pct_comentioned_uncapped")

# ---- 3.5 corroboration: the gain is a ratio of two stored enrichments -------
# The all-ten pooling is trrust2's cross_model_curve. hypothesis_final.json holds the *five-model*
# version (5.3x / 28.4x), which the manuscript reports only as a direction check -- reading the gain
# off that file would silently restate the selected numbers as the headline.
CMC = T2.get("cross_model_curve", {})
one, two = CMC.get("1", {}).get("fold"), CMC.get("2", {}).get("fold")
check("corroboration gain", rf"a {N}× gain in precision from corroboration",
      None if not (one and two) else two / one, "trrust2:cross_model_curve (ge2/ge1)")
check("single-model fold", rf"are enriched {N}-fold over the degree-matched null", one, "trrust2:curve.1.fold")
check("two-model fold", rf"are\s*\n?enriched {N}-fold \(31 edges\)", two, "trrust2:curve.2.fold")
prec = CMC.get("2", {}).get("precision")
check("two-model precision %", rf"is a precision of {N} %", None if prec is None else 100 * prec,
      "trrust2:curve.2.precision")
# "about one in every 480" is 481.6 written to two significant figures, so this one check compares
# at the precision the word "about" claims rather than at the last digit shown.
n2, h2 = CMC.get("2", {}).get("n_pairs"), CMC.get("2", {}).get("hits")
check("pairs per correct edge", rf"about one correct edge in every {N} pairs proposed",
      None if not (n2 and h2) else n2 / h2, "trrust2:curve.2 (recomputed n/hits)", tol_rel=0.02)

# ---- 3.5 Perturb-seq, read per recurrence level rather than pooled ----------
def by(line, k, field):
    return PT.get(line, {}).get("by_n_models", {}).get(k, {}).get(field)


check("RPE1 >=3 fold", rf"it does in RPE1 \(1\.28× → {N}× at ≥3 models", by("RPE1", "3", "fold"), "perturb:RPE1.3.fold")
check("RPE1 >=3 z", rf"at ≥3 models, z = {N}", by("RPE1", "3", "z"), "perturb:RPE1.3.z")
check("K562 >=2 fold", rf"but not in K562 \({N}× at ≥2 and at ≥3 models", by("K562", "2", "fold"), "perturb:K562.2.fold")
check("K562 >=4 fold", rf"at ≥3 models; {N}× at ≥4 models", by("K562", "4", "fold"), "perturb:K562.4.fold")
check("K562 >=4 n", rf"at ≥4 models on {N} pairs", by("K562", "4", "n"), "perturb:K562.4.n")
check("K562 >=4 p", rf"at ≥4 models on [\d,]+ pairs, p = {N}", by("K562", "4", "p_emp"), "perturb:K562.4.p_emp")

# ---- 3.4 equal-budget re-scoring -------------------------------------------
MU = SM.get("models", {}).get("UCE", {})
check("UCE random-draw fold", rf"\({N} ± [\d.]+× against 3\.9×\)",
      MU.get("randN", {}).get("fold_mean"), "sizematched:UCE.randN.fold_mean")
check("UCE random-draw sd", rf"\([\d.]+ ± {N}× against 3\.9×\)", MU.get("randN", {}).get("fold_sd"), "sizematched:UCE.randN.fold_sd")

# ---- Discussion: the understudied genes the relaxed filter puts in play -----
check("understudied kept", rf"understudied genes in play from {N} to",
      SBC.get("retention", {}).get("understudied_kept"), "study_bias_coverage:retention")

# ---- report ----------------------------------------------------------------
bad = [x for x in rows if not x[4]]
w = max(len(x[0]) for x in rows)
print(f"{'claim':<{w}}  {'source':<38}  {'source vs text':>22}  status")
print("-" * (w + 70))
for claim, src, val, status, ok in rows:
    print(f"{claim:<{w}}  {src:<38}  {val:>22}  {status if ok else '*** ' + status + ' ***'}")
print(f"\n{len(rows)-len(bad)}/{len(rows)} verified against source")
if bad:
    print("\nneeds a look:")
    for claim, src, val, status, _ in bad:
        print(f"   {claim:<24} {status:<18} {val}   [{src}]")
# A check that could not find its sentence did not run at all, which is worse than a mismatch.
sys.exit(1 if bad else 0)
