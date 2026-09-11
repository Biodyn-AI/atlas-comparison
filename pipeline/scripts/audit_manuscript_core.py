#!/usr/bin/env python
"""Second half of the manuscript audit: sections 3.1-3.3, whose numbers predate today's work and
come from controls.json, alllayer_null.json, depth_backbone.json and kegg_robust.json.

Same design as audit_manuscript.py -- pull the number out of the sentence it lives in and compare
numerically within the precision the text itself shows, so rounding is not reported as an error.
Ranges in the manuscript are typeset with an en dash, hence D below.
    python scripts/audit_manuscript_core.py
"""
from __future__ import annotations
import json, re, statistics, sys

C = "/Users/annaantipova/Desktop/biomech/outputs/atlas/comparative"
FLAT = re.sub(r"\s+", " ", open(f"{C}/MANUSCRIPT.md", encoding="utf-8").read())


def load(n):
    try:
        return json.load(open(f"{C}/{n}"))
    except Exception as e:
        return {"__error__": str(e)}


CTL, ALN = load("controls.json"), load("alllayer_null.json")
DPB, KEG = load("depth_backbone.json"), load("kegg_robust.json")
rows = []


def dig(d, *path, default=None):
    for p in path:
        if isinstance(d, dict) and p in d:
            d = d[p]
        elif isinstance(d, list) and isinstance(p, int) and -len(d) <= p < len(d):
            d = d[p]
        else:
            return default
    return d


def check(claim, pattern, expected, source):
    if expected is None:
        rows.append((claim, source, "-", "source missing", False)); return
    m = re.search(pattern, FLAT)
    if not m:
        rows.append((claim, source, str(expected), "no such sentence", False)); return
    txt = m.group(1).replace(",", "").replace("−", "-")
    got = float(txt)
    dec = len(txt.split(".")[1]) if "." in txt else 0
    tol = 0.5 * 10 ** (-dec) + 1e-9
    ok = abs(got - float(expected)) <= tol
    rows.append((claim, source, f"{expected} vs {got}", "ok" if ok else "MISMATCH", ok))


N = r"([-−]?[\d,]+\.?\d*)"     # a number as the manuscript writes it
D = r"[-–—]"                   # the manuscript typesets ranges with an en dash
F = r"[\d.]+"                  # the other end of a range, unused

# ---- 3.1 the retracted naive core -----------------------------------------
check("3.1 all-layer core", rf"annotator reports {N} concepts shared by all ten",
      dig(CTL, "null", "core_observed"), "controls:null")
check("3.1 mid real core", rf"\(real {N} vs null", dig(CTL, "random_null", "real_core"), "controls:random_null")
check("3.1 mid null core", rf"\(real \d+ vs null {N}", dig(CTL, "random_null", "rand_core_mean"), "controls:random_null")
check("3.1 mid z", rf"vs null \d+, z = {N}\)", dig(CTL, "random_null", "core_z"), "controls:random_null")
check("3.1 alllayer null mean", rf"random-gene core of {N} ±", dig(ALN, "null_core_mean"), "alllayer_null")
check("3.1 alllayer null sd", rf"core of [\d,.]+ ± {N} against", dig(ALN, "null_core_sd"), "alllayer_null")
check("3.1 alllayer z", rf"against the real 2,267 \(z = {N}", dig(ALN, "core_z"), "alllayer_null")
check("3.1 alllayer fold", rf"i\.e\. {N}× the null", dig(ALN, "core_fold_over_null"), "alllayer_null")
check("3.1 alllayer perms", rf"in all {N} permutations the random core", dig(ALN, "n_perm"), "alllayer_null")
check("3.1 heldout jaccard", rf"its Jaccard of {N} matches",
      dig(CTL, "random_null", "heldout_observed_core_jaccard"), "controls:random_null")
check("3.1 two-draw jaccard", rf"matches the {N} obtained from two random draws",
      dig(CTL, "random_null", "two_random_draw_core_jaccard"), "controls:random_null")

# ---- 3.2 the calibrated backbone ------------------------------------------
BN = dig(CTL, "stats_final", "backbone_null", default={})
DN = dig(CTL, "degree_null", "tiers", default={})
HN = dig(CTL, "stats_final", "heldout_null", default={})
RC = dig(CTL, "recalibration", default={})
check("3.2 all-ten core", rf"all-ten core is near-empty \({N} concepts\)", dig(BN, ">=10", "real"), "controls:backbone_null")
check("3.2 backbone size", rf"backbone of ~{N} concepts is shared", dig(BN, ">=8", "real"), "controls:backbone_null")
check("3.2 config real lo", rf"\({N}{D}\d+ across five calibrated", dig(RC, "ge8_real_range", 0), "controls:recalibration")
check("3.2 config real hi", rf"\(\d+{D}{N} across five calibrated", dig(RC, "ge8_real_range", 1), "controls:recalibration")
check("3.2 uniform fold", rf"uniform random-gene null {N}-fold", dig(BN, ">=8", "fold"), "controls:backbone_null")
check("3.2 uniform null val", rf"null 22-fold \(78 vs {N} ±", dig(BN, ">=8", "null_mean"), "controls:backbone_null")
check("3.2 uniform null sd", rf"\(78 vs [\d.]+ ± {N}, z = 36", dig(BN, ">=8", "null_sd"), "controls:backbone_null")
check("3.2 uniform z", rf"\(78 vs 3\.5 ± 2\.0, z = {N}\)", dig(BN, ">=8", "z"), "controls:backbone_null")
check("3.2 degree fold", rf"degree-matched\*\* null {N}-fold", dig(DN, ">=8", "degree_fold"), "controls:degree_null")
check("3.2 degree null mean", rf"null 59-fold \(78 vs {N} ±", dig(DN, ">=8", "degree_null_mean"), "controls:degree_null")
check("3.2 degree null sd", rf"\(78 vs 1\.3 ± {N}, z", dig(DN, ">=8", "degree_null_sd"), "controls:degree_null")
check("3.2 degree z", rf"\(78 vs 1\.3 ± 1\.3, z = {N}\)", dig(DN, ">=8", "degree_z"), "controls:degree_null")
check("3.2 backbone perms", rf"did any of {N} permutations reach", dig(CTL, "stats_final", "n_perm_backbone"),
      "controls:stats_final")
check("3.2 config fold lo", rf"\(fold {N}{D}", dig(RC, "ge8_fold_range", 0), "controls:recalibration")
check("3.2 config fold hi", rf"\(fold [\d.]+{D}{N}×", dig(RC, "ge8_fold_range", 1), "controls:recalibration")

dep = dig(DPB, "depths", default=[])
ge8 = [dig(x, "tiers", ">=8", "real") for x in dep if dig(x, "tiers", ">=8", "real") is not None]
fld = [dig(x, "tiers", ">=8", "fold") for x in dep if dig(x, "tiers", ">=8", "fold") is not None]
jac = [x["jaccard_vs_mid"] for x in dep if x.get("jaccard_vs_mid", 1) < 1]
mid = [x for x in dep if x.get("frac") == 0.5]
check("3.2 depth concepts lo", rf"significant \({N}{D}\d+ concepts", min(ge8) if ge8 else None, "depth_backbone")
check("3.2 depth concepts hi", rf"significant \(\d+{D}{N} concepts", max(ge8) if ge8 else None, "depth_backbone")
check("3.2 depth fold lo", rf"concepts, {N}{D}[\d.]+× over the random-gene", min(fld) if fld else None, "depth_backbone")
check("3.2 depth fold hi", rf"concepts, [\d.]+{D}{N}× over the random-gene", max(fld) if fld else None, "depth_backbone")
check("3.2 depth mid slice", rf"sweep gives {N} concepts", dig(mid, 0, "tiers", ">=8", "real"), "depth_backbone f=0.5")
check("3.2 depth jaccard lo", rf"Jaccard {N}{D}[\d.]+ versus the mid-layer", min(jac) if jac else None, "depth_backbone")
check("3.2 depth jaccard hi", rf"Jaccard [\d.]+{D}{N} versus the mid-layer", max(jac) if jac else None, "depth_backbone")
check("3.2 heldout jaccard", rf"backbone Jaccard {N} versus a", dig(HN, "observed_jaccard"), "controls:heldout_null")
check("3.2 heldout null", rf"versus a {N} two-draw baseline", dig(HN, "null_mean"), "controls:heldout_null")
check("3.2 heldout max", rf"two-draw baseline, max {N},", dig(HN, "null_max"), "controls:heldout_null")
check("3.2 KEGG share", rf"\({N} of the 78", dig(KEG, "kegg_sourced_in_ge8"), "kegg_robust")
check("3.2 KEGG drop concepts", rf"still finds {N} concepts", dig(KEG, "ge8_dropKEGG"), "kegg_robust")
check("3.2 KEGG drop fold", rf"≥8/10 models at {N}× over the random-gene",
      dig(KEG, "dropKEGG_GO_Reactome", ">=8", "fold"), "kegg_robust")

# ---- 3.3 annotation-independent -------------------------------------------
ck = dig(CTL, "survivor_nulls", "cka_null", default=[])
sv = dig(CTL, "survivor_nulls", "svd_null", default=[])
if ck:
    real = [x["real_cka"] for x in ck]; fl = [x["shuffled_cka"] for x in ck]
    check("3.3 CKA pairs", rf"all {N} model pairs", len(ck), "controls:cka_null")
    check("3.3 CKA min", rf"Real CKA spans {N}{D}", min(real), "controls:cka_null")
    check("3.3 CKA max", rf"Real CKA spans [\d.]+{D}{N} \(median", max(real), "controls:cka_null")
    check("3.3 CKA median", rf"\(median {N}\) against a shuffle", statistics.median(real), "controls:cka_null")
    check("3.3 CKA floor lo", rf"shuffle floor of {N}{D}", min(fl), "controls:cka_null")
    check("3.3 CKA floor hi", rf"shuffle floor of [\d.]+{D}{N}, and", max(fl), "controls:cka_null")
    ratio = min(r / f for r, f in zip(real, fl))
    rows.append(("3.3 'more than tenfold'", "controls:cka_null", f"worst pair {ratio:.1f}×",
                 "ok" if ratio > 10 else "MISMATCH", ratio > 10))
if sv:
    sae = [x["sae_var_explained"] for x in sv]; pca = [x["svd_var_at_k"] for x in sv]
    gaps = {x["model"]: x["sae_var_explained"] - x["svd_var_at_k"] for x in sv}
    check("3.3 SAE var lo", rf"code explains {N}{D}", min(sae), "controls:svd_null")
    check("3.3 SAE var hi", rf"code explains [\d.]+{D}{N} of residual", max(sae), "controls:svd_null")
    check("3.3 PCA var lo", rf"residual variance versus {N}{D}", min(pca), "controls:svd_null")
    check("3.3 PCA var hi", rf"variance versus [\d.]+{D}{N} for the best fixed", max(pca), "controls:svd_null")
    check("3.3 gap Tahoe", rf"gap runs from \+{N} \(Tahoe-x1\)", gaps.get("Tahoe"), "controls:svd_null")
    check("3.3 gap tGPT", rf"down to \+{N} for tGPT", gaps.get("tGPT"), "controls:svd_null")
    check("3.3 tGPT PCA", rf"PCA alone reaches {N}\)", dig([x for x in sv if x["model"] == "tGPT"], 0, "svd_var_at_k"),
          "controls:svd_null")
    neg = [x for x in sv if x.get("random_dict_var_explained", 0) < 0]
    rows.append(("3.3 random dict negative", "controls:svd_null", f"{len(neg)}/{len(sv)} models negative",
                 "ok" if len(neg) == len(sv) else "MISMATCH", len(neg) == len(sv)))
    rows.append(("3.3 all ten ahead", "controls:svd_null", f"{len(sv)} models, all ahead",
             "ok" if ("SAE is ahead in all ten models" in FLAT and len(sv) == 10
                      and all(x["sae_var_explained"] > x["svd_var_at_k"] for x in sv)) else "MISMATCH",
             "SAE is ahead in all ten models" in FLAT and len(sv) == 10
             and all(x["sae_var_explained"] > x["svd_var_at_k"] for x in sv)))

# ---- report ----------------------------------------------------------------
bad = [x for x in rows if not x[4]]
w = max(len(x[0]) for x in rows)
print(f"{'claim':<{w}}  {'source':<26}  {'source vs text':>26}  status")
print("-" * (w + 60))
for claim, src, val, status, ok in rows:
    print(f"{claim:<{w}}  {src:<26}  {val:>26}  {status if ok else '*** ' + status + ' ***'}")
print(f"\n{len(rows)-len(bad)}/{len(rows)} verified against source")
if bad:
    print("\nneeds a look:")
    for claim, src, val, status, _ in bad:
        print(f"   {claim:<26} {status:<18} {val}   [{src}]")
sys.exit(1 if any(x[3] == "MISMATCH" for x in rows) else 0)
