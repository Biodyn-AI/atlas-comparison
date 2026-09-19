#!/usr/bin/env python
"""Rebuild the atlas page's `hypothesis` block from the result files it reports.

The block that feeds the site's prediction, independent-evidence and novelty panels was assembled by
hand. Nothing regenerated it and nothing checked it, so when the literature null was made
reproducible the page kept serving the old z = 40.48 while the manuscript said 34.7 -- the same
class of drift the manuscript audit exists to catch, on the public half of the project.

This derives every field from the canonical JSONs, so re-running an analysis and re-injecting is
enough to keep the site true. It edits only the `hypothesis` key of atlas_full_notf.json.
    python scripts/atlas_hypothesis_block.py [--check]   -> rewrites (or just reports drift)
"""
from __future__ import annotations

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import json, sys

C = f"{_B}/outputs/atlas/comparative"
ATLAS = f"{C}/atlas_full_notf.json"
N_EXAMPLES = 18          # what the page lists; the full set stays in the result files
CHECK = "--check" in sys.argv


def load(n):
    return json.load(open(f"{C}/{n}"))


T2, FIN = load("hypothesis_trrust2.json"), load("hypothesis_final.json")
ROB, SM = load("hypothesis_robust.json"), load("hypothesis_sizematched.json")
PUB, PT = load("hypothesis_pubmed.json"), load("hypothesis_perturb.json")

# per-model recovery on held-out TRRUST
per_model = {m: {k: v[k] for k in ("n_pairs", "hits", "null_mean", "fold", "p_emp")}
             for m, v in T2["per_model"].items()}

# the recurrence curve. A fold against a null that produced no edge at all is not a number, and the
# page must not print one: 1.5e9 rendered as "1520000000x" would be the most confident claim on it.
curve = {}
for k, v in T2["cross_model_curve"].items():
    curve[k] = {"n_pairs": v["n_pairs"], "hits": v["hits"], "precision": v["precision"],
                "null_precision": v["null_precision"],
                "fold": None if not v["null_precision"] else v["fold"], "p_emp": v["p_emp"]}

sizematched = {"budget": SM["budget"], "rank_preserved": SM["ranking_preserved"],
               "rank_corr": SM["spearman_full_vs_matched"],
               "models": {m: {"graph": d["graph"], "testable": d["testable"],
                              "fold": d.get("topN", {}).get("fold"),
                              "p": d.get("topN", {}).get("p_emp"),
                              "rand_fold": d.get("randN", {}).get("fold_mean"),
                              "full_fold": d.get("full_fold")}
                          for m, d in SM["models"].items()}}

lit = {"observed": PUB["observed_pairs_comentioned"], "testable": PUB["n_pairs_testable"],
       "null_mean": PUB["null_pairs_comentioned_mean"], "null_sd": PUB["null_sd"],
       "fold": PUB["fold"], "z": PUB["z"], "p": PUB["p_emp"], "cap": PUB["mega_paper_cap"]}

perturb = {ln: {"pairs": PT[ln]["n_ordered_pairs"], "rate": PT[ln]["top5pct_rate"],
                "null": PT[ln]["null_top5pct_rate"], "fold": PT[ln]["top5pct_fold"],
                "z": PT[ln]["z_top5"], "p": PT[ln]["p_top5"]} for ln in ("K562", "RPE1")}

PAIRS = json.load(open(f"{C}/hypothesis_pairs_full.json"))
PAIRS_N = PAIRS["n"]
# every released pair, compactly, so the page can list, filter and export them rather than show
# fourteen examples and a total: [gene A, gene B, models agreeing, weight, literature co-mentions]
def _same_family(a, b):
    """Same naming rule as variant R1 in hypothesis_robust.py: a shared stem of >= 3 characters and
    short differing suffixes (ACTB/ACTG2, S100A11/S100A4, ATP5ME/ATP5MK)."""
    a, b = a.upper(), b.upper()
    if a == b:
        return True
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    stem = a[:n].rstrip("-")
    return len(stem) >= 3 and len(a) - n <= 3 and len(b) - n <= 3


# ... plus whether the two genes are paralogues by name (ACTB-ACTG2), the easiest kind of link.
# "Absent from every database" is not "new": the co-mention count says whether the literature
# already puts the two together, and the page must show both or it contradicts itself.
PAIRS_ALL = sorted(([c["pair"][0], c["pair"][1], c["n_models"], c["weight"], c.get("co_mentions"),
                     _same_family(c["pair"][0], c["pair"][1])]
                    for c in PAIRS["pairs"]), key=lambda r: (-r[2], -r[3]))

block = {
    # the curve and the per-model table are the unselected all-ten result; the shortlist and the
    # robustness variants are the five models that passed, which is what the page's captions say
    "scope": "all ten models (unselected)",
    "truth": FIN["truth"],
    # the per-model bars the page draws come from trrust2, so the rewiring count printed beside
    # them must be trrust2's (500), not hypothesis_final's (200) -- the page said 200 for a while
    "n_perm": T2["n_perm"],
    "n_perm_curve": T2["n_perm_curve"],
    "n_hypotheses": PAIRS_N,
    "per_model": per_model,
    "curve": curve,
    "validated": FIN["models_validated"],
    "examples": FIN["hypotheses"][:N_EXAMPLES],
    "pairs_all": PAIRS_ALL,
    # the near-duplicate check of Table S12, for the prediction panel's reading
    "redundancy": {m: {k: v[k] for k in ("fire_pct_community", "fire_pct_rest", "pct_pred_block_carried",
                                          "hits_block_carried", "hits_rest", "hits_dedup")}
                   for m, v in load("pair_redundancy.json")["models"].items()},
    "pairs_all_columns": ["gene_a", "gene_b", "n_models", "weight", "co_mentions", "same_family"],
    # the page never reads this sub-block, but its key naming is part of the published file
    "robust": {name: {f"k{k}": v for k, v in var.items()} for name, var in ROB["variants"].items()},
    "curve_selected5": {k: {"fold": v["fold"], "hits": v["hits"]}
                        for k, v in FIN["cross_model_curve"].items()
                        if int(k) <= 4},
    "sizematched": sizematched,
    "untestable": [m for m, d in SM["models"].items() if not d["testable"]],
    "independent": {"literature": lit, "perturb": perturb},
    "novelty": {"never_comentioned": PUB["n_never_comentioned"], "testable": PUB["n_pairs_testable"],
                "both_studied": PUB["n_never_but_both_studied"],
                "examples": [c["pair"] for c in PUB["never_but_both_studied"]]},
}

atlas = json.load(open(ATLAS))
old = atlas.get("hypothesis")

# The site's copy of the CKA / SVD nulls was a four-model subset from an early run; the paper's
# Tables S1 and S7 read the full ten-model cka_svd_null.json. The site reads the same file now.
SN = load("cka_svd_null.json")
old_sn = atlas.get("controls", {}).get("survivor_nulls")

# The backbone term list the page shows was cut to 40 of 78 when first written. backbone_terms.py
# recomputes it in full under the same annotator; the page keeps its "SOURCE:term" format.
try:
    BT = load("backbone_terms.json")["tiers"][">=8"]
    BT_TERMS = [f"{t['source']}:{t['term']}" for t in BT]
except Exception:
    BT_TERMS = None
old_bt = atlas.get("controls", {}).get("recalibration", {}).get("final", {}).get("shared_ge8_terms")


def diffs(a, b, path=""):
    """Every leaf that changed, so a rebuild says what it moved rather than only that it ran."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            yield from diffs(a.get(k), b.get(k), f"{path}/{k}")
    elif a != b:
        yield path, a, b


changed = list(diffs(old, block)) if old else [("(no previous block)", None, None)]
if old_sn != SN:
    n_old = {k: len(v) for k, v in (old_sn or {}).items()}
    changed.append(("/controls/survivor_nulls", f"{n_old}", f"{ {k: len(v) for k, v in SN.items()} } (cka_svd_null.json)"))
if BT_TERMS is not None and old_bt != BT_TERMS:
    changed.append(("/controls/recalibration/final/shared_ge8_terms", f"{len(old_bt or [])} terms", f"{len(BT_TERMS)} terms (backbone_terms.json)"))
if not changed:
    print("hypothesis block already matches the result files")
elif CHECK:
    print(f"{len(changed)} полей разошлись с результатами:")
    for p, a, b in changed[:40]:
        print(f"  {p:<52} {json.dumps(a)[:28]:>28}  ->  {json.dumps(b)[:28]}")
else:
    atlas["hypothesis"] = block
    atlas.setdefault("controls", {})["survivor_nulls"] = SN
    if BT_TERMS is not None:
        atlas["controls"].setdefault("recalibration", {}).setdefault("final", {})["shared_ge8_terms"] = BT_TERMS
    json.dump(atlas, open(ATLAS, "w"))
    print(f"переписано {len(changed)} полей в atlas_full_notf.json:")
    for p, a, b in changed[:40]:
        print(f"  {p:<52} {json.dumps(a)[:28]:>28}  ->  {json.dumps(b)[:28]}")
    print("\n==> теперь запусти scripts/inject_atlas.py и выложи index.html")
sys.exit(1 if (CHECK and changed) else 0)
