#!/usr/bin/env python
"""Step 7 (job item 7): rebuild the released prediction list with the corrected same-family rule.

Rebuild exactly as hypothesis_trrust3.py:128-138: pairs in the W>=5 graphs of >= 2 of the five
passing models; drop pairs in TRRUST, in STRING (string_edges.json), or co-members of any curated
GO_BP/KEGG/Reactome set of <= 200 genes. Then drop same-family pairs with
  (old)  the regex rule actually used for the release (hypothesis_trrust3.py:124-126,136)
  (new)  the corrected rule of hypothesis_robust.py:35-49 (copied in common.same_family)
Check the old-rule list against the released hypothesis_pairs_full.json (4,349 pairs), count the
corrected list, and recompute the literature numbers that CAN be recomputed from the per-pair fields
stored in hypothesis_pairs_full.json (co_mentions = uncapped paper count with both genes; papers).
The capped co-mention test (mega-paper cap, bipartite-shuffle null) needs NCBI gene2pubmed +
gene_info, which are not in the repo or on this machine; it was NOT re-run.
Curated-set membership is checked with a gene -> set index (no giant pair set), same definition.
Output: OUT/s07_released_list.json
"""
import json, os, re, sys, pickle
from collections import defaultdict, Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

truth, _, _ = load_truth()
STRING = load_string()
g2s = defaultdict(set); sid = 0
for src in ("GO_BP", "KEGG", "Reactome"):
    for t, gs in json.load(open(f"{G}/{src}_gene_sets.json")).items():
        gl = {str(x).upper() for x in gs}
        if len(gl) <= 200:
            for g in gl: g2s[g].add(sid)
        sid += 1
in_seen = lambda a, b: bool(g2s[a] & g2s[b]) if a in g2s and b in g2s else False

PRED, NFEAT = pickle.load(open(f"{C}/.cofire_cache.pkl", "rb"))
FAM = re.compile(r"^([A-Z]{2,}[0-9]*?)[0-9]*[A-Z]?$")
def family_old(g):
    m = FAM.match(g); return m.group(1) if m else g
def same_old(a, b):
    fa, fb = family_old(a), family_old(b); return fa == fb and len(fa) >= 3

allp = defaultdict(int)
for m in PASSED:
    for e in PRED[m]: allp[e] += 1
cands = []
for e, n in allp.items():
    if n < 2 or e in truth or e in STRING or in_seen(*e): continue
    cands.append({"pair": list(e), "n_models": n, "models": sorted(m for m in PASSED if e in PRED[m]),
                  "weight": int(sum(PRED[m][e] for m in PASSED if e in PRED[m])),
                  "old_rule": same_old(*e), "new_rule": same_family(*e)})
old_list = [c for c in cands if not c["old_rule"]]
new_list = [c for c in cands if not c["new_rule"]]
both_list = [c for c in cands if not c["new_rule"] and not c["old_rule"]]

rel = json.load(open(f"{ORIG}/outputs/atlas/comparative/hypothesis_pairs_full.json"))
relset = {tuple(p["pair"]) for p in rel["pairs"]}
oldset = {tuple(c["pair"]) for c in old_list}
newset = {tuple(c["pair"]) for c in new_list}
relmap = {tuple(p["pair"]): p for p in rel["pairs"]}


def lit(pairs):
    rows = [relmap[p] for p in pairs if p in relmap]
    testable = [r for r in rows if r.get("co_mentions") is not None and r.get("papers") and None not in r["papers"]]
    never = [r for r in testable if r["co_mentions"] == 0]
    both_studied = [r for r in never if min(r["papers"]) >= 20]
    return {"n_pairs": len(pairs), "n_with_stored_fields": len(rows), "n_testable": len(testable),
            "n_comentioned_uncapped": sum(r["co_mentions"] > 0 for r in testable),
            "pct_comentioned_uncapped": round(100 * sum(r["co_mentions"] > 0 for r in testable) / max(len(testable), 1), 2),
            "n_never_comentioned": len(never), "n_never_but_both_ge20_papers": len(both_studied),
            "never_but_both_studied": [r["pair"] for r in both_studied]}


# what the stored fields look like for unmapped pairs
sample_unmapped = [p for p in rel["pairs"] if p.get("co_mentions") is None or not p.get("papers") or None in (p.get("papers") or [None])][:3]
new_flagged_in_release = [c for c in old_list if c["new_rule"]]
tier = Counter(c["n_models"] for c in new_flagged_in_release)
xist = [p for p in rel["pairs"] if "XIST" in p["pair"]]
res = {
    "n_candidates_before_family_filter": len(cands),
    "old_rule": {"n_dropped": sum(c["old_rule"] for c in cands), "n_kept": len(old_list),
                 "identical_to_released_list": oldset == relset, "released_n": rel["n"]},
    "new_rule": {"n_dropped": sum(c["new_rule"] for c in cands), "n_kept": len(new_list)},
    "overlap": {"dropped_by_both": sum(c["old_rule"] and c["new_rule"] for c in cands),
                "dropped_only_old": sum(c["old_rule"] and not c["new_rule"] for c in cands),
                "dropped_only_new": sum(c["new_rule"] and not c["old_rule"] for c in cands),
                "kept_by_both": len(both_list)},
    "released_pairs_flagged_by_new_rule": {"n": len(new_flagged_in_release), "by_n_models": dict(sorted(tier.items())),
                                            "examples_top": [c["pair"] for c in sorted(new_flagged_in_release, key=lambda c: (-c["n_models"], -c["weight"]))[:25]]},
    "pairs_dropped_only_by_old_rule": [c["pair"] for c in cands if c["old_rule"] and not c["new_rule"]][:60],
    "corrected_list_by_n_models": dict(sorted(Counter(c["n_models"] for c in new_list).items())),
    "literature_fields_released_list": lit(relset),
    "literature_fields_corrected_list": lit(newset),
    "literature_fields_corrected_list_note": "only pairs present in the released list carry stored co_mentions/papers; "
                                              "pairs dropped only by the old rule and kept by the new rule have no stored literature fields",
    "published_hypothesis_pubmed": {k: json.load(open(f"{ORIG}/outputs/atlas/comparative/hypothesis_pubmed.json"))[k]
                                     for k in ("n_pairs_testable", "observed_pairs_comentioned", "fold", "p_emp",
                                               "n_never_comentioned", "n_never_but_both_studied",
                                               "observed_pairs_comentioned_uncapped")},
    "xist_pairs_in_release": [{"pair": p["pair"], "papers": p.get("papers"), "co_mentions": p.get("co_mentions"),
                               "n_models": p["n_models"]} for p in xist],
    "sample_unmapped_rows": sample_unmapped,
    "capped_literature_test_rerun": False,
    "capped_literature_test_reason": "gene2pubmed.gz and Homo_sapiens.gene_info.gz (NCBI) are not in the repository and were not "
                                      "downloaded for this analysis, so the capped literature test was not re-run",
}
json.dump(res, open(f"{OUT}/s07_released_list.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ("pairs_dropped_only_by_old_rule",)}, indent=1)[:6000])
