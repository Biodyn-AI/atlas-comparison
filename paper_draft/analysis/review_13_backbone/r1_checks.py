#!/usr/bin/env python
"""Hostile-review checks of notes 13 (backbone composition). Reuses bb_common.py and s2_mid_composition.py from
analysis/backbone_composition/ (same annotator caches, same T / O gene sets, same 50 % rule), so every number is on
the same ruler as notes 13.

(a) singleton programme classes under the primary rule (hit_weighted | complete | J>=0.3)
(b) median / mean of the programme count over the 30 clustering rules
(c) model-balanced gene-content class: each model that has a concept contributes weight 1 in total (its hit genes
    normalised within model), instead of pooling raw hit counts (which lets feature-rich models such as C2S and
    tGPT dominate). Same T, O and 50 % thresholds.
(d) how much of the 'all 7 of 7 agree: 127 at 0 % vs 15 at mid' contrast comes from MaxToki losing the
    translation block after the input: count all-7 concepts by class, and all-7 minus protein synthesis.
(e) near input: arithmetic floor -- with 7 non-embedding models, any >=8-of-10 concept must include >=1 of the
    three embedding-layer models.
Output: r1_checks.json
"""
import json, os, statistics, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
BC = os.path.join(os.path.dirname(HERE), "backbone_composition")
sys.path.insert(0, BC)
import bb_common as b                       # noqa: E402
from s2_mid_composition import load_slice, run_slice, concept_table  # noqa: E402

out = {}
s2 = json.load(open(f"{BC}/s2_mid_composition.json"))
bb = s2["backbone"]

# (a)
P = bb["programmes"]["hit_weighted|complete|J>=0.3"]["clusters"]
out["a_primary_clusters"] = [{"id": i, "n": p["n_members"], "class": p["class"], "representative": p["representative"],
                              "members": p["members"]} for i, p in enumerate(P, 1)]
out["a_singletons_by_class"] = dict(Counter(p["class"] for p in P if p["n_members"] == 1))
out["a_multi_by_class"] = dict(Counter(p["class"] for p in P if p["n_members"] > 1))

# (b)
vals = {k: v["n_programmes"] for k, v in bb["programmes"].items()}
out["b_rule_counts"] = vals
out["b_summary"] = {"n_rules": len(vals), "min": min(vals.values()), "max": max(vals.values()),
                    "median": statistics.median(vals.values()), "mean": round(sum(vals.values()) / len(vals), 2),
                    "rank_of_primary_25_(1=lowest)": sorted(vals.values()).index(25) + 1,
                    "complete_J>=0.3_by_basis": {k.split("|")[0]: v for k, v in vals.items() if k.endswith("complete|J>=0.3")}}

# (c) model-balanced classes at the mid layer
per = load_slice("mid")
TO = b.T_GENES | b.O_ALL
bal = {}
for t in bb["concepts"]:
    fT = []; fTO = []; perm = {}
    for m in b.ORDER:
        if t not in per[m]["terms"]: continue
        h = Counter()
        for fi, hs in per[m]["terms"][t]: h.update(hs)
        tot = sum(h.values())
        a = sum(c for g, c in h.items() if g in b.T_GENES) / tot
        c2 = sum(c for g, c in h.items() if g in TO) / tot
        fT.append(a); fTO.append(c2); perm[m] = {"frac_T": round(a, 3), "frac_TO": round(c2, 3), "n_hits": tot,
                                                 "n_features": len(per[m]["terms"][t])}
    mT, mTO = sum(fT) / len(fT), sum(fTO) / len(fTO)
    cls = "protein_synthesis" if mT >= 0.5 else "other_housekeeping" if mTO >= 0.5 else "specific"
    # share of pooled hits that come from the single largest-contributing model
    top_share = max(v["n_hits"] for v in perm.values()) / sum(v["n_hits"] for v in perm.values())
    bal[t] = {"class_model_balanced": cls, "frac_T_model_balanced": round(mT, 3), "frac_TO_model_balanced": round(mTO, 3),
              "class_weighted_pooled": bb["concepts"][t]["class_weighted"], "class_union": bb["concepts"][t]["class"],
              "largest_model_share_of_pooled_hits": round(top_share, 3), "per_model": perm}
out["c_class_counts_model_balanced"] = dict(Counter(v["class_model_balanced"] for v in bal.values()))
out["c_changed_vs_primary"] = {t: {k: v[k] for k in ("class_weighted_pooled", "class_model_balanced", "class_union",
                                                      "frac_T_model_balanced", "frac_TO_model_balanced",
                                                      "largest_model_share_of_pooled_hits")}
                               for t, v in bal.items() if v["class_model_balanced"] != v["class_weighted_pooled"]}
tops = sorted(v["largest_model_share_of_pooled_hits"] for v in bal.values())
out["c_largest_model_share_summary"] = {"median": statistics.median(tops), "min": tops[0], "max": tops[-1],
                                        "n_concepts_where_one_model_gives_>=50pct_of_hits": sum(1 for x in tops if x >= 0.5)}
dom = Counter()
for v in bal.values():
    mm = max(v["per_model"], key=lambda m: v["per_model"][m]["n_hits"]); dom[mm] += 1
out["c_model_giving_most_hits_count"] = dict(dom.most_common())
out["c_SARS_per_model"] = bal["Reactome:SARS-CoV-2-host Interactions R-HSA-9705683"]["per_model"]

# (d) all-7 contrast
SEVEN = [m for m in b.ORDER if m not in b.EMBED_L0]
d = {}
for nm in ("d0", "mid", "d100"):
    pr = load_slice(nm)
    res, cm, _, _ = run_slice(nm, pr, models=SEVEN, k=7)
    cc = res["class_counts"]["class_weighted"]
    # also: concepts present in all 6 non-MaxToki block-output models
    six = [m for m in SEVEN if m != "MaxToki"]
    res6, _, _, _ = run_slice(nm, pr, models=six, k=6)
    d[nm] = {"all7_n": res["n_concepts"], "all7_class_weighted": cc,
             "all7_minus_protein_synthesis": res["n_concepts"] - cc.get("protein_synthesis", 0),
             "all6_without_MaxToki_n": res6["n_concepts"],
             "all6_without_MaxToki_class_weighted": res6["class_counts"]["class_weighted"]}
out["d_all7_contrast"] = d

# (e)
out["e_arithmetic_floor"] = ("10 models, 3 of them embedding-layer models (AIDO, tGPT, GeneCompass): a concept in >=8 of 10 "
                             "must include at least 8-7 = 1 of the three. 'Every >=8 concept has one or two of the three' "
                             "is therefore half forced; only 'none has all three' and the 1-vs-2 split are informative.")
json.dump(out, open(f"{HERE}/r1_checks.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("a_primary_clusters", "b_rule_counts")}, indent=1)[:6000])
