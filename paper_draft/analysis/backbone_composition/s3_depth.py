#!/usr/bin/env python
"""Steps (4)-(5). The >=8-of-10 backbone at each depth slice, its composition, the depth-invariant core, and the
near-input (0 %) slice with and without the three models whose layer 0 is the raw token embedding.

Uses the per-slice caches written by s1_annotate_slices.py and the composition code of s2_mid_composition.py
(gene-content classes, programme clustering, content labels). No nulls here (see s4_nulls.py).
Output: s3_depth.json
"""
import json
from collections import Counter

import numpy as np
import bb_common as b
from s2_mid_composition import load_slice, run_slice, content, classify

DEPTHS = ["d0", "d25", "d50", "d75", "d100"]
SEVEN = [m for m in b.ORDER if m not in b.EMBED_L0]
PRIMARY = ["hit_weighted|complete|J>=0.3", "features|complete|J>=0.3", "full_sets|complete|J>=0.3",
           "hit_weighted|complete|J>=0.5", "features|complete|J>=0.5", "full_sets|complete|J>=0.5"]


def slim(res):
    """summary of one slice (drop the per-cluster detail except for the primary rule)"""
    out = {"n_concepts": res["n_concepts"], "class_counts": res["class_counts"],
           "share_protein_synthesis": round(res["class_counts"]["class"].get("protein_synthesis", 0) / max(res["n_concepts"], 1), 3),
           "share_protein_synthesis_weighted": round(res["class_counts"]["class_weighted"].get("protein_synthesis", 0) / max(res["n_concepts"], 1), 3),
           "n_distinct_dominant_labels": res["n_distinct_dominant_labels"],
           "dominant_label_counts": res["dominant_label_counts"],
           "programmes": {k: {"n_programmes": res["programmes"][k]["n_programmes"],
                              "n_by_class": res["programmes"][k]["n_by_class"]} for k in PRIMARY},
           "clusters_hit_weighted_complete_J>=0.3": [
               {"representative": p["representative"], "n_members": p["n_members"], "class": p["class"],
                "dominant_label": p["dominant_label"], "top_hit_genes": p["top_hit_genes"][:6]}
               for p in res["programmes"]["hit_weighted|complete|J>=0.3"]["clusters"]],
           "concepts": {t: {"n_models": c["n_models"], "class": c["class"], "class_weighted": c["class_weighted"],
                            "frac_T_weighted": c["frac_T_weighted"], "dominant_label": c["dominant_label"],
                            "missing": c["missing"]} for t, c in res["concepts"].items()}}
    return out


out = {"slices": {}, "notes": "k = 8 of 10 for 10-model slices; k = 6 of 7 (and 5 of 7) for the 7-model slices"}
per_all = {nm: load_slice(nm) for nm in DEPTHS + ["mid", "d0_block1"]}
counts = {}; hits_by = {}; sets8 = {}
for nm in DEPTHS + ["mid", "d0_block1"]:
    res, cm, nfeat, hits = run_slice(nm, per_all[nm])
    out["slices"][nm] = slim(res); out["slices"][nm]["layers"] = {m: per_all[nm][m]["layer"] for m in b.ORDER}
    counts[nm] = cm; hits_by[nm] = hits; sets8[nm] = set(res["concepts"])
    print(nm, res["n_concepts"], res["class_counts"]["class"], {k: res["programmes"][k]["n_programmes"] for k in PRIMARY[:3]}, flush=True)

head = sets8["mid"]
for nm in out["slices"]:
    s = sets8[nm]
    out["slices"][nm]["overlap_with_headline78"] = {"shared": len(s & head), "only_here": len(s - head),
                                                    "only_in_78": len(head - s), "jaccard": round(b.jaccard(s, head), 3)}

# ---- depth-invariant core ----
def pooled_class(t):
    h = Counter()
    for nm in DEPTHS:
        h.update(hits_by[nm].get(t, Counter()))
    full = b.RAW[t.split(":", 1)[0]][t.split(":", 1)[1]]
    c = content(h, full)
    return classify(c, key_T="frac_T_weighted", key_TO="frac_TO_weighted"), classify(c), c["frac_T_weighted"], b.dominant_label(h)[0]

union8 = set().union(*[sets8[nm] for nm in DEPTHS])
alld = {t: [counts[nm].get(t, 0) for nm in DEPTHS] for t in sorted(union8 | head)}
core8 = sorted(t for t, v in alld.items() if min(v) >= 8)
core7 = sorted(t for t in set().union(*[set(counts[nm]) for nm in DEPTHS])
               if min(counts[nm].get(t, 0) for nm in DEPTHS) >= 7)
reader30 = sorted(t for t in head if all(t in sets8[nm] for nm in DEPTHS))
core_info = {}
for t in sorted(set(core8) | set(core7) | set(reader30)):
    cw, cu, fT, lab = pooled_class(t)
    core_info[t] = {"counts_d0_d25_d50_d75_d100": [counts[nm].get(t, 0) for nm in DEPTHS], "class_weighted": cw,
                    "class_union": cu, "frac_T_weighted_pooled": fT, "dominant_label": lab}
out["depth_invariant"] = {
    "ge8_all_five": {"n": len(core8), "class_weighted": dict(Counter(core_info[t]["class_weighted"] for t in core8)),
                     "class_union": dict(Counter(core_info[t]["class_union"] for t in core8)),
                     "labels": dict(Counter(core_info[t]["dominant_label"] for t in core8)), "terms": core8},
    "reader_30_(78 & ge8 at all five)": {"n": len(reader30),
                                         "class_weighted": dict(Counter(core_info[t]["class_weighted"] for t in reader30)),
                                         "class_union": dict(Counter(core_info[t]["class_union"] for t in reader30)),
                                         "terms": reader30},
    "ge7_all_five": {"n": len(core7), "class_weighted": dict(Counter(core_info[t]["class_weighted"] for t in core7)),
                     "labels": dict(Counter(core_info[t]["dominant_label"] for t in core7)),
                     "non_protein_synthesis": {t: core_info[t] for t in core7 if core_info[t]["class_weighted"] != "protein_synthesis"}},
    "core_info": core_info,
}
# per-concept min count across the five slices, for the 78
mins = {t: min(alld[t]) for t in head}
out["headline78_min_count_over_depths"] = {
    "histogram": dict(sorted(Counter(mins.values()).items())),
    "by_class": {c: dict(sorted(Counter(mins[t] for t in head if out["slices"]["mid"]["concepts"][t]["class_weighted"] == c).items()))
                 for c in ("protein_synthesis", "other_housekeeping", "specific")},
    "counts_per_concept": {t: alld[t] for t in sorted(head, key=lambda x: (-mins[x], x))}}
# how many depth slices each of the 78 is >=8 at, by class
nd = {t: sum(t in sets8[nm] for nm in DEPTHS) for t in head}
out["headline78_n_depths_at_ge8_by_class"] = {
    c: dict(sorted(Counter(nd[t] for t in head if out["slices"]["mid"]["concepts"][t]["class_weighted"] == c).items()))
    for c in ("protein_synthesis", "other_housekeeping", "specific")}

# ---- near input ----
d0, d0b = sets8["d0"], sets8["d0_block1"]
cls0 = out["slices"]["d0"]["concepts"]
only0 = sorted(d0 - head)
emb_share = {}
for t in only0:
    emb_share[t] = [m for m in b.EMBED_L0 if t in per_all["d0"][m]["terms"]]
out["near_input"] = {
    "d0_vs_mid": out["slices"]["d0"]["overlap_with_headline78"],
    "d0_only_concepts_by_class": dict(Counter(cls0[t]["class_weighted"] for t in only0)),
    "d0_only_concepts_by_label": dict(Counter(cls0[t]["dominant_label"] for t in only0)),
    "d0_only_concepts": {t: {"class": cls0[t]["class_weighted"], "label": cls0[t]["dominant_label"],
                             "n_models": cls0[t].get("n_models", counts["d0"][t]),
                             "embedding_layer_models_having_it": emb_share[t]} for t in only0},
    "d0_only_n_embedding_layer_models_having_it_hist": dict(sorted(Counter(len(v) for v in emb_share.values()).items())),
    "d0_only_still_ge8_when_block1_used_for_the_three": sum(1 for t in only0 if t in d0b),
    "embedding_models_per_ge8_concept_hist": {
        nm: dict(sorted(Counter(sum(1 for m in b.EMBED_L0 if t in per_all[nm][m]["terms"]) for t in sets8[nm]).items()))
        for nm in ("d0", "d0_block1", "mid", "d50")},
    "d0_block1_vs_d0": {"shared": len(d0 & d0b), "jaccard": round(b.jaccard(d0, d0b), 3),
                        "n_d0": len(d0), "n_d0_block1": len(d0b)},
    "d0_block1_vs_mid": out["slices"]["d0_block1"]["overlap_with_headline78"],
}
# 7-model slices (drop AIDO, tGPT, GeneCompass)
seven = {}
for nm in DEPTHS + ["mid"]:
    row = {}
    for k in (5, 6):
        res, cm7, _, _ = run_slice(nm, per_all[nm], models=SEVEN, k=k)
        row[f">={k}_of_7"] = {"n_concepts": res["n_concepts"], "class_weighted": res["class_counts"]["class_weighted"],
                              "share_protein_synthesis_weighted": round(res["class_counts"]["class_weighted"].get("protein_synthesis", 0) / max(res["n_concepts"], 1), 3),
                              "programmes_hit_weighted_complete_J>=0.3": res["programmes"]["hit_weighted|complete|J>=0.3"]["n_programmes"],
                              "programmes_full_sets_complete_J>=0.3": res["programmes"]["full_sets|complete|J>=0.3"]["n_programmes"],
                              "n_distinct_dominant_labels": res["n_distinct_dominant_labels"],
                              "terms": sorted(res["concepts"])}
    seven[nm] = row
    print("7-model", nm, {k: v["n_concepts"] for k, v in row.items()}, flush=True)
h7 = set(seven["mid"][">=6_of_7"]["terms"])
for nm in seven:
    s = set(seven[nm][">=6_of_7"]["terms"])
    seven[nm][">=6_of_7"]["jaccard_vs_mid_7model"] = round(b.jaccard(s, h7), 3)
    seven[nm][">=6_of_7"]["shared_with_mid_7model"] = len(s & h7)
out["seven_models_no_embedding_layer0"] = {"models": SEVEN, "slices": seven}

# compact per-depth table
out["table"] = [{"slice": nm, "n_ge8": out["slices"][nm]["n_concepts"],
                 "protein_synthesis": out["slices"][nm]["class_counts"]["class_weighted"].get("protein_synthesis", 0),
                 "other_housekeeping": out["slices"][nm]["class_counts"]["class_weighted"].get("other_housekeeping", 0),
                 "specific": out["slices"][nm]["class_counts"]["class_weighted"].get("specific", 0),
                 "share_protein_synthesis_weighted": out["slices"][nm]["share_protein_synthesis_weighted"],
                 **{f"programmes[{k}]": out["slices"][nm]["programmes"][k]["n_programmes"] for k in PRIMARY},
                 "n_distinct_dominant_labels": out["slices"][nm]["n_distinct_dominant_labels"],
                 "jaccard_vs_78": out["slices"][nm]["overlap_with_headline78"]["jaccard"]}
                for nm in DEPTHS + ["mid", "d0_block1"]]
print("==>", b.dump(out, "s3_depth.json"))
for r in out["table"]: print(r)
print("core8", out["depth_invariant"]["ge8_all_five"]["n"], out["depth_invariant"]["ge8_all_five"]["class_weighted"])
print("reader30", out["depth_invariant"]["reader_30_(78 & ge8 at all five)"]["n"], out["depth_invariant"]["reader_30_(78 & ge8 at all five)"]["class_weighted"])
print("core7", out["depth_invariant"]["ge7_all_five"]["n"], out["depth_invariant"]["ge7_all_five"]["class_weighted"], out["depth_invariant"]["ge7_all_five"]["labels"])
