#!/usr/bin/env python
"""Step 6. How the backbone depends on the annotator settings.

(a) Collects every stored annotator-grid result (read-only from the ORIGINAL repo's outputs/atlas/comparative):
    recalibrate.json, recalibrate2.json, recalibrate_robust.json, kegg_robust.json, stats_final.json,
    degree_null.json, recalibrate_final.json.
(b) Adds the new 250-permutation nulls from s4_nulls.json (caps 150/200/300, top-12).
(c) For each re-run setting, the composition of the real >=8 set: gene-content classes and programme counts,
    using the same code as s2 (annotate_detail with that cap / top-N on the headline mid layer).
Output: s5_sensitivity.json
"""
import json
from collections import Counter

import bb_common as b
import s2_mid_composition as s2

C = b.ORIG_COMPARATIVE
stored = {n: json.load(open(f"{C}/{n}.json")) for n in
          ("recalibrate", "recalibrate2", "recalibrate_robust", "kegg_robust", "stats_final", "degree_null", "recalibrate_final")}
nulls = json.load(open(f"{b.OUT}/s4_nulls.json"))

rows = []
for src in ("recalibrate", "recalibrate2"):
    for r in stored[src]["configs"]:
        rows.append({"source": f"{src}.json:configs", "config": r["config"], "tier": "10 of 10 (core)", "real": r["real_core"],
                     "null_mean": r["rand_core_mean"], "null_sd": r["rand_core_sd"], "n_perm": 20,
                     "real_rate_pct": r["real_rate"], "rand_rate_pct": r["rand_rate"], "z_stored": r["core_z"],
                     "fold": round(r["real_core"] / r["rand_core_mean"], 2) if r["rand_core_mean"] > 0 else None})
for r in stored["recalibrate_robust"]["grid"]:
    for k in ("7", "8", "9", "10"):
        v = r[f">={k}"]
        rows.append({"source": "recalibrate_robust.json:grid", "config": r["config"], "tier": f">={k}", "real": v["real"],
                     "null_mean": v["null"], "null_sd": None, "n_perm": stored["recalibrate_robust"]["n_perm"],
                     "fold_stored_floor0.3": v["fold"]})
for arm in ("full_GO_KEGG_Reactome", "dropKEGG_GO_Reactome"):
    for k in ("7", "8", "9", "10"):
        v = stored["kegg_robust"][arm][f">={k}"]
        rows.append({"source": f"kegg_robust.json:{arm}", "config": "top10 a3 max200" + (" (no KEGG)" if "drop" in arm else ""),
                     "tier": f">={k}", "real": v["real"], "null_mean": v["null"], "null_sd": None,
                     "n_perm": stored["kegg_robust"]["n_perm"], "fold_stored_floor0.3": v["fold"]})
for k, v in stored["stats_final"]["backbone_null"].items():
    rows.append({"source": "stats_final.json:backbone_null", "config": "top10 a3 max200 (headline)", "tier": k,
                 "real": v["real"], "null_mean": v["null_mean"], "null_sd": v["null_sd"], "n_perm": 250,
                 "p_emp": v["p_emp"], "fold_stored_floor0.1": v["fold"]})
for k, v in stored["degree_null"]["tiers"].items():
    rows.append({"source": "degree_null.json:tiers (degree-matched null)", "config": "top10 a3 max200 (headline)", "tier": k,
                 "real": v["real"], "null_mean": v["degree_null_mean"], "null_sd": v["degree_null_sd"], "n_perm": 250,
                 "p_emp": v["degree_p"], "fold_stored_floor0.1": v["degree_fold"]})

new = {}
for name, cfg in nulls["configs"].items():
    if not name.startswith("mid_"): continue
    new[name] = {"cap": cfg["cap"], "top": cfg["top"], "n_perm": cfg["n_perm"], "tiers": cfg["tiers"],
                 "programmes_fullset_complete_J>=0.3": cfg["programmes_fullset_complete_J>=0.3"],
                 "programmes_fullset_complete_J>=0.5": cfg["programmes_fullset_complete_J>=0.5"]}

# composition of the real >=8 set under each re-run setting
comp = {}
for name, (mx, top) in {"mid_cap150": (150, 10), "mid_cap200": (200, 10), "mid_cap300": (300, 10),
                        "mid_cap200_top12": (200, 12)}.items():
    ann = b.Annotator(mx, top=top)
    per = {}
    for m in b.ORDER:
        feats, L = b.load_cat(b.mid_path(m))
        det = ann.annotate_detail(feats)
        per[m] = {"layer": L, "terms": {t: [[fi, list(h)] for fi, h in v] for t, v in det.items()}}
    res, cm, _, _ = s2.run_slice(name, per)
    assert res["n_concepts"] == new[name]["tiers"][">=8"]["real"], (name, res["n_concepts"])
    comp[name] = {"n_ge8": res["n_concepts"], "class_weighted": res["class_counts"]["class_weighted"],
                  "class_union": res["class_counts"]["class"],
                  "share_protein_synthesis_weighted": round(res["class_counts"]["class_weighted"].get("protein_synthesis", 0) / res["n_concepts"], 3),
                  "n_distinct_dominant_labels": res["n_distinct_dominant_labels"],
                  "dominant_label_counts": res["dominant_label_counts"],
                  "programmes": {k: res["programmes"][k]["n_programmes"] for k in
                                 ("hit_weighted|complete|J>=0.3", "features|complete|J>=0.3", "full_sets|complete|J>=0.3",
                                  "hit_weighted|complete|J>=0.5", "features|complete|J>=0.5", "full_sets|complete|J>=0.5")},
                  "programmes_by_class_hit_weighted_J>=0.3": res["programmes"]["hit_weighted|complete|J>=0.3"]["n_by_class"]}
    print(name, comp[name]["n_ge8"], comp[name]["class_weighted"], comp[name]["programmes"], flush=True)

# terms gained / lost relative to the headline 78
h78 = set(json.load(open(f"{b.OUT}/s2_mid_composition.json"))["backbone"]["concepts"])
out = {"stored_grid_rows": rows, "new_nulls_250perm": new, "composition_real_ge8": comp,
       "note": "fold_raw = real / null mean with no floor; stored folds use floors 0.3 (recalibrate_robust, kegg_robust) "
               "or 0.1 (stats_final, degree_null)."}
print("==>", b.dump(out, "s5_sensitivity.json"))
