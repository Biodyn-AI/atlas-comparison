#!/usr/bin/env python
"""Step 1. Run the calibrated annotator (exact repo rule) on every layer slice we need and cache, per model and
per concept, which features passed and which of their top-10 genes fell in the set.

Slices:
  mid        : cats[len(cats)//2]           (headline rule, recalibrate_final.py:64)  -> should give 142/78/10/2
  d0,d25,d50,d75,d100 : cats[round(f*(n-1))] (depth_backbone_fast.py:85)             -> should give 90/102/75/48/41
  d0_block1  : like d0, but AIDO/tGPT/GeneCompass use their layer-1 catalogue (block-1 output, like the other 7)
For every slice the detailed annotator is checked against the exact copy of recalibrate_final.annotate on
every model; the script stops if any concept set differs.
Output: $ATLAS_CACHE/bb_cache/slice_<name>.json  (per model: layer, n_features, {term: [[feature, [hits]], ...]})
        analysis/backbone_composition/s1_slices_summary.json
"""
import json, os, sys, time
import bb_common as b

SCR = b.CACHE + "/bb_cache"
os.makedirs(SCR, exist_ok=True)
ann = b.Annotator(200)

SLICES = {"mid": lambda m: b.mid_path(m)}
for f, nm in [(0.0, "d0"), (0.25, "d25"), (0.5, "d50"), (0.75, "d75"), (1.0, "d100")]:
    SLICES[nm] = (lambda f_: (lambda m: b.depth_path(m, f_)))(f)
SLICES["d0_block1"] = lambda m: b.layer_path(m, 1) if m in b.EMBED_L0 else b.depth_path(m, 0.0)

summary = {"annotator": "top10 genes & background, >=3 in GO_BP/KEGG/Reactome set of 5-200 genes, hypergeom, BH<=0.05 "
                        "(exact copy of recalibrate_final.annotate)", "slices": {}}
for name, pf in SLICES.items():
    t0 = time.time(); per = {}; sets = {}
    for m in b.ORDER:
        path = pf(m)
        feats, L = b.load_cat(path)
        det = ann.annotate_detail(feats)
        ex = ann.annotate_exact(feats)
        if set(det) != ex:
            print(f"MISMATCH {name} {m}: detail {len(det)} vs exact {len(ex)}"); sys.exit(1)
        per[m] = {"layer": L, "file": os.path.basename(path), "n_features": len(feats),
                  "terms": {t: [[fi, list(h)] for fi, h in v] for t, v in det.items()}}
        sets[m] = set(det)
    cm = b.count_models(sets)
    spec = {str(k): sum(1 for v in cm.values() if v == k) for k in range(1, 11)}
    json.dump(per, open(f"{SCR}/slice_{name}.json", "w"))
    summary["slices"][name] = {"layers": {m: per[m]["layer"] for m in b.ORDER},
                               "n_features": {m: per[m]["n_features"] for m in b.ORDER},
                               "n_concepts": {m: len(sets[m]) for m in b.ORDER},
                               "spectrum_exact_k": spec, "tiers": b.tiers(cm),
                               "detail_equals_exact": True, "seconds": round(time.time() - t0, 1)}
    print(name, summary["slices"][name]["tiers"], f"{time.time()-t0:.0f}s", flush=True)

print("==>", b.dump(summary, "s1_slices_summary.json"))
