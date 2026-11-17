#!/usr/bin/env python
"""(h) Near-input excess without MaxToki: among the 6 block-output models other than MaxToki (C2S, Geneformer, UCE,
scGPT, scFoundation, Tahoe), count concepts in >=5 of 6 and 6 of 6 at each depth slice, with weighted classes.
Same caches / annotator / class rule as notes 13 (s2_mid_composition.run_slice). No null (observed counts only).
Output: r3_near_input_no_maxtoki.json
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
BC = os.path.join(os.path.dirname(HERE), "backbone_composition")
sys.path.insert(0, BC)
import bb_common as b                                   # noqa: E402
from s2_mid_composition import load_slice, run_slice   # noqa: E402
SIX = [m for m in b.ORDER if m not in b.EMBED_L0 + ["MaxToki"]]
out = {"models": SIX, "slices": {}}
for nm in ("d0", "d25", "d50", "mid", "d75", "d100"):
    pr = load_slice(nm); row = {}
    for k in (5, 6):
        res, _, _, _ = run_slice(nm, pr, models=SIX, k=k)
        row[f">={k}_of_6"] = {"n": res["n_concepts"], "class_weighted": res["class_counts"]["class_weighted"]}
    out["slices"][nm] = row
    print(nm, {k: v["n"] for k, v in row.items()}, flush=True)
json.dump(out, open(f"{HERE}/r3_near_input_no_maxtoki.json", "w"), indent=1)
