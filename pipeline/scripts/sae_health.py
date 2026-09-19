#!/usr/bin/env python
"""Dictionary health for every model-layer, not just the mid layer.

The manuscript certified SAE health at one layer while Section 3.4 pools features over all of them,
so the quality control did not cover the analysis that depends on it. This records the dead-feature
fraction per model across every layer, which is what that section actually consumes, and writes it
where the manuscript audit can read it back.
    python scripts/sae_health.py -> outputs/atlas/comparative/sae_health.json
"""
from __future__ import annotations

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import glob, json, re
import numpy as np

BASE = _B
C = f"{BASE}/outputs/atlas/comparative"
ALLCAT, TS = f"{BASE}/outputs/atlas/alllayer_cat", f"{BASE}/outputs/atlas/ts3_out"
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
THRESH = 0.10

out = {"threshold": THRESH, "note": "dead_frac per layer, read from each layer's feature catalogue", "models": {}}
tot = bad = 0
print(f"{'model':<14}{'layers':>7}{'mid':>8}{'mean':>8}{'max':>8}{'>10% dead':>12}")
for m in ORDER:
    d = ALLCAT if glob.glob(f"{ALLCAT}/{m}/feature_catalog_L*.json") else TS
    paths = sorted(glob.glob(f"{d}/{m}/feature_catalog_L*.json"),
                   key=lambda p: int(re.search(r"_L(\d+)\.json$", p).group(1)))
    v = np.array([json.load(open(p)).get("dead_frac", 0.0) for p in paths], float)
    n_bad = int((v > THRESH).sum()); tot += len(v); bad += n_bad
    out["models"][m] = {"n_layers": len(v), "mid": round(float(v[len(v) // 2]) * 100, 1),
                        "mean": round(float(v.mean()) * 100, 1), "max": round(float(v.max()) * 100, 1),
                        "layers_over_threshold": n_bad}
    print(f"{m:<14}{len(v):>7}{out['models'][m]['mid']:>7.1f}%{out['models'][m]['mean']:>7.1f}%"
          f"{out['models'][m]['max']:>7.1f}%{f'{n_bad}/{len(v)}':>12}")
out["total_layers"] = tot
out["layers_over_threshold"] = bad
out["pct_layers_over_threshold"] = round(100 * bad / tot, 1)
out["clean_models"] = [m for m, d_ in out["models"].items() if d_["max"] == 0.0]
json.dump(out, open(f"{C}/sae_health.json", "w"), indent=1)
print(f"\n{bad} of {tot} model-layers ({out['pct_layers_over_threshold']} %) exceed {int(THRESH*100)} % dead features")
print(f"models with no dead features at any layer: {len(out['clean_models'])}")
print("==> sae_health.json")
