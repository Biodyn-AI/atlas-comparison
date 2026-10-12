#!/usr/bin/env python
"""Fill the controls.json blocks that no other local script writes, from their source files:
  seeds           <- seed_variance.json            (GPU side: seed_variance.py, mid layer, 3 seeds)
  seeds_alllayer  <- seed_variance_alllayer.json   (GPU side: seed_variance_alllayer.py, every layer)
                     + alllayer_null.json real_core (the canonical all-layer core, for comparison)
  heldout         <- heldout.json                  (heldout_compare.py)
  random_null, survivor_nulls (cka_svd_null), stats_final, degree_null <- verbatim copies of those files
  recalibration   <- recalibrate, recalibrate2, recalibrate_final (+ its >=8 null), recalibrate_robust, heldout_calibrated
Spread across seeds is the population SD (ddof 0); CV = SD / mean in percent.
The seed files are copied back from the GPU host into outputs/atlas/seeds/.

    python scripts/controls_import.py -> merges into outputs/atlas/comparative/controls.json
"""
from __future__ import annotations

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import json
import numpy as np

C = f"{_B}/outputs/atlas/comparative"; SEEDS = f"{_B}/outputs/atlas/seeds"


def cv(xs):
    return float(np.std(xs) / np.mean(xs) * 100)


def main():
    ctl = json.load(open(f"{C}/controls.json"))

    sv = json.load(open(f"{SEEDS}/seed_variance.json"))
    rate_sd = {m: round(float(np.std([r["annot_rate"] for r in v])), 2) for m, v in sv["models"].items()}
    ncv = {m: round(cv([r["n_concepts"] for r in v]), 1) for m, v in sv["models"].items()}   # rounded per model, then averaged
    cores = sv["core_by_seed"]
    ctl["seeds"] = {"n_seeds": len(cores), "layer": "mid", "core_by_seed": cores,
                    "core_mean": round(float(np.mean(cores)), 1), "core_sd": round(float(np.std(cores)), 1),
                    "core_cv_pct": round(cv(cores), 1), "rate_sd_pp": rate_sd,
                    "rate_sd_max": round(max(rate_sd.values()), 2), "rate_sd_mean": round(float(np.mean(list(rate_sd.values()))), 2),
                    "nconc_cv_max": round(max(ncv.values()), 1), "nconc_cv_mean": round(float(np.mean(list(ncv.values()))), 1)}

    sa = json.load(open(f"{SEEDS}/seed_variance_alllayer.json"))
    nb = sa["nconc_by_model_seed"]
    models = sorted({k.split("#")[0] for k in nb}); seeds = sorted({int(k.split("#")[1]) for k in nb})
    acores = sa["core_by_seed"]
    ctl["seeds_alllayer"] = {"n_seeds": len(acores), "scope": "all layers, all saved positions, canonical-aligned annotator",
                             "core_by_seed": acores, "core_mean": round(float(np.mean(acores)), 1),
                             "core_sd": round(float(np.std(acores)), 1), "core_cv_pct": round(cv(acores), 1),
                             "canonical_core": float(json.load(open(f"{C}/alllayer_null.json"))["real_core"]),
                             "nconc_cv_mean": round(float(np.mean([cv([nb[f"{m}#{s}"] for s in seeds]) for m in models])), 1)}

    ctl["heldout"] = json.load(open(f"{C}/heldout.json"))

    # verbatim copies of single result files
    for block, f in (("random_null", "random_null.json"), ("survivor_nulls", "cka_svd_null.json"),
                     ("stats_final", "stats_final.json"), ("degree_null", "degree_null.json")):
        ctl[block] = json.load(open(f"{C}/{f}"))

    # recalibration: both annotator grids, the final calibrated annotator with its >=8 null, the robustness
    # grid with its >=8 ranges, and the calibrated held-out comparison
    final = json.load(open(f"{C}/recalibrate_final.json"))
    ge8_null = round(sum(v for k, v in final["null_spectrum_mean"].items() if int(k) >= 8), 1)
    final.update({"shared_ge8_real": final["shared_ge8"], "shared_ge8_null": ge8_null,
                  "ge8_fold": round(final["shared_ge8"] / max(ge8_null, 0.1), 1)})
    robust = json.load(open(f"{C}/recalibrate_robust.json"))
    ge8 = [g[">=8"] for g in robust["grid"]]
    ctl["recalibration"] = {"grid": json.load(open(f"{C}/recalibrate.json"))["configs"] + json.load(open(f"{C}/recalibrate2.json"))["configs"],
                            "final": final, "robust": robust,
                            "ge8_fold_range": [min(g["fold"] for g in ge8), max(g["fold"] for g in ge8)],
                            "ge8_real_range": [min(g["real"] for g in ge8), max(g["real"] for g in ge8)],
                            "heldout_calibrated": json.load(open(f"{C}/heldout_calibrated.json"))}
    json.dump(ctl, open(f"{C}/controls.json", "w"), indent=1)
    print(f"seeds: core {ctl['seeds']['core_by_seed']}  all-layer: {acores}  held-out core Jaccard {ctl['heldout']['core_jaccard']}")
    print(f"==> {C}/controls.json")


if __name__ == "__main__":
    main()
