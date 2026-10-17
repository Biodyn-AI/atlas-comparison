#!/usr/bin/env python
"""Step 4. Random-gene nulls for the new comparisons, on the repo's ruler.

Null (as recalibrate_final.py:79-87 / stats_final.py:90-95): every feature of every model is replaced by TOP genes
drawn uniformly (with replacement) from the background; the calibrated annotator is re-run; concepts are counted
across models. Feature counts per model are those of the real catalogue at the same slice. rng = default_rng(0)
fresh for each configuration, models drawn in ORDER, so the cap-200 mid run must reproduce stats_final.json
(null >=8 mean 3.46, sd 2.04) exactly -- this is checked.
Also records, for every null draw, how many distinct programmes the null >=k set forms when clustered by Jaccard
of the database gene sets (complete linkage, J>=0.3 and J>=0.5), so the backbone can also be compared to its null
in units of programmes rather than concept names.
Empirical p = (#null >= obs + 1) / (N + 1).
Output: s4_nulls.json
"""
import json, sys, time
from collections import Counter

import numpy as np
import bb_common as b

NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 250
SUM = json.load(open(f"{b.OUT}/s1_slices_summary.json"))["slices"]
SEVEN = [m for m in b.ORDER if m not in b.EMBED_L0]


def full_sets(terms):
    return {t: b.RAW[t.split(":", 1)[0]][t.split(":", 1)[1]] for t in terms}


def n_prog(terms, thr):
    fs = full_sets(terms)
    return len(b.cluster_matrix(sorted(terms), lambda a, c: b.jaccard(fs[a], fs[c]), thr, "complete"))


def real_sets(ann, slice_name, models, top=10):
    out = {}
    for m in models:
        L = SUM[slice_name]["layers"][m]
        feats, _ = b.load_cat(b.layer_path(m, L))
        if top != 10:
            a2 = b.Annotator(ann.mx, top=top); out[m] = a2.annotate_exact(feats)
        else:
            out[m] = ann.annotate_exact(feats)
    return out


def run(name, slice_name, models, mx=200, top=10, ks=(7, 8, 9, 10), prog_k=8):
    t0 = time.time()
    ann = b.Annotator(mx, top=top)
    nfeat = {m: SUM[slice_name]["n_features"][m] for m in models}
    rs = real_sets(ann, slice_name, models, top)
    cm = b.count_models(rs)
    real = {k: sum(1 for v in cm.values() if v >= k) for k in ks}
    real_terms = sorted(t for t, v in cm.items() if v >= prog_k)
    real_prog = {thr: n_prog(real_terms, thr) for thr in (0.3, 0.5)}
    rng = np.random.default_rng(0)
    null = {k: [] for k in ks}; nprog = {0.3: [], 0.5: []}
    for it in range(NPERM):
        sets = {m: ann.annotate_fast([b.BG_ARR[rng.integers(0, b.BGN, top)].tolist() for _ in range(nfeat[m])])
                for m in models}
        c = b.count_models(sets)
        for k in ks:
            null[k].append(sum(1 for v in c.values() if v >= k))
        nt = [t for t, v in c.items() if v >= prog_k]
        for thr in (0.3, 0.5):
            nprog[thr].append(n_prog(nt, thr))
    row = {"slice": slice_name, "models": models, "n_models": len(models), "cap": mx, "top": top,
           "n_features": nfeat, "n_perm": NPERM, "seed": 0, "tiers": {}, "programmes_at_k": prog_k}
    for k in ks:
        arr = np.array(null[k], float)
        row["tiers"][f">={k}"] = {"real": real[k], "null_mean": round(float(arr.mean()), 3), "null_sd": round(float(arr.std()), 3),
                                  "null_max": int(arr.max()),
                                  "fold_raw": round(real[k] / arr.mean(), 2) if arr.mean() > 0 else None,
                                  "fold_floor0.1": round(real[k] / max(arr.mean(), 0.1), 1),
                                  "z": round(float((real[k] - arr.mean()) / (arr.std() or 1)), 1),
                                  "p_emp": round((int((arr >= real[k]).sum()) + 1) / (NPERM + 1), 5)}
    for thr in (0.3, 0.5):
        arr = np.array(nprog[thr], float)
        row[f"programmes_fullset_complete_J>={thr}"] = {
            "real": real_prog[thr], "null_mean": round(float(arr.mean()), 3), "null_sd": round(float(arr.std()), 3),
            "fold_raw": round(real_prog[thr] / arr.mean(), 2) if arr.mean() > 0 else None,
            "p_emp": round((int((arr >= real_prog[thr]).sum()) + 1) / (NPERM + 1), 5)}
    row["seconds"] = round(time.time() - t0, 1)
    print(name, {k: (v["real"], v["null_mean"], v["null_sd"], v["p_emp"]) for k, v in row["tiers"].items()},
          "prog0.3", row["programmes_fullset_complete_J>=0.3"]["real"], row["programmes_fullset_complete_J>=0.3"]["null_mean"],
          f"{row['seconds']}s", flush=True)
    return row


out = {"null": "uniform random background genes per feature (recalibrate_final.py:80), default_rng(0) per config",
       "n_perm": NPERM, "configs": {}}
out["configs"]["mid_cap200"] = run("mid_cap200", "mid", b.ORDER, 200)
sf = json.load(open(f"{b.ORIG_COMPARATIVE}/stats_final.json"))["backbone_null"]
out["check_vs_stats_final"] = {k: {"stats_final": [sf[k]["null_mean"], sf[k]["null_sd"]],
                                   "here": [round(out["configs"]["mid_cap200"]["tiers"][k]["null_mean"], 2),
                                            round(out["configs"]["mid_cap200"]["tiers"][k]["null_sd"], 2)]}
                               for k in sf} if NPERM == 250 else "only comparable at 250 permutations"
print("check vs stats_final:", out["check_vs_stats_final"], flush=True)
out["configs"]["mid_cap150"] = run("mid_cap150", "mid", b.ORDER, 150)
out["configs"]["mid_cap300"] = run("mid_cap300", "mid", b.ORDER, 300)
out["configs"]["mid_cap200_top12"] = run("mid_cap200_top12", "mid", b.ORDER, 200, top=12)
out["configs"]["d0_block1_cap200"] = run("d0_block1_cap200", "d0_block1", b.ORDER, 200)
for sl in ("d0", "d50", "mid", "d100"):
    out["configs"][f"{sl}_7models_cap200"] = run(f"{sl}_7models_cap200", sl, SEVEN, 200, ks=(5, 6, 7), prog_k=6)
print("==>", b.dump(out, "s4_nulls.json"))
