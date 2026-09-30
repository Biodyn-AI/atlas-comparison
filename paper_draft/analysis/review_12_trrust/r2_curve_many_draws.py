#!/usr/bin/env python
"""Review of note 12, check R2: how large is the >= 2-model fold once the null is estimated well?

Note 12 found 14.06x (all ten) and 17.53x (five passing) on one 100-draw stream, against the
published 20.8x (hypothesis_trrust2.json, 100 draws) and 28.43x (hypothesis_final.json, 200 draws),
and called the fold "noisy". Here the SAME null (repo stub-matching config_shuffle, each model's
pooled W>=5 graph rewired independently per draw; per-draw null precision = hits / pairs at >= k
models, as hypothesis_trrust2.py:124-146) is run for NDRAW draws. Reported:
  - the null precision mean over all draws, its Monte Carlo standard error, and the fold;
  - the spread of the fold across independent blocks of 100 draws (what a 100-draw run gives)
    and of 200 draws (what hypothesis_final's run gives);
  - a null-free number: observed precision at >= 2 models / observed precision at >= 1 model.
Graphs from note 12's common.py (identical to the repo's cached graphs). Seed 2024.
Output: r2_curve_many_draws.json (this folder)
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "trrust_robustness"))
from common import *  # noqa

NDRAW = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
genes = load_genes(); gid = {g: i for i, g in enumerate(genes)}
truth, _, _ = load_truth()
tkey = truth_keys(truth, gid)
G5 = {}
for m in ORDER:
    feat, lay, L = load_model(m)
    G5[m] = graph_from(feat, lay, None, 5)[0]
    del feat, lay

out = {"n_draws": NDRAW, "seed": 2024, "null": "repo stub-matching config_shuffle per model, per draw", "sets": {}}
t0 = time.time()
for label, models in (("all_ten", ORDER), ("five_passing", PASSED)):
    allk = np.concatenate([G5[m] for m in models if len(G5[m])])
    kk, vv = np.unique(allk, return_counts=True)
    ist = np.isin(kk, tkey)
    obs = {k: (int((vv >= k).sum()), int(ist[vv >= k].sum())) for k in (1, 2)}
    rng = np.random.default_rng(2024)
    prec = {1: [], 2: []}; nh = {1: [], 2: []}; npair = {1: [], 2: []}
    for d in range(NDRAW):
        allr = np.concatenate([config_shuffle_keys(G5[m], rng) for m in models if len(G5[m])])
        k3, v3 = np.unique(allr, return_counts=True)
        i3 = np.isin(k3, tkey)
        for k in (1, 2):
            sel = v3 >= k
            s = int(sel.sum()); h = int(i3[sel].sum())
            prec[k].append(h / max(s, 1)); nh[k].append(h); npair[k].append(s)
        if (d + 1) % 500 == 0:
            print(f"  {label} {d+1}/{NDRAW} [{time.time()-t0:.0f}s]", flush=True)
    res = {}
    for k in (1, 2):
        pc = np.array(prec[k]); n, h = obs[k]; po = h / n
        se = pc.std(ddof=1) / np.sqrt(len(pc))
        blocks = {}
        for B in (100, 200):
            nb = len(pc) // B
            f = [po / pc[i * B:(i + 1) * B].mean() if pc[i * B:(i + 1) * B].mean() > 0 else None for i in range(nb)]
            fv = np.array([x for x in f if x is not None])
            blocks[f"blocks_of_{B}"] = {"n_blocks": nb, "n_blocks_null_zero": int(sum(x is None for x in f)),
                                         "fold_min": round(float(fv.min()), 2), "fold_median": round(float(np.median(fv)), 2),
                                         "fold_max": round(float(fv.max()), 2),
                                         "fold_p05": round(float(np.percentile(fv, 5)), 2), "fold_p95": round(float(np.percentile(fv, 95)), 2)}
        res[k] = {"n_pairs": n, "hits": h, "precision": round(po, 6),
                  "null_precision_mean": round(float(pc.mean()), 8), "null_precision_se": round(float(se), 9),
                  "null_hits_mean": round(float(np.mean(nh[k])), 4), "null_pairs_mean": round(float(np.mean(npair[k])), 1),
                  "null_draws_with_zero_hits": int((np.array(nh[k]) == 0).sum()),
                  "fold": round(po / pc.mean(), 2),
                  "fold_approx_95ci_from_se": [round(po / (pc.mean() + 1.96 * se), 2), round(po / max(pc.mean() - 1.96 * se, 1e-12), 2)],
                  "p_emp": round((int((pc >= po).sum()) + 1) / (len(pc) + 1), 5), **blocks}
    res["null_free_precision_ratio_k2_over_k1"] = round(res[2]["precision"] / res[1]["precision"], 3)
    out["sets"][label] = res
    print(label, json.dumps(res), flush=True)
json.dump(out, open(f"{HERE}/r2_curve_many_draws.json", "w"), indent=1)
print("done", time.time() - t0)
