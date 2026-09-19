#!/usr/bin/env python
"""Do models agree on which INDIVIDUAL cells are hard? For each model, a stratified-CV linear cell-type
probe on the deepest-layer per-cell embedding (cell_cka --all-layers emb); a cell is 'hard' for that model
if the probe misclassifies it. Then across models: per-cell hardness = how many models miss it, and the
pairwise correlation of the per-cell error vectors (do the SAME cells resist every architecture?).
    python hardcell_agreement.py --out out_alllayers --corpus external/perturb/tabula_3tissue_6k.h5ad
-> out_alllayers/hardcell_agreement.json
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from collections import Counter

MODELS = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
CT = ["cell_type", "cell_ontology_class", "celltype", "free_annotation"]


def labels(corpus, max_cells=2000, min_per_class=25):
    import scanpy as sc
    ad = sc.read_h5ad(corpus)
    idx = (np.sort(np.random.default_rng(0).choice(ad.n_obs, max_cells, replace=False))
           if max_cells and ad.n_obs > max_cells else np.arange(ad.n_obs))
    c = next((x for x in CT if x in ad.obs.columns), None); assert c
    y = ad.obs[c].astype(str).values[idx]
    vc = Counter(y); keep = {k for k, v in vc.items() if v >= min_per_class}
    mask = np.array([v in keep for v in y])
    return y[mask], mask


def cv_err(X, y, seed=0):
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    yi = LabelEncoder().fit_transform(y); Xs = StandardScaler().fit_transform(X); pred = np.zeros_like(yi)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xs, yi):
        clf = LogisticRegression(max_iter=2000, C=1.0); clf.fit(Xs[tr], yi[tr]); pred[te] = clf.predict(Xs[te])
    return (pred != yi).astype(np.int8)                     # 1 = misclassified (hard)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out_alllayers")
    ap.add_argument("--corpus", default="external/perturb/tabula_3tissue_6k.h5ad"); args = ap.parse_args()
    y, mask = labels(args.corpus)
    errs, used = [], []
    for m in MODELS:
        p = os.path.join(args.out, "cka", f"{m}_emb.npz")
        if not os.path.exists(p):
            print(f"  {m}: no emb, skip", flush=True); continue
        z = np.load(p); rk = sorted((k for k in z.files if k.startswith("res_")), key=lambda k: int(k.split("_")[1]))
        if not rk:
            continue
        e = cv_err(z[rk[-1]][mask], y); errs.append(e); used.append(m)
        print(f"  {m}: error rate {e.mean():.3f}", flush=True)
    E = np.array(errs)                                       # [n_models, n_cells]
    hard = E.sum(0)                                          # per-cell: how many models miss it
    N = len(used)
    hist = {str(k): int((hard == k).sum()) for k in range(N + 1)}
    # pairwise error correlation (phi coefficient on boolean error vectors)
    corrs = []
    for i in range(N):
        for j in range(i + 1, N):
            a, b = E[i], E[j]
            if a.std() > 0 and b.std() > 0:
                corrs.append(float(np.corrcoef(a, b)[0, 1]))
    # cell types of the consistently-hard cells (missed by >= 60% of models)
    thr = max(2, int(0.6 * N))
    hardmask = hard >= thr
    hard_types = Counter(y[hardmask]).most_common(10)
    easy_types = Counter(y[hard == 0]).most_common(6)
    out = {"models": used, "n_models": N, "n_cells": int(len(y)),
           "hardness_hist": hist, "mean_pairwise_err_corr": round(float(np.mean(corrs)), 3) if corrs else None,
           "hard_thresh": thr, "n_consistently_hard": int(hardmask.sum()),
           "hard_celltypes": [{"ct": c, "n": n} for c, n in hard_types],
           "always_easy_celltypes": [{"ct": c, "n": n} for c, n in easy_types]}
    json.dump(out, open(f"{args.out}/hardcell_agreement.json", "w"))
    print(f"==> hardcell_agreement.json | mean pairwise err-corr {out['mean_pairwise_err_corr']} | "
          f"{out['n_consistently_hard']} cells hard for >={thr}/{N} models | top hard: {hard_types[:3]}", flush=True)


if __name__ == "__main__":
    main()
