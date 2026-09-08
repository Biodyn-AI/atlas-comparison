#!/usr/bin/env python
"""Per-cell-type decodability across models: which cell types are HARDEST to read out linearly, averaged
over all models. Uses the per-cell residual embeddings cell_cka already saved (deepest layer, where the
nonlinearity panel shows concepts are most linearly accessible) + a stratified-CV linear probe; reports
per-class recall per model and the cross-model mean. Tiny output.
    python celltype_difficulty.py --out out_alllayers --corpus external/perturb/tabula_3tissue_6k.h5ad
-> out_alllayers/celltype_difficulty.json  {classes, per_model:{m:{class:recall}}, mean:{class:recall}}
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
from collections import Counter, defaultdict

MODELS = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
CT_CANDIDATES = ["cell_type", "cell_ontology_class", "celltype", "free_annotation"]


def labels(corpus, max_cells=2000, min_per_class=25):
    import scanpy as sc
    ad = sc.read_h5ad(corpus)
    idx = (np.sort(np.random.default_rng(0).choice(ad.n_obs, max_cells, replace=False))
           if max_cells and ad.n_obs > max_cells else np.arange(ad.n_obs))
    ct = next((c for c in CT_CANDIDATES if c in ad.obs.columns), None)
    assert ct, f"no cell-type column in {list(ad.obs.columns)}"
    y = ad.obs[ct].astype(str).values[idx]
    vc = Counter(y); keep = {c for c, n in vc.items() if n >= min_per_class}
    mask = np.array([v in keep for v in y])
    return y[mask], mask, sorted(keep)


def per_class_recall(X, y, seed=0):
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    le = LabelEncoder(); yi = le.fit_transform(y)
    Xs = StandardScaler().fit_transform(X)
    pred = np.zeros_like(yi)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xs, yi):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xs[tr], yi[tr]); pred[te] = clf.predict(Xs[te])
    rec = {}
    for ci, cname in enumerate(le.classes_):
        m = yi == ci
        rec[cname] = round(float((pred[m] == ci).mean()), 3)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out_alllayers")
    ap.add_argument("--corpus", default="external/perturb/tabula_3tissue_6k.h5ad")
    ap.add_argument("--layer", default="last", help="'last', 'mid', or an integer layer index into the npz")
    args = ap.parse_args()
    y, mask, classes = labels(args.corpus)
    print(f"cell types (>=25 cells): {len(classes)} | {classes}", flush=True)
    per_model = {}
    for m in MODELS:
        p = os.path.join(args.out, "cka", f"{m}_emb.npz")
        if not os.path.exists(p):
            print(f"  {m}: no emb npz — skip", flush=True); continue
        z = np.load(p)
        rkeys = sorted((k for k in z.files if k.startswith("res_")), key=lambda k: int(k.split("_")[1]))
        if not rkeys:
            print(f"  {m}: no res_ keys", flush=True); continue
        pick = rkeys[-1] if args.layer == "last" else (rkeys[len(rkeys) // 2] if args.layer == "mid"
                                                        else f"res_{int(args.layer)}")
        X = z[pick][mask]
        per_model[m] = per_class_recall(X, y)
        print(f"  {m} ({pick}): mean recall {np.mean(list(per_model[m].values())):.3f}", flush=True)
    mean = {c: round(float(np.mean([per_model[m][c] for m in per_model if c in per_model[m]])), 3)
            for c in classes}
    out = {"classes": classes, "per_model": per_model, "mean": mean,
           "hardest": sorted(mean, key=mean.get)[:8], "easiest": sorted(mean, key=mean.get)[-8:]}
    json.dump(out, open(f"{args.out}/celltype_difficulty.json", "w"))
    print(f"==> {args.out}/celltype_difficulty.json | hardest: {out['hardest']}", flush=True)


if __name__ == "__main__":
    main()
