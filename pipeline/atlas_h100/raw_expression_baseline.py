#!/usr/bin/env python
"""Raw-expression baseline for the cell-level geometry results (CKA, tissue and cell-type decoding).

Uses the 2,000 cells cell_cka.py and nonlinearity.py use (seed-0 subsample of the 6,000-cell corpus)
and asks what the expression matrix alone gives, with no model:
  - decoding: the nonlinearity.py probes (5-fold stratified CV, StandardScaler, logistic / MLP,
    balanced accuracy) on PCA-50 of log1p CP10k over 2,000 highly variable genes, and on the
    2,000 HVGs themselves;
  - CKA: linear CKA between the HVG matrix and every layer of every model's residual stream,
    with a cell-shuffle floor (rows of the expression matrix permuted, 20 draws).

    python raw_expression_baseline.py --out out_alllayers
        -> out_alllayers/baselines/raw_expression_baseline.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from cell_cka import linear_cka
from common.layers import depth_matched
from nonlinearity import concepts_for, cv_probe

MODELS = ["AIDO", "scGPT", "tGPT", "scFoundation", "GeneCompass", "MaxToki",
          "Geneformer", "UCE", "C2S", "Tahoe"]


def raw_matrices(corpus, max_cells, n_hvg=2000, n_pc=50):
    import scanpy as sc
    ad = sc.read_h5ad(corpus)
    idx = np.sort(np.random.default_rng(0).choice(ad.n_obs, max_cells, replace=False))
    ad = ad[idx].copy()
    sc.pp.filter_genes(ad, min_cells=10)
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sc.pp.highly_variable_genes(ad, n_top_genes=n_hvg, flavor="seurat")
    X = ad[:, ad.var["highly_variable"]].X
    X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float64)
    Xc = X - X.mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    pcs = U[:, :n_pc] * S[:n_pc]
    return X, pcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="external/perturb/tabula_3tissue_6k.h5ad")
    ap.add_argument("--out", default="out_alllayers")
    ap.add_argument("--max-cells", type=int, default=2000)
    ap.add_argument("--n-shuffle", type=int, default=20)
    args = ap.parse_args()

    X, pcs = raw_matrices(args.corpus, args.max_cells)
    print(f"raw: HVG {X.shape}, PCA {pcs.shape}", flush=True)
    res = {"cells": int(X.shape[0]), "hvg": int(X.shape[1]), "pcs": int(pcs.shape[1]),
           "decoding": {}, "cka": {}}

    for cname, (y, mask, classes) in concepts_for(args.corpus, args.max_cells, "compartment").items():
        rec = {"n_classes": len(classes), "n_cells": int(mask.sum()), "chance": round(1 / len(classes), 3)}
        for rep, M in (("pca50", pcs), ("hvg2000", X)):
            lin = cv_probe(M[mask], y, "lin"); mlp = cv_probe(M[mask], y, "mlp")
            rec[rep] = {"lin": round(lin[0], 3), "lin_sd": round(lin[1], 3),
                        "mlp": round(mlp[0], 3), "mlp_sd": round(mlp[1], 3)}
        res["decoding"][cname] = rec
        print(f"[{cname}] {rec}", flush=True)

    rng = np.random.default_rng(0)
    perms = [rng.permutation(X.shape[0]) for _ in range(args.n_shuffle)]
    for m in MODELS:
        p = os.path.join(args.out, "cka", f"{m}_emb.npz")
        if not os.path.exists(p):
            print(f"{m}: no {p}, skip", flush=True); continue
        e = np.load(p)
        layers = [int(k.split("_")[1]) for k in e.files if k.startswith("res_")]
        layers.sort()
        mid = depth_matched(len(layers))[2]
        per = {}
        for L in layers:
            R = e[f"res_{L}"].astype(np.float64)
            per[str(L)] = round(linear_cka(X, R), 3)
        Rm = e[f"res_{mid}"].astype(np.float64)
        null = [linear_cka(X[pi], Rm) for pi in perms]
        res["cka"][m] = {"n_layers": len(layers), "mid_layer": mid, "per_layer": per,
                         "mid": per[str(mid)], "max": max(per.values()),
                         "shuffle_mean": round(float(np.mean(null)), 4),
                         "shuffle_max": round(float(np.max(null)), 4)}
        print(f"{m}: CKA(raw, mid L{mid}) = {per[str(mid)]}, max {max(per.values())}, "
              f"shuffle {np.mean(null):.4f}", flush=True)

    od = os.path.join(args.out, "baselines"); os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, "raw_expression_baseline.json")
    json.dump(res, open(fp, "w"), indent=1)
    print(f"==> {fp}", flush=True)


if __name__ == "__main__":
    main()
