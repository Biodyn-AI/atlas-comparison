#!/usr/bin/env python
"""Build the SHARED cell corpus every model is extracted on (fair comparison). Default
target ~2000 cells with broad gene coverage. Point --input at a Tabula Sapiens (immune+
kidney+lung) or CELLxGENE h5ad for the real run; falls back to scanpy pbmc3k for smoke.

    python data/prepare_corpus.py --input tabula_sapiens.h5ad --n-cells 2000 --out data/corpus.h5ad
    python data/prepare_corpus.py --pbmc --out data/corpus.h5ad     # smoke corpus
"""
from __future__ import annotations
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None)
    ap.add_argument("--pbmc", action="store_true")
    ap.add_argument("--n-cells", type=int, default=2000)
    ap.add_argument("--out", default="data/corpus.h5ad")
    args = ap.parse_args()
    import scanpy as sc, numpy as np

    if args.pbmc or not args.input:
        adata = sc.datasets.pbmc3k()
        print("using pbmc3k (smoke corpus)")
    else:
        adata = sc.read_h5ad(args.input)
    # stratified subsample if a cell-type column exists, else random
    if adata.n_obs > args.n_cells:
        rng = np.random.default_rng(0)
        ctcol = next((c for c in ("cell_type", "celltype", "cell_ontology_class") if c in adata.obs), None)
        if ctcol:
            idx = (adata.obs.groupby(ctcol, observed=True).apply(
                lambda g: g.sample(min(len(g), max(1, args.n_cells // adata.obs[ctcol].nunique())),
                                   random_state=0)).index)
            adata = adata[[adata.obs_names.get_loc(i[-1] if isinstance(i, tuple) else i) for i in idx]].copy()
        else:
            adata = adata[rng.choice(adata.n_obs, args.n_cells, replace=False)].copy()
    adata.var_names_make_unique()
    adata.write_h5ad(args.out)
    print(f"wrote {args.out}: {adata.n_obs} cells x {adata.n_vars} genes")


if __name__ == "__main__":
    main()
