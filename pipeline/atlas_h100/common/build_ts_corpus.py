#!/usr/bin/env python
"""Build the shared 3-tissue Tabula Sapiens corpus for the atlas re-run (Igor's spec:
immune + kidney + lung, ~1000 cells each). Pulls ONLY the needed cells from CZ CELLxGENE
Census (no giant figshare downloads), restricted to Tabula Sapiens datasets, with raw counts
and both gene symbols (var_names) and Ensembl ids (var['ensembl_id']) so every adapter works
(symbol-based: AIDO/scGPT/tGPT/UCE; Ensembl-based: Geneformer/MaxToki/Tahoe). Cells shuffled so
the 4M-position cap still sees all cell types.

Run on the cluster (census needs Linux/tiledbsoma):
    pip install cellxgene-census
    python build_ts_corpus.py --out external/perturb/tabula_3tissue_3k.h5ad --per-tissue 1000

VERIFY: census obs field names ('tissue_general', 'cell_type'); the immune filter (below) samples
immune-lineage cell types across TS tissues — adjust the IMMUNE list if Igor's "immune" subset differs.
"""
from __future__ import annotations
import argparse
import numpy as np

IMMUNE = [
    "T cell", "CD4-positive, alpha-beta T cell", "CD8-positive, alpha-beta T cell",
    "B cell", "plasma cell", "natural killer cell", "macrophage", "monocyte",
    "classical monocyte", "non-classical monocyte", "dendritic cell",
    "conventional dendritic cell", "plasmacytoid dendritic cell", "mast cell",
    "neutrophil", "regulatory T cell", "memory B cell", "naive B cell",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="external/perturb/tabula_3tissue_3k.h5ad")
    ap.add_argument("--per-tissue", type=int, default=1000)
    ap.add_argument("--census-version", default="stable")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import cellxgene_census
    import anndata as ad
    import pandas as pd
    from scipy import sparse

    rng = np.random.default_rng(args.seed)
    with cellxgene_census.open_soma(census_version=args.census_version) as census:
        # Tabula Sapiens dataset ids
        ds = census["census_info"]["datasets"].read().concat().to_pandas()
        ts = ds[ds["collection_name"].str.contains("Tabula Sapiens", case=False, na=False)]
        ts_ids = ts["dataset_id"].tolist()
        assert ts_ids, "no Tabula Sapiens datasets found in census"
        ids_sql = "[" + ", ".join(f"'{i}'" for i in ts_ids) + "]"
        print(f"Tabula Sapiens datasets: {len(ts_ids)}", flush=True)

        def pull(name, extra_filter):
            filt = (f"dataset_id in {ids_sql} and is_primary_data == True and {extra_filter}")
            a = cellxgene_census.get_anndata(
                census, organism="Homo sapiens", X_name="raw",
                obs_value_filter=filt,
                obs_column_names=["cell_type", "tissue", "tissue_general", "assay"],
                var_column_names=["feature_id", "feature_name"])
            if a.n_obs > args.per_tissue:
                keep = rng.choice(a.n_obs, args.per_tissue, replace=False)
                a = a[keep].copy()
            a.obs["compartment"] = name
            print(f"  {name}: {a.n_obs} cells", flush=True)
            return a

        parts = [
            pull("kidney", "tissue_general == 'kidney'"),
            pull("lung", "tissue_general == 'lung'"),
            pull("immune", "cell_type in [" + ", ".join(f"'{c}'" for c in IMMUNE) + "]"),
        ]

    A = ad.concat(parts, join="inner", index_unique="-")
    # var: symbols as names, Ensembl kept; X raw counts
    A.var["ensembl_id"] = A.var["feature_id"].astype(str).values
    A.var_names = [str(s).upper() for s in A.var["feature_name"]]
    A.var_names_make_unique()
    if not sparse.issparse(A.X):
        A.X = sparse.csr_matrix(A.X)
    order = rng.permutation(A.n_obs)          # shuffle so position-capped models see all types
    A = A[order].copy()
    A.write_h5ad(args.out)
    print(f"==> wrote {args.out}: {A.n_obs} cells x {A.n_vars} genes")
    print("   tissues:", dict(A.obs['compartment'].value_counts()))
    print("   integer counts:", bool(np.allclose(A.X[:50].toarray(), np.round(A.X[:50].toarray()))))


if __name__ == "__main__":
    main()
