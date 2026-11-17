#!/usr/bin/env python
"""Step A: rebuild the 6,000-cell Tabula Sapiens corpus with the SAME logic as
pipeline/atlas_h100/build_ts_corpus.py (seed 0, --per-tissue 2000, is_primary_data, raw counts,
kidney / lung by tissue_general, immune by the 18 immune cell types across all TS tissues),
but without loading every matching cell into memory first.

How it stays equal to the original logic:
  * The original calls get_anndata(filter) for each compartment, then
    keep = rng.choice(a.n_obs, per_tissue, replace=False); a = a[keep].
    The three pulls share one rng (default_rng(seed)), in the order kidney, lung, immune,
    and a final rng.permutation(n) shuffles the concatenated cells.
  * Here we read only the obs table for the same filter (same row order as get_anndata, which
    returns rows in soma_joinid order; checked below), draw keep with the same rng calls in the
    same order, then fetch X only for the kept soma_joinids and put the rows in keep order.
  * Extra obs columns (dataset_id, donor_id, ...) are requested; that does not change the rows.

Run with the census venv:
  census_venv/bin/python build_corpus_lowmem.py --out <scratch>/corpus_seed0.h5ad
Writes a JSON summary next to this script (corpus_summary.json).
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np

IMMUNE = [   # copied verbatim from build_ts_corpus.py
    "T cell", "CD4-positive, alpha-beta T cell", "CD8-positive, alpha-beta T cell",
    "B cell", "plasma cell", "natural killer cell", "macrophage", "monocyte",
    "classical monocyte", "non-classical monocyte", "dendritic cell",
    "conventional dendritic cell", "plasmacytoid dendritic cell", "mast cell",
    "neutrophil", "regulatory T cell", "memory B cell", "naive B cell",
]
OBS_COLS_ORIG = ["cell_type", "tissue", "tissue_general", "assay"]
OBS_COLS_EXTRA = ["dataset_id", "donor_id", "suspension_type", "raw_sum", "nnz",
                  "is_primary_data", "tissue_type", "cell_type_ontology_term_id"]
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-tissue", type=int, default=2000)
    ap.add_argument("--census-version", default="stable")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import cellxgene_census
    import anndata as ad
    import pandas as pd
    from scipy import sparse
    import tiledbsoma as soma

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    desc = cellxgene_census.get_census_version_description(args.census_version)
    summary = {"census_version_requested": args.census_version,
               "census_release_build": desc.get("release_build"),
               "census_soma_uri": desc.get("soma", {}).get("uri"),
               "cellxgene_census_pkg": cellxgene_census.__version__,
               "tiledbsoma_pkg": soma.__version__, "seed": args.seed,
               "per_tissue": args.per_tissue, "pulls": {}}
    print(json.dumps(summary, indent=1), flush=True)

    with cellxgene_census.open_soma(census_version=args.census_version) as census:
        ds = census["census_info"]["datasets"].read().concat().to_pandas()
        ts = ds[ds["collection_name"].str.contains("Tabula Sapiens", case=False, na=False)]
        ts_ids = ts["dataset_id"].tolist()
        assert ts_ids, "no Tabula Sapiens datasets found in census"
        summary["n_ts_datasets"] = len(ts_ids)
        summary["ts_collections"] = sorted(ts["collection_name"].unique().tolist())
        summary["ts_datasets"] = ts[["dataset_id", "dataset_title", "dataset_total_cell_count"]] \
            .to_dict(orient="records")
        ids_sql = "[" + ", ".join(f"'{i}'" for i in ts_ids) + "]"
        print(f"Tabula Sapiens datasets: {len(ts_ids)}", flush=True)
        hs = census["census_data"]["homo_sapiens"]
        var = hs.ms["RNA"].var.read(column_names=["soma_joinid", "feature_id", "feature_name"]) \
            .concat().to_pandas()

        parts = []
        for name, extra in [
                ("kidney", "tissue_general == 'kidney'"),
                ("lung", "tissue_general == 'lung'"),
                ("immune", "cell_type in [" + ", ".join(f"'{c}'" for c in IMMUNE) + "]")]:
            filt = f"dataset_id in {ids_sql} and is_primary_data == True and {extra}"
            obs = hs.obs.read(value_filter=filt,
                              column_names=["soma_joinid"] + OBS_COLS_ORIG + OBS_COLS_EXTRA) \
                .concat().to_pandas()
            jid = obs["soma_joinid"].to_numpy()
            ascending = bool(np.all(np.diff(jid) > 0))
            n_all = len(obs)
            if n_all > args.per_tissue:
                keep = rng.choice(n_all, args.per_tissue, replace=False)
            else:
                keep = np.arange(n_all)
            sub = obs.iloc[keep].reset_index(drop=True)
            want = sub["soma_joinid"].to_numpy()
            # fetch X for these cells only
            with hs.axis_query(measurement_name="RNA",
                               obs_query=soma.AxisQuery(coords=(np.sort(want),))) as q:
                a = q.to_anndata(X_name="raw", column_names={"obs": ["soma_joinid"],
                                                              "var": ["soma_joinid"]})
            pos = {j: i for i, j in enumerate(a.obs["soma_joinid"].to_numpy())}
            order = np.array([pos[j] for j in want])
            X = sparse.csr_matrix(a.X)[order]
            assert np.array_equal(a.var["soma_joinid"].to_numpy(), var["soma_joinid"].to_numpy())
            b = ad.AnnData(X=X.astype(np.float32),
                           obs=sub.set_index(sub["soma_joinid"].astype(str).rename("cell_id")),   # FIXED after run 1: index named soma_joinid clashed with the column at write time
                           var=var.set_index(var["soma_joinid"].astype(str)).copy())
            b.obs["compartment"] = name
            summary["pulls"][name] = {"n_matching_cells": int(n_all), "n_kept": int(b.n_obs),
                                      "obs_rows_in_soma_joinid_order": ascending}
            print(f"  {name}: {n_all} matching cells -> kept {b.n_obs}; "
                  f"obs ascending={ascending}; {time.time()-t0:.0f}s", flush=True)
            parts.append(b)
            del a, X, obs

    A = ad.concat(parts, join="inner", merge="same", index_unique="-")
    A.var["ensembl_id"] = [str(e).split(".")[0] for e in A.var["feature_id"]]
    A.var_names = [str(s).upper() for s in A.var["feature_name"]]
    A.var_names_make_unique()
    A.X = sparse.csr_matrix(A.X)
    order = rng.permutation(A.n_obs)
    A = A[order].copy()
    for c in A.obs.columns:
        if A.obs[c].dtype == object:
            A.obs[c] = A.obs[c].astype(str)
    A.write_h5ad(args.out)
    Xd = A.X[:50].toarray()
    summary["out"] = args.out
    summary["n_cells"] = int(A.n_obs); summary["n_genes"] = int(A.n_vars)
    summary["integer_counts_first50"] = bool(np.allclose(Xd, np.round(Xd)))
    summary["wall_s"] = round(time.time() - t0, 1)
    o = A.obs
    summary["compartment_counts"] = o["compartment"].value_counts().to_dict()
    summary["assay_counts"] = o["assay"].value_counts().to_dict()
    summary["assay_by_compartment"] = pd.crosstab(o["compartment"], o["assay"]).to_dict(orient="index")
    summary["dataset_counts"] = o["dataset_id"].value_counts().to_dict()
    summary["donor_counts"] = o["donor_id"].value_counts().to_dict()
    summary["suspension_type_counts"] = o["suspension_type"].value_counts().to_dict()
    summary["tissue_by_compartment"] = {c: o.loc[o["compartment"] == c, "tissue"].value_counts().to_dict()
                                        for c in ["kidney", "lung", "immune"]}
    summary["cell_type_by_compartment"] = {c: o.loc[o["compartment"] == c, "cell_type"].value_counts().to_dict()
                                           for c in ["kidney", "lung", "immune"]}
    summary["median_raw_sum_by_compartment"] = o.groupby("compartment")["raw_sum"].median().to_dict()
    summary["median_nnz_by_compartment"] = o.groupby("compartment")["nnz"].median().to_dict()
    json.dump(summary, open(os.path.join(HERE, "corpus_summary.json"), "w"), indent=1, default=str)
    print(f"==> wrote {args.out}: {A.n_obs} x {A.n_vars}; {summary['wall_s']} s", flush=True)


if __name__ == "__main__":
    main()
