#!/usr/bin/env python
"""Recovery step for build_corpus_lowmem.py.

The first run pulled all 6,000 cells and wrote X (6000 x 61,497 raw counts) to the h5ad, then
crashed while writing obs (the obs index was named 'soma_joinid', which clashed with the
'soma_joinid' column after ad.concat added '-0/-1/-2' suffixes). X is complete on disk.

This script rebuilds obs and var without re-downloading X:
  * re-reads the obs table for the same three filters (same census release, same row order),
  * replays the same rng calls (default_rng(0): choice for kidney, lung, immune, then permutation),
  * re-reads var (all genes, soma_joinid order; build_ts_corpus.py keeps all of them),
  * CHECKS the reconstruction: every cell's X row sum must equal census obs 'raw_sum' and its
    number of non-zero genes must equal census obs 'nnz'.
Then writes <out>.h5ad with the recovered X, obs and var and updates corpus_summary.json.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_corpus_lowmem import IMMUNE, OBS_COLS_ORIG, OBS_COLS_EXTRA, HERE   # same constants

PARTIAL, OUT = sys.argv[1], sys.argv[2]
RELEASE = "2025-11-08"      # what 'stable' resolved to in the first run (corpus_summary.json)
SEED, PER = 0, 2000

import cellxgene_census, h5py
import anndata as ad
import pandas as pd
from scipy import sparse

t0 = time.time()
with h5py.File(PARTIAL, "r") as f:
    g = f["X"]; shape = tuple(g.attrs["shape"])
    X = sparse.csr_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape)
print("X", X.shape, X.nnz, flush=True)

rng = np.random.default_rng(SEED)
summary = json.load(open(os.path.join(HERE, "corpus_summary.json"))) if os.path.exists(
    os.path.join(HERE, "corpus_summary.json")) else {}
summary.update({"census_version_requested": "stable", "census_release_build": RELEASE,
                "cellxgene_census_pkg": cellxgene_census.__version__, "seed": SEED, "per_tissue": PER,
                "recovery": "X from first run; obs/var rebuilt by fix_corpus_obs.py", "pulls": {}})
with cellxgene_census.open_soma(census_version=RELEASE) as census:
    ds = census["census_info"]["datasets"].read().concat().to_pandas()
    ts = ds[ds["collection_name"].str.contains("Tabula Sapiens", case=False, na=False)]
    ts_ids = ts["dataset_id"].tolist()
    summary["n_ts_datasets"] = len(ts_ids)
    summary["ts_collections"] = sorted(ts["collection_name"].unique().tolist())
    summary["ts_datasets"] = ts[["dataset_id", "collection_name", "dataset_title", "dataset_total_cell_count"]].to_dict(orient="records")
    ids_sql = "[" + ", ".join(f"'{i}'" for i in ts_ids) + "]"
    hs = census["census_data"]["homo_sapiens"]
    var = hs.ms["RNA"].var.read(column_names=["soma_joinid", "feature_id", "feature_name"]).concat().to_pandas()
    parts = []
    for name, extra in [("kidney", "tissue_general == 'kidney'"), ("lung", "tissue_general == 'lung'"),
                        ("immune", "cell_type in [" + ", ".join(f"'{c}'" for c in IMMUNE) + "]")]:
        filt = f"dataset_id in {ids_sql} and is_primary_data == True and {extra}"
        obs = hs.obs.read(value_filter=filt, column_names=["soma_joinid"] + OBS_COLS_ORIG + OBS_COLS_EXTRA).concat().to_pandas()
        asc = bool(np.all(np.diff(obs["soma_joinid"].to_numpy()) > 0))
        n_all = len(obs)
        keep = rng.choice(n_all, PER, replace=False) if n_all > PER else np.arange(n_all)
        sub = obs.iloc[keep].reset_index(drop=True); sub["compartment"] = name
        summary["pulls"][name] = {"n_matching_cells": int(n_all), "n_kept": int(len(sub)),
                                  "obs_rows_in_soma_joinid_order": asc}
        print(f"  {name}: {n_all} matching, kept {len(sub)}, ascending {asc}, {time.time()-t0:.0f}s", flush=True)
        parts.append(sub); del obs

O = pd.concat(parts, ignore_index=True)
order = rng.permutation(len(O))
O = O.iloc[order].reset_index(drop=True)
rs = np.asarray(X.sum(1)).ravel(); nz = np.diff(X.indptr)
ok_sum = np.allclose(rs, O["raw_sum"].to_numpy()); ok_nnz = bool(np.array_equal(nz, O["nnz"].to_numpy()))
print("check raw_sum", ok_sum, "check nnz", ok_nnz, flush=True)
assert ok_sum and ok_nnz, "reconstruction does not match the downloaded X"
summary["reconstruction_check"] = {"row_sums_equal_census_raw_sum": bool(ok_sum), "row_nnz_equal_census_nnz": ok_nnz}
O.index = [f"{j}-{c}" for j, c in zip(O["soma_joinid"], O["compartment"])]
O.index.name = "cell_id"
for c in O.columns:
    if O[c].dtype == object: O[c] = O[c].astype(str)
V = var.copy()
V["ensembl_id"] = [str(e).split(".")[0] for e in V["feature_id"]]
V.index = [str(s).upper() for s in V["feature_name"]]; V.index.name = None
A = ad.AnnData(X=X.astype(np.float32), obs=O, var=V)
A.var_names_make_unique()
A.write_h5ad(OUT)
o = A.obs
summary.update({"out": OUT, "n_cells": int(A.n_obs), "n_genes": int(A.n_vars),
                "integer_counts": bool(np.allclose(X.data, np.round(X.data))),
                "compartment_counts": o["compartment"].value_counts().to_dict(),
                "assay_counts": o["assay"].value_counts().to_dict(),
                "assay_by_compartment": pd.crosstab(o["compartment"], o["assay"]).to_dict(orient="index"),
                "dataset_counts": o["dataset_id"].value_counts().to_dict(),
                "dataset_by_compartment": pd.crosstab(o["dataset_id"], o["compartment"]).to_dict(orient="index"),
                "n_donors": int(o["donor_id"].nunique()),
                "donor_counts": o["donor_id"].value_counts().to_dict(),
                "suspension_type_counts": o["suspension_type"].value_counts().to_dict(),
                "tissue_by_compartment": {c: o.loc[o["compartment"] == c, "tissue"].value_counts().to_dict() for c in ["kidney", "lung", "immune"]},
                "cell_type_by_compartment": {c: o.loc[o["compartment"] == c, "cell_type"].value_counts().to_dict() for c in ["kidney", "lung", "immune"]},
                "median_raw_sum_by_compartment": o.groupby("compartment")["raw_sum"].median().to_dict(),
                "median_nnz_by_compartment": o.groupby("compartment")["nnz"].median().to_dict(),
                "wall_s_recovery": round(time.time() - t0, 1)})
json.dump(summary, open(os.path.join(HERE, "corpus_summary.json"), "w"), indent=1, default=str)
print("==> wrote", OUT, A.shape, flush=True)
