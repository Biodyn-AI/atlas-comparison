#!/usr/bin/env python
"""Question 2: the pairs most models agree on are salivary-gland secretory genes (HTN1, HTN3, PRB1-4,
PRH2, STATH, SMR3B ...). In the rebuilt 6,000-cell corpus: how many cells express them, in which
compartment / tissue / dataset / donor / cell type, at what levels, and does it look like ambient RNA
(few UMIs in many cells of one salivary-gland sample) or real expression (high counts in few cells)?

usage: python salivary_check.py <corpus.h5ad>  -> salivary_check.json next to this script
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = sys.argv[1]
CORE = ["HTN1", "HTN3", "PRB1", "PRB2", "PRB3", "PRB4", "PRH1", "PRH2", "STATH", "SMR3A", "SMR3B"]
EXTRA = ["MUC7", "AMY1A", "AMY1B", "AMY1C", "PRR27", "PRR4", "CST1", "CST2", "CST4", "CST5",
         "ZG16B", "LPO", "BPIFA2", "OPRPN", "PIP", "LYZ"]           # LYZ = non-salivary control (myeloid)
TOP_PAIRS = [("HTN1", "PRB4"), ("PRB3", "PRH2"), ("PRB3", "STATH"), ("HTN3", "PRB4"), ("PRB4", "SMR3B"),
             ("PRH2", "SMR3B"), ("HTN1", "SMR3B")]    # hypothesis_trrust2.json:hypotheses[0..6]

import anndata as ad
from scipy import sparse

A = ad.read_h5ad(CORPUS)
X = sparse.csr_matrix(A.X, dtype=np.float32)
names = [str(s).upper() for s in A.var_names]
idx = {}
for i, s in enumerate(names): idx.setdefault(s, i)
obs = A.obs.copy()
for c in obs.columns:
    if hasattr(obs[c], "cat"): obs[c] = obs[c].astype(str)
tot = np.asarray(X.sum(1)).ravel()
obs["total_counts"] = tot
SAL = [g for g in CORE if g in idx]
SALX = [g for g in CORE + EXTRA if g in idx]
missing = [g for g in CORE + EXTRA if g not in idx]
Xs = X[:, [idx[g] for g in SALX]].toarray()
D = pd.DataFrame(Xs, columns=SALX, index=obs.index)
core = D[SAL].sum(1)
obs["sal_core_counts"] = core.values
obs["sal_core_ngenes"] = (D[SAL] > 0).sum(1).values
obs["sal_core_frac"] = core.values / np.maximum(tot, 1)
SG_TISSUES = [t for t in obs["tissue"].unique() if any(k in t.lower() for k in ("saliv", "parotid", "submandibular", "sublingual", "tongue"))]
obs["from_salivary_tissue"] = obs["tissue"].isin(SG_TISSUES)

res = {"corpus": CORPUS, "n_cells": int(A.n_obs), "core_genes": SAL, "extra_genes_checked": SALX[len(SAL):],
       "not_in_corpus": missing, "salivary_like_tissues_in_corpus": SG_TISSUES}

# ---- per-gene summary
pg = {}
for g in SALX:
    v = D[g].to_numpy(); e = v > 0
    d = {"n_cells_expr": int(e.sum()), "total_umi": float(v.sum()),
         "max": float(v.max()) if e.any() else 0.0,
         "median_among_expr": float(np.median(v[e])) if e.any() else None,
         "share_expr_with_count_1": round(float((v[e] == 1).mean()), 3) if e.any() else None,
         "by_compartment_n_expr": obs.loc[e, "compartment"].value_counts().to_dict(),
         "by_tissue_n_expr": obs.loc[e, "tissue"].value_counts().head(8).to_dict(),
         "share_umi_from_salivary_tissue_cells": round(float(v[obs["from_salivary_tissue"].to_numpy()].sum()) / max(v.sum(), 1), 4)}
    pg[g] = d
res["per_gene"] = pg

# ---- per-cell core salivary counts
e = obs["sal_core_counts"] > 0
res["cells_with_any_core_read"] = int(e.sum())
res["core_counts_distribution_among_expr"] = {
    q: float(np.quantile(obs.loc[e, "sal_core_counts"], p)) for q, p in
    (("min", 0), ("q25", .25), ("median", .5), ("q75", .75), ("q95", .95), ("max", 1))} if e.any() else None
res["core_frac_of_library_among_expr"] = {
    q: float(np.quantile(obs.loc[e, "sal_core_frac"], p)) for q, p in
    (("median", .5), ("q95", .95), ("max", 1))} if e.any() else None
for col in ["compartment", "tissue", "dataset_id", "donor_id", "cell_type", "assay"]:
    t = obs.groupby(col).agg(n_cells=("sal_core_counts", "size"),
                             n_with_core=("sal_core_counts", lambda x: int((x > 0).sum())),
                             core_umi=("sal_core_counts", "sum"),
                             median_core_among_expr=("sal_core_counts", lambda x: float(np.median(x[x > 0])) if (x > 0).any() else 0.0))
    t["share_with_core"] = (t["n_with_core"] / t["n_cells"]).round(4)
    t = t.sort_values("core_umi", ascending=False)
    res[f"by_{col}"] = t.head(25).reset_index().to_dict(orient="records")

# ---- where do all core salivary UMIs come from
tu = obs["sal_core_counts"].sum()
res["core_umi_total"] = float(tu)
res["core_umi_share_from_salivary_tissue"] = round(float(obs.loc[obs["from_salivary_tissue"], "sal_core_counts"].sum()) / max(tu, 1), 4)
res["cells_from_salivary_tissue"] = int(obs["from_salivary_tissue"].sum())
res["cells_from_salivary_tissue_by_compartment"] = obs.loc[obs["from_salivary_tissue"], "compartment"].value_counts().to_dict()
res["cells_from_salivary_tissue_by_celltype"] = obs.loc[obs["from_salivary_tissue"], "cell_type"].value_counts().to_dict()
sg = obs["from_salivary_tissue"]
res["share_with_core_read"] = {"salivary_tissue_cells": round(float((obs.loc[sg, "sal_core_counts"] > 0).mean()), 4) if sg.any() else None,
                               "other_cells": round(float((obs.loc[~sg, "sal_core_counts"] > 0).mean()), 4)}
# top 20 cells by core counts
top = obs.sort_values("sal_core_counts", ascending=False).head(20)
res["top20_cells"] = top[["compartment", "tissue", "cell_type", "dataset_id", "donor_id", "assay",
                          "total_counts", "sal_core_counts", "sal_core_ngenes", "sal_core_frac"]].reset_index().to_dict(orient="records")
# is it spread across many cells with few UMIs (ambient) or concentrated (real)?
if e.any():
    s = np.sort(obs.loc[e, "sal_core_counts"].to_numpy())[::-1]
    res["umi_concentration"] = {"share_of_core_umi_in_top1pct_expr_cells": round(float(s[:max(1, len(s)//100)].sum() / s.sum()), 4),
                                "share_of_core_umi_in_top10_cells": round(float(s[:10].sum() / s.sum()), 4),
                                "n_cells_core_ge_10": int((s >= 10).sum()), "n_cells_core_ge_100": int((s >= 100).sum())}

# ---- co-expression of the salivary pairs (Pearson on counts/10k + log1p), overall and by subset
L = np.log1p(Xs / np.maximum(tot, 1)[:, None] * 1e4)
Ld = pd.DataFrame(L, columns=SALX, index=obs.index)
def r_of(a, b, m=None):
    x = Ld[a].to_numpy(); y = Ld[b].to_numpy()
    if m is not None: x, y = x[m], y[m]
    if x.std() == 0 or y.std() == 0: return None
    return round(float(np.corrcoef(x, y)[0, 1]), 4)
res["top_pairs_pearson"] = {f"{a}-{b}": {"all": r_of(a, b) if a in Ld and b in Ld else None,
                                        "salivary_tissue_cells": r_of(a, b, sg.to_numpy()) if a in Ld and b in Ld and sg.any() else None,
                                        "other_cells": r_of(a, b, (~sg).to_numpy()) if a in Ld and b in Ld else None,
                                        "n_cells_both_expr": int(((D[a] > 0) & (D[b] > 0)).sum()) if a in D and b in D else None}
                            for a, b in TOP_PAIRS}
# ---- same summaries without PRH1 (PRH1 overlaps the PRH1-PRR4 / PRH1-TAS2R14 read-through locus and
#      shows a different, broad 1-UMI pattern) and per-gene counts of the 20 top cells
SAL2 = [g for g in SAL if g != "PRH1"]
c2 = D[SAL2].sum(1).to_numpy(); sgm = obs["from_salivary_tissue"].to_numpy()
res["core_without_PRH1"] = {
    "genes": SAL2, "cells_with_any_read": int((c2 > 0).sum()),
    "cells_with_any_read_salivary_tissue": int(((c2 > 0) & sgm).sum()),
    "cells_with_any_read_other_tissue": int(((c2 > 0) & ~sgm).sum()),
    "umi_total": float(c2.sum()), "umi_share_from_salivary_tissue": round(float(c2[sgm].sum() / max(c2.sum(), 1)), 4),
    "median_umi_salivary_tissue_expr_cells": float(np.median(c2[(c2 > 0) & sgm])) if ((c2 > 0) & sgm).any() else None,
    "median_umi_other_expr_cells": float(np.median(c2[(c2 > 0) & ~sgm])) if ((c2 > 0) & ~sgm).any() else None,
    "max_umi_other_cells": float(c2[~sgm].max()),
    "n_other_cells_ge_10_umi": int(((c2 >= 10) & ~sgm).sum()),
    "other_cells_ge_10_umi": obs.loc[(c2 >= 10) & ~sgm, ["compartment", "tissue", "cell_type", "donor_id", "assay", "total_counts"]].assign(core_umi=c2[(c2 >= 10) & ~sgm]).reset_index().to_dict(orient="records"),
    "salivary_tissue_cells_n": int(sgm.sum()),
    "salivary_tissue_cells_with_all_10_genes": int(((D[SAL2] > 0).sum(1).to_numpy() == len(SAL2))[sgm].sum()),
    "salivary_tissue_by_donor_tissue": obs.loc[sgm].groupby(["donor_id", "tissue"]).size().reset_index(name="n").to_dict(orient="records"),
    "salivary_tissue_share_with_read_by_donor_tissue": obs.loc[sgm].assign(has=(c2[sgm] > 0)).groupby(["donor_id", "tissue"])["has"].mean().round(3).reset_index().to_dict(orient="records")}
res["top20_cells_gene_counts"] = D.loc[obs.sort_values("sal_core_counts", ascending=False).head(20).index, SAL].astype(int).reset_index().to_dict(orient="records")
json.dump(res, open(os.path.join(HERE, "salivary_check.json"), "w"), indent=1, default=str)
print(json.dumps({k: res[k] for k in ["cells_with_any_core_read", "core_umi_total", "core_umi_share_from_salivary_tissue",
                                        "cells_from_salivary_tissue", "share_with_core_read", "salivary_like_tissues_in_corpus"]}, indent=1, default=str))
for g in SAL:
    print(g, pg[g]["n_cells_expr"], pg[g]["total_umi"], pg[g]["max"], pg[g]["by_compartment_n_expr"])
