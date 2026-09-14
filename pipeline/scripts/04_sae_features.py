#!/usr/bin/env python
"""Step 4 (CPU ok): annotate cell-state SAE features — cell-type selectivity
(AUROC), gene attribution, exemplar cells. Reads the outputs of 03.

    python scripts/04_sae_features.py --config configs/cellplm_audit.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mechaudit.config import load_config, resolve
from mechaudit.utils.io import ensure_dir, load_npz, load_gene_list, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-features-detailed", type=int, default=50,
                    help="how many top-selective features to write gene attributions for")
    args = ap.parse_args()
    cfg = load_config(args.config)

    import numpy as np
    import scipy.sparse as sp

    from mechaudit.audit.sae_features import (
        feature_stats, annotate_features_by_celltype, top_celltype_per_feature,
        feature_gene_attribution, top_activating_cells)

    layer = cfg.sae.layer
    act_dir = os.path.join(resolve(cfg.outputs.dir), cfg.outputs.activations_subdir)
    sae_dir = os.path.join(resolve(cfg.outputs.dir), cfg.outputs.sae_subdir)
    out_dir = ensure_dir(sae_dir)

    F = load_npz(os.path.join(sae_dir, f"features_layer{layer}.npz"))["F"]
    cell_types = np.load(os.path.join(act_dir, "cell_types.npy"), allow_pickle=True)
    print(f"==> features {F.shape}, {len(np.unique(cell_types))} cell types")

    feature_stats(F).to_csv(os.path.join(out_dir, f"feature_stats_layer{layer}.csv"),
                            index=False)

    ann = annotate_features_by_celltype(F, cell_types)
    ann.to_csv(os.path.join(out_dir, f"feature_celltype_auroc_layer{layer}.csv"), index=False)
    top = top_celltype_per_feature(ann, top=1)
    top.to_csv(os.path.join(out_dir, f"feature_top_celltype_layer{layer}.csv"), index=False)
    if not top.empty:
        print("==> Most cell-type-selective features:")
        print(top.sort_values("selectivity", ascending=False).head(20).to_string(index=False))

    # gene attribution for the most selective features ---------------------
    expr_path = os.path.join(act_dir, "expr.npz")
    if os.path.exists(expr_path) and not top.empty:
        expr = sp.load_npz(expr_path)
        expr = np.log1p(expr.toarray().astype(np.float32))
        gene_ids = load_gene_list(os.path.join(act_dir, "expr_gene_ids.txt"))
        detailed = {}
        feats = top.sort_values("selectivity", ascending=False)["feature"].tolist()
        for fi in feats[:args.n_features_detailed]:
            attr = feature_gene_attribution(F, expr, gene_ids, int(fi), top_k=30)
            exemplars = top_activating_cells(F, int(fi), top_k=20)
            detailed[int(fi)] = {
                "cell_type": top.loc[top.feature == fi, "cell_type"].iloc[0],
                "top_genes": attr.to_dict(orient="records"),
                "exemplar_cell_types": [str(cell_types[i]) for i in exemplars],
            }
        save_json(os.path.join(out_dir, f"feature_gene_attribution_layer{layer}.json"),
                  detailed)
        print(f"==> Wrote gene attributions for {len(detailed)} features")

    print("==> Feature annotation written to", out_dir)


if __name__ == "__main__":
    main()
