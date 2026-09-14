#!/usr/bin/env python
"""Step 5 (GPU): causal circuit tracing. Ablate the most cell-type-selective
SAE features (from step 4) via activation patching and check whether the cells
whose embeddings shift most are the annotated cell type — causal confirmation
of the Layer-3 annotation.

    python scripts/05_circuits.py --config configs/cellplm_audit.yaml --top-features 30
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mechaudit.config import load_config, resolve
from mechaudit.utils.io import ensure_dir, load_npz, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--top-features", type=int, default=30)
    args = ap.parse_args()
    cfg = load_config(args.config)

    import numpy as np
    import pandas as pd
    import anndata as ad
    import torch

    from mechaudit.models.cellplm_wrapper import CellPLMWrapper
    from mechaudit.sae.topk_sae import TopKSAE
    from mechaudit.audit.circuits import ablation_effects

    layer = cfg.sae.layer
    act_dir = os.path.join(resolve(cfg.outputs.dir), cfg.outputs.activations_subdir)
    sae_dir = os.path.join(resolve(cfg.outputs.dir), cfg.outputs.sae_subdir)
    out_dir = ensure_dir(sae_dir)

    # inputs from steps 3 & 4 ---------------------------------------------
    F = load_npz(os.path.join(sae_dir, f"features_layer{layer}.npz"))["F"]
    cell_types = np.load(os.path.join(act_dir, "cell_types.npy"), allow_pickle=True)
    pp = ad.read_h5ad(os.path.join(act_dir, "preprocessed.h5ad"))
    top_csv = os.path.join(sae_dir, f"feature_top_celltype_layer{layer}.csv")
    top = pd.read_csv(top_csv).sort_values("selectivity", ascending=False)
    feats = top["feature"].head(args.top_features).astype(int).tolist()
    annotated = dict(zip(top["feature"].astype(int), top["cell_type"].astype(str)))
    print(f"==> Tracing {len(feats)} features at layer {layer}")

    # load model + SAE ----------------------------------------------------
    wrapper = CellPLMWrapper(
        pretrain_prefix=cfg.model.pretrain_prefix,
        pretrain_directory=resolve(cfg.model.pretrain_directory),
        device=cfg.model.device,
        cellplm_repo=resolve("external/CellPLM"),
    ).load()

    bundle = torch.load(os.path.join(sae_dir, f"sae_layer{layer}.pt"),
                        map_location=cfg.model.device)
    scfg = bundle["cfg"]
    sae = TopKSAE(bundle["d_in"], scfg["d_sae"], scfg["k"], auxk=scfg["auxk"]).to(cfg.model.device)
    sae.load_state_dict(bundle["state_dict"])
    sae.eval()

    results = ablation_effects(
        wrapper, sae, bundle["norm_mean"], bundle["norm_scale"], layer, pp,
        features=feats, feature_activations=F, cell_types=cell_types,
        annotated_type=annotated, ensembl_auto_conversion=cfg.data.ensembl_auto_conversion)

    save_json(os.path.join(out_dir, f"circuit_effects_layer{layer}.json"), results)

    # headline: causal-confirmation rate ----------------------------------
    checks = [(fi, r) for fi, r in results.items() if r.get("matches_annotation") is not None]
    match_rate = np.mean([r["matches_annotation"] for _, r in checks]) if checks else float("nan")
    rows = [{"feature": fi, "annotated_type": annotated.get(fi),
             "most_perturbed_type": r["most_perturbed_type"],
             "mean_shift": r["mean_shift"], "matches": r["matches_annotation"],
             "n_cells": r["n_cells"]} for fi, r in results.items()]
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"circuit_summary_layer{layer}.csv"), index=False)
    print(f"==> Causal confirmation rate: {match_rate:.2%} "
          f"({sum(r['matches_annotation'] for _, r in checks)}/{len(checks)} features)")
    print("==> Circuit results in", out_dir)


if __name__ == "__main__":
    main()
