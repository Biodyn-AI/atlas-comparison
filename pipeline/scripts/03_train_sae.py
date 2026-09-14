#!/usr/bin/env python
"""Step 3 (GPU): extract the per-cell residual stream at one encoder layer and
train a TopK SAE on it. Saves the SAE, its input-norm stats, the feature
activations, and the aligned cell metadata + expression (for offline
annotation in 04).

    python scripts/03_train_sae.py --config configs/cellplm_audit.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mechaudit.config import load_config, resolve
from mechaudit.utils.io import ensure_dir, save_npz, save_gene_list, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    import numpy as np
    import scipy.sparse as sp
    import torch

    from mechaudit.models.cellplm_wrapper import CellPLMWrapper
    from mechaudit.data.loaders import load_adata
    from mechaudit.sae.topk_sae import (SAEConfig, train_sae, feature_activations)

    act_dir = ensure_dir(os.path.join(resolve(cfg.outputs.dir),
                                      cfg.outputs.activations_subdir))
    sae_dir = ensure_dir(os.path.join(resolve(cfg.outputs.dir), cfg.outputs.sae_subdir))
    layer = cfg.sae.layer

    print("==> Loading CellPLM")
    wrapper = CellPLMWrapper(
        pretrain_prefix=cfg.model.pretrain_prefix,
        pretrain_directory=resolve(cfg.model.pretrain_directory),
        device=cfg.model.device,
        cellplm_repo=resolve("external/CellPLM"),
    ).load()

    print("==> Extracting per-cell residual stream at layer", layer)
    adata = load_adata(resolve(cfg.data.adata_path), max_cells=cfg.data.get("max_cells"))
    states = wrapper.extract_cell_states(adata, layers=[layer], capture_latent=True,
                                         ensembl_auto_conversion=cfg.data.ensembl_auto_conversion)
    resid = states[f"resid_layer{layer}"]           # [n_cells, hidden]
    pp = states["_adata"]
    print(f"    residual {resid.shape}")

    # persist aligned metadata + expression for offline annotation ---------
    ct_field = cfg.data.get("cell_type_field")
    cell_types = (pp.obs[ct_field].astype(str).to_numpy()
                  if ct_field and ct_field in pp.obs else np.array(["NA"] * pp.n_obs))
    np.save(os.path.join(act_dir, "cell_types.npy"), cell_types)
    save_gene_list(os.path.join(act_dir, "expr_gene_ids.txt"), list(pp.var_names))
    X = pp.X
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
    sp.save_npz(os.path.join(act_dir, "expr.npz"), X.astype(np.float32))
    save_npz(os.path.join(act_dir, f"resid_layer{layer}.npz"), resid=resid)
    # persist the preprocessed AnnData so Layer-4 (circuits) can re-run the model
    # on the exact same, obs-aligned cells.
    try:
        pp.write_h5ad(os.path.join(act_dir, "preprocessed.h5ad"))
    except Exception as e:  # noqa
        print(f"    (could not write preprocessed.h5ad: {e})")

    # train SAE ------------------------------------------------------------
    print("==> Training TopK SAE")
    sae_cfg = SAEConfig(d_sae=cfg.sae.d_sae, k=cfg.sae.k, auxk=cfg.sae.auxk,
                        auxk_coef=cfg.sae.auxk_coef, lr=cfg.sae.lr,
                        batch_size=cfg.sae.batch_size, epochs=cfg.sae.epochs,
                        dead_steps_threshold=cfg.sae.dead_steps_threshold, seed=cfg.sae.seed)
    sae, stats, log = train_sae(resid, sae_cfg, device=cfg.model.device)

    torch.save({"state_dict": sae.state_dict(), "cfg": sae_cfg.__dict__,
                "norm_mean": stats.mean.cpu(), "norm_scale": stats.scale,
                "d_in": resid.shape[1], "layer": layer},
               os.path.join(sae_dir, f"sae_layer{layer}.pt"))
    save_json(os.path.join(sae_dir, f"train_log_layer{layer}.json"), log.history)

    print("==> Computing feature activations")
    F = feature_activations(sae, resid, stats, device=cfg.model.device)
    save_npz(os.path.join(sae_dir, f"features_layer{layer}.npz"), F=F)
    print(f"    features {F.shape}; final FVU={log.history[-1]['fvu']:.4f} "
          f"dead={log.history[-1]['dead_frac']:.3f}")
    print("==> Done. SAE + features in", sae_dir)


if __name__ == "__main__":
    main()
