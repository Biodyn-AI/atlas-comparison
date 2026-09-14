#!/usr/bin/env python
"""Step 1 (GPU): load CellPLM, dump the gene-embedding matrix and per-layer /
latent activations to outputs/activations/. Run on the H100.

    python scripts/01_extract.py --config configs/cellplm_audit.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

# make `mechaudit` importable when run from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mechaudit.config import load_config, resolve
from mechaudit.utils.io import ensure_dir, save_npz, save_gene_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    from mechaudit.models.cellplm_wrapper import CellPLMWrapper
    from mechaudit.data.loaders import load_adata

    out_dir = ensure_dir(os.path.join(resolve(cfg.outputs.dir),
                                       cfg.outputs.activations_subdir))

    print("==> Loading CellPLM")
    wrapper = CellPLMWrapper(
        pretrain_prefix=cfg.model.pretrain_prefix,
        pretrain_directory=resolve(cfg.model.pretrain_directory),
        device=cfg.model.device,
        cellplm_repo=resolve("external/CellPLM"),
    ).load()

    # Gene-embedding geometry (used by the spectral audit) -----------------
    print("==> Dumping gene-embedding matrix")
    E = wrapper.gene_embedding_numpy()
    gene_ids = wrapper.gene_ids
    save_npz(os.path.join(out_dir, "gene_embedding.npz"), embedding=E)
    save_gene_list(os.path.join(out_dir, "gene_ids.txt"), gene_ids)
    print(f"    gene_embedding: {E.shape}, n_gene_ids={len(gene_ids)}")

    # Per-cell activations -------------------------------------------------
    if any([cfg.extract.capture_layer_residual, cfg.extract.capture_latent,
            cfg.extract.capture_cell_pred]):
        print("==> Loading AnnData and capturing activations")
        adata = load_adata(resolve(cfg.data.adata_path),
                           max_cells=cfg.data.get("max_cells"))
        acts = wrapper.run_and_capture(
            adata,
            capture_layer_residual=cfg.extract.capture_layer_residual,
            capture_latent=cfg.extract.capture_latent,
            capture_cell_pred=cfg.extract.capture_cell_pred,
            ensembl_auto_conversion=cfg.data.ensembl_auto_conversion,
        )
        for k, v in acts.items():
            try:
                print(f"    {k}: shape={getattr(v, 'shape', None)}")
            except Exception:
                pass
        save_npz(os.path.join(out_dir, "cell_activations.npz"),
                 **{k: v for k, v in acts.items() if hasattr(v, "shape")})

    print("==> Done. Activations in", out_dir)


if __name__ == "__main__":
    main()
