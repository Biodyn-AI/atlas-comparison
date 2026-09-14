#!/usr/bin/env python
"""Step 2 (CPU ok): spectral audit of the gene-embedding matrix + gene-set
annotation of the axes. Reads outputs/activations/gene_embedding.npz.

    python scripts/02_spectral.py --config configs/cellplm_audit.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mechaudit.config import load_config, resolve
from mechaudit.utils.io import (ensure_dir, load_npz, load_gene_list,
                                save_json)
from mechaudit.audit.spectral import spectral_decompose, summarise_axes
from mechaudit.annotate.genesets import (load_trrust_genesets, load_symbol_map,
                                         mygene_symbol_map, annotate_axes,
                                         top_annotations_per_axis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    act_dir = os.path.join(resolve(cfg.outputs.dir), cfg.outputs.activations_subdir)
    out_dir = ensure_dir(os.path.join(resolve(cfg.outputs.dir),
                                      cfg.outputs.spectral_subdir))

    E = load_npz(os.path.join(act_dir, "gene_embedding.npz"))["embedding"]
    gene_ids = load_gene_list(os.path.join(act_dir, "gene_ids.txt"))
    print(f"==> gene_embedding {E.shape}, {len(gene_ids)} genes")

    res = spectral_decompose(
        E, gene_ids,
        n_axes=cfg.spectral.n_axes,
        center=cfg.spectral.center,
        standardize=cfg.spectral.standardize,
    )
    axes_summary = summarise_axes(res, top_genes_per_axis=cfg.spectral.top_genes_per_axis)
    save_json(os.path.join(out_dir, "axes_summary.json"), axes_summary)
    print(f"==> {res.axes.shape[0]} axes; "
          f"top EVR: {res.explained_variance_ratio[:5].round(4).tolist()}")

    # ---- annotate axes against TRRUST gene sets --------------------------
    trrust_path = resolve(cfg.annotate.trrust_path)
    genesets = load_trrust_genesets(trrust_path).filter_by_size(cfg.annotate.min_geneset_size)
    print(f"==> {len(genesets.sets)} gene sets from TRRUST")

    smap_path = cfg.annotate.get("gene_symbol_map")
    symbol_map = load_symbol_map(resolve(smap_path) if smap_path else None)
    if not symbol_map:
        print("==> No symbol map provided; trying mygene (needs network)...")
        try:
            symbol_map = mygene_symbol_map(gene_ids)
        except Exception as e:  # noqa
            print(f"    mygene failed ({e}); skipping annotation. "
                  f"Provide annotate.gene_symbol_map in the config to enable it.")
            symbol_map = {}

    if symbol_map:
        ann = annotate_axes(res.gene_loadings, gene_ids, genesets, symbol_map,
                            min_geneset_size=cfg.annotate.min_geneset_size)
        ann.to_csv(os.path.join(out_dir, "axis_annotations.csv"), index=False)
        top = top_annotations_per_axis(ann, top=5)
        top.to_csv(os.path.join(out_dir, "axis_annotations_top.csv"), index=False)
        print("==> Top axis annotations:")
        print(top.head(20).to_string(index=False))

    print("==> Spectral audit written to", out_dir)


if __name__ == "__main__":
    main()
