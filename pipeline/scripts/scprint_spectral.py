#!/usr/bin/env python
"""scPRINT-12M audit — Layer 2 (spectral geometry). Runs LOCALLY with just torch
+ numpy by reading the gene-embedding matrix straight from the Lightning .ckpt
(no scprint/lamindb needed). SVD of gene embeddings, species-axis check
(scPRINT is cross-species human+mouse), and TRRUST/TF annotation of human axes.

    conda activate bae   # any env with torch 2.2 + numpy<2
    python scripts/scprint_spectral.py --checkpoint ckpt_scprint/medium-v1.5.ckpt
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from mechaudit.models.scprint_wrapper import ScPrintWrapper
from mechaudit.utils.io import ensure_dir, save_json, save_gene_list
from mechaudit.audit.spectral import spectral_decompose, summarise_axes
from mechaudit.annotate.genesets import (load_trrust_genesets, annotate_axes,
                                         top_annotations_per_axis)
from mechaudit.audit.bae_manifold import auroc_all_latents  # reuse rank-AUROC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ckpt_scprint/medium-v1.5.ckpt")
    ap.add_argument("--trrust", default="external/single_cell_mechinterp/external/networks/trrust_human.tsv")
    ap.add_argument("--biomart", default="external/scprint_data/biomart_pos.parquet")
    ap.add_argument("--tfs", default="external/scprint_data/TFs.txt")
    ap.add_argument("--n-axes", type=int, default=32)
    ap.add_argument("--out", default="outputs/scprint")
    args = ap.parse_args()

    import pandas as pd
    out = ensure_dir(args.out)

    E, genes = ScPrintWrapper.light_gene_embedding(args.checkpoint)
    is_human = np.array([g.startswith("ENSG") for g in genes])
    print(f"==> gene embedding {E.shape}; {is_human.sum()} human / {(~is_human).sum()} mouse genes")
    np.savez_compressed(os.path.join(out, "gene_embedding.npz"), embedding=E)
    save_gene_list(os.path.join(out, "gene_ids.txt"), genes)

    # full-vocab SVD -> which axis encodes species?
    res_all = spectral_decompose(E, genes, n_axes=args.n_axes, center=True)
    sp_auroc = auroc_all_latents(res_all.gene_loadings, is_human)  # per-axis
    sp_axis = int(np.abs(sp_auroc - 0.5).argmax())
    print(f"==> species (human/mouse) axis = {sp_axis} "
          f"(|AUROC-0.5|·2 = {abs(sp_auroc[sp_axis]-0.5)*2:.2f}); "
          f"top EVR {res_all.explained_variance_ratio[:4].round(3).tolist()}")

    # human-only SVD -> clean biological axes
    Eh, gh = E[is_human], [g for g, m in zip(genes, is_human) if m]
    res = spectral_decompose(Eh, gh, n_axes=args.n_axes, center=True)
    save_json(os.path.join(out, "axes_summary_human.json"),
              summarise_axes(res, top_genes_per_axis=40))

    bm = pd.read_parquet(args.biomart)
    smap = {ens: str(sym) for ens, sym in bm["hgnc_symbol"].items()}
    gs = load_trrust_genesets(args.trrust).filter_by_size(10)
    gs.add("TF_list", {l.strip().upper() for l in open(args.tfs)})
    ann = annotate_axes(res.gene_loadings, gh, gs, smap, min_geneset_size=10)
    ann.to_csv(os.path.join(out, "axis_annotations_human.csv"), index=False)
    top = top_annotations_per_axis(ann, top=3)
    top.to_csv(os.path.join(out, "axis_annotations_top.csv"), index=False)
    print("==> Top human axis annotations (TRRUST regulons / TF list):")
    print(top.head(24).to_string(index=False))
    print("==> Layer-2 spectral audit written to", out)


if __name__ == "__main__":
    main()
