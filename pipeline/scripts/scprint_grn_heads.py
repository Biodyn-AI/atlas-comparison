#!/usr/bin/env python
"""scPRINT Layer-1 deep dive: per-(layer,head) attention. Runs GNInfer with
head_agg="none" so the adjacency keeps the head/layer dimension
[genes, genes, n_layers*n_heads]. Saves it for per-head TRRUST evaluation.

    conda activate scprint
    python scripts/scprint_grn_heads.py --num-genes 1000
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ckpt_scprint/medium-v1.5.ckpt")
    ap.add_argument("--num-genes", type=int, default=1000)
    ap.add_argument("--max-cells", type=int, default=64)
    ap.add_argument("--trrust", default="external/single_cell_mechinterp/external/networks/trrust_human.tsv")
    ap.add_argument("--biomart", default="external/scprint_data/biomart_pos.parquet")
    ap.add_argument("--out", default="outputs/scprint")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import scanpy as sc
    import torch
    from scdataloader import Preprocessor
    from scprint import scPrint
    from scprint.tasks import GNInfer

    os.makedirs(args.out, exist_ok=True)
    adata = sc.datasets.pbmc3k()
    adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
    adata = Preprocessor(is_symbol=True, skip_validate=True, min_valid_genes_id=1000,
                         min_nnz_genes=100, filter_gene_by_counts=False)(adata)

    m = scPrint.load_from_checkpoint(args.checkpoint, precpt_gene_emb=None, transformer="normal")
    m.eval()
    print(f"==> model: {m.nlayers} layers, {m.attn.num_heads if hasattr(m.attn,'num_heads') else '?'} heads/layer")

    bm = pd.read_parquet(args.biomart)
    sym2ens = {str(s).upper(): e for e, s in bm["hgnc_symbol"].items()}
    trrust_tfs = {l.split("\t")[0].upper() for l in open(args.trrust)}
    tf_ens = [sym2ens[t] for t in trrust_tfs if t in sym2ens and sym2ens[t] in set(m.genes)]
    # FIXED gene set (same for every cell) so the per-head accumulator sizes match:
    # top-HVG + TRRUST TFs, intersected with the model vocab.
    sc.pp.highly_variable_genes(adata, n_top_genes=args.num_genes, flavor="seurat_v3")
    hvg = adata.var.index[adata.var.highly_variable].tolist()
    mgenes = set(m.genes)
    fixed = [g for g in dict.fromkeys(list(hvg) + tf_ens) if g in mgenes]
    print(f"==> fixed gene set: {len(fixed)} genes ({len(tf_ens)} TFs forced in)")

    # head_agg='none' keeps per-head (note: scPRINT stores only the first 2 heads/layer
    # in this path); how='given' with a fixed gene list avoids the variable-size bug.
    grn = GNInfer(how="given", genes=fixed, max_cells=args.max_cells,
                  head_agg="none", filtration="none", comp_attn=False, drop_unexpressed=False,
                  preprocess="softmax", precision="32", dtype=torch.float32)
    print("==> running per-head attention→GRN…")
    out = grn(m, adata)

    key = list(out.varp.keys())[0]
    adj = out.varp[key]
    adj = adj.toarray() if hasattr(adj, "toarray") else np.asarray(adj)
    genes = list(out.var.index)
    print(f"==> per-head adjacency {adj.shape} (expect [G,G,H]); {len(genes)} genes")
    np.savez_compressed(os.path.join(args.out, "grn_adj_perhead.npz"), adj=adj.astype(np.float32))
    with open(os.path.join(args.out, "grn_perhead_genes.txt"), "w") as f:
        f.write("\n".join(genes))
    print("==> saved per-head adjacency to", args.out)


if __name__ == "__main__":
    main()
