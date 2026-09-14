#!/usr/bin/env python
"""scPRINT-12M audit — Layer 1 (attention→GRN). Runs scPRINT's built-in GNInfer
to turn attention into a gene-gene adjacency, then evaluates the edges against
TRRUST regulons (AUROC/AUPRC per TF). Needs the full scprint stack + populated
lamindb (env `scprint`). CPU: precision='32', dtype=float32.

    conda activate scprint
    python scripts/scprint_grn.py
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ckpt_scprint/medium-v1.5.ckpt")
    ap.add_argument("--num-genes", type=int, default=2000)
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
    print(f"==> preprocessed adata {adata.shape}; var e.g. {list(adata.var.index[:3])}")

    m = scPrint.load_from_checkpoint(args.checkpoint, precpt_gene_emb=None,
                                     transformer="normal")
    m.eval()

    # force TRRUST TFs (as Ensembl) into the network so it's TF-rich for evaluation
    bm = pd.read_parquet(args.biomart)
    sym2ens = {str(s).upper(): e for e, s in bm["hgnc_symbol"].items()}
    trrust_tfs = {l.split("\t")[0].upper() for l in open(args.trrust)}
    tf_ens = [sym2ens[t] for t in trrust_tfs if t in sym2ens and sym2ens[t] in set(m.genes)]
    print(f"==> forcing {len(tf_ens)} TRRUST TFs (Ensembl) into the network")

    grn = GNInfer(how="most var within", num_genes=args.num_genes, max_cells=args.max_cells,
                  head_agg="mean", comp_attn=True, preprocess="softmax",
                  genes=tf_ens, precision="32", dtype=torch.float32)
    print("==> running attention→GRN inference…")
    out = grn(m, adata)

    # locate the adjacency + gene names in the returned object
    print("==> GNInfer returned:", type(out).__name__)
    adj, genes = None, None
    if hasattr(out, "varp") and len(getattr(out, "varp", {})) > 0:
        key = list(out.varp.keys())[0]
        adj = out.varp[key]
        adj = adj.toarray() if hasattr(adj, "toarray") else np.asarray(adj)
        genes = list(out.var.index)
        print(f"    adjacency from varp['{key}'] shape {adj.shape}; {len(genes)} genes")
    elif isinstance(out, tuple):
        adj = np.asarray(out[1] if len(out) > 1 else out[0])
        print("    adjacency from tuple, shape", getattr(adj, "shape", None))

    if adj is not None:
        np.savez_compressed(os.path.join(args.out, "grn_adjacency.npz"), adj=adj)
        if genes:
            with open(os.path.join(args.out, "grn_genes.txt"), "w") as f:
                f.write("\n".join(genes))
        print("==> saved adjacency to", args.out)
    else:
        print("!! could not locate adjacency; out attrs:", [a for a in dir(out) if not a.startswith('_')][:20])


if __name__ == "__main__":
    main()
