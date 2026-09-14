#!/usr/bin/env python
"""scPRINT Layer-3 (SAE) step 1: capture the per-gene-token residual stream at one
transformer block via a forward hook (independent of the broken attention path).
Each captured row is a gene's representation in a cell; saved with its gene id.

    conda activate scprint
    python scripts/scprint_capture_residual.py --layer 4 --max-tokens 40000
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ckpt_scprint/medium-v1.5.ckpt")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--n-genes", type=int, default=600)
    ap.add_argument("--n-cells", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=40000)
    ap.add_argument("--out", default="outputs/scprint")
    args = ap.parse_args()

    import numpy as np
    import scanpy as sc
    import torch
    from scdataloader import Preprocessor, SimpleAnnDataset, Collator
    from torch.utils.data import DataLoader
    from scprint import scPrint

    os.makedirs(args.out, exist_ok=True)
    adata = sc.datasets.pbmc3k()
    adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
    adata = Preprocessor(is_symbol=True, skip_validate=True, min_valid_genes_id=1000,
                         min_nnz_genes=100, filter_gene_by_counts=False)(adata)

    m = scPrint.load_from_checkpoint(args.checkpoint, precpt_gene_emb=None, transformer="normal")
    m.eval()
    n_cell = int(m.attn.additional_tokens)
    print(f"==> {m.nlayers} layers, d_model {m.gene_encoder.embeddings.weight.shape[1]}, "
          f"cell tokens {n_cell}")

    sc.pp.highly_variable_genes(adata, n_top_genes=args.n_genes, flavor="seurat_v3")
    fixed = [g for g in adata.var.index[adata.var.highly_variable].tolist() if g in set(m.genes)]
    print(f"==> {len(fixed)} genes; running {args.n_cells} cells")

    ds = SimpleAnnDataset(adata[: args.n_cells], obs_to_output=["organism_ontology_term_id"])
    col = Collator(organisms=m.organisms, valid_genes=m.genes, how="some", genelist=fixed, max_len=0)
    dl = DataLoader(ds, collate_fn=col, batch_size=16, shuffle=False)

    captured = {}
    h = m.transformer.blocks[args.layer].register_forward_hook(
        lambda mod, inp, out: captured.__setitem__("h", out[0] if isinstance(out, tuple) else out))

    acts, gene_ids = [], []
    genes_arr = np.asarray(m.genes)
    total = 0
    with torch.no_grad():
        for batch in dl:
            gp = batch["genes"]; expr = batch["x"]
            captured.clear()
            # call forward the way scPRINT's predict_step does (depth is required)
            m(gene_pos=gp, expression=expr, req_depth=batch["depth"],
              depth_mult=expr.sum(1))
            hid = captured["h"]                           # [B, seq, d]
            seq = hid.shape[1]; ng = gp.shape[1]
            g0 = seq - ng                                 # gene tokens are the last ng positions
            hg = hid[:, g0:, :]                           # [B, ng, d]
            for b in range(hg.shape[0]):
                acts.append(hg[b].cpu().numpy())
                gene_ids.append(genes_arr[gp[b].cpu().numpy()])
                total += ng
            if total >= args.max_tokens:
                break
    h.remove()
    X = np.concatenate(acts, axis=0)                      # [N, d]
    gid = np.concatenate(gene_ids, axis=0)                # [N] ensembl
    print(f"==> captured residual {X.shape} at layer {args.layer}; {len(set(gid))} unique genes")
    np.savez_compressed(os.path.join(args.out, f"residual_L{args.layer}.npz"), X=X.astype(np.float32))
    with open(os.path.join(args.out, f"residual_L{args.layer}_genes.txt"), "w") as f:
        f.write("\n".join(gid))
    print("==> saved to", args.out)


if __name__ == "__main__":
    main()
