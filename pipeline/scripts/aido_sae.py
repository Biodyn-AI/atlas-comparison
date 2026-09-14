#!/usr/bin/env python
"""AIDO.Cell-10M audit — Layer 3 (SAE on the residual stream). Tests whether the
gene semantics that are WEAK in the input gene(-position) embedding (Layer 2, flat
spectrum) emerge downstream in the residual. Captures a middle-layer hidden state
for TRRUST genes, trains a TopK SAE, and annotates the gene-level features vs
TRRUST regulons (reusing mechaudit). Run in env `scprint` (has gb_cell + mechaudit).

    conda activate scprint
    python scripts/aido_sae.py --layer 4
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="genbio-ai/AIDO.Cell-10M")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--n-cells", type=int, default=48)
    ap.add_argument("--max-tokens", type=int, default=40000)
    ap.add_argument("--d-sae", type=int, default=2048)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--trrust", default="external/single_cell_mechinterp/external/networks/trrust_human.tsv")
    ap.add_argument("--tfs", default="external/scprint_data/TFs.txt")
    ap.add_argument("--out", default="outputs/aido")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import scanpy as sc
    import torch
    from gb_cell.models import CellFoundationModel, CellFoundationConfig
    from gb_cell.utils import align_adata, preprocess_counts
    from mechaudit.sae.topk_sae import SAEConfig, train_sae, feature_activations
    from mechaudit.annotate.genesets import (load_trrust_genesets, annotate_axes,
                                             top_annotations_per_axis)

    os.makedirs(args.out, exist_ok=True)
    genes = [l.split("\t")[0] for l in open("external/scprint_data/aido_genes.tsv").read().splitlines()[1:]]
    sym2pos = {g: i for i, g in enumerate(genes)}

    # TRRUST genes (TFs ∪ targets) present in the AIDO vocab -> positions to keep
    tr = pd.read_csv(args.trrust, sep="\t", header=None, names=["tf", "tg", "m", "p"])
    tr_genes = set(tr.tf.str.upper()) | set(tr.tg.str.upper())
    keep = sorted(sym2pos[g] for g in tr_genes if g in sym2pos)
    keep_sym = [genes[i] for i in keep]
    print(f"==> keeping {len(keep)} TRRUST genes (of {len(genes)}) as gene tokens")

    cfg = CellFoundationConfig.from_pretrained(args.model)
    m = CellFoundationModel.from_pretrained(args.model, config=cfg).eval()
    print(f"==> AIDO.Cell-10M: hidden {cfg.hidden_size}, {cfg.num_hidden_layers} layers")

    adata = sc.datasets.pbmc3k()[: args.n_cells].copy()
    adata_al, attn = align_adata(adata)
    attn_t = torch.from_numpy(attn).unsqueeze(0)

    acts, gid = [], []
    total = 0
    with torch.no_grad():
        for s in range(0, adata_al.n_obs, 8):
            xb = adata_al.X[s:s+8]
            xb = xb.toarray() if hasattr(xb, "toarray") else xb
            inp = preprocess_counts(xb, device="cpu")
            am = attn_t.repeat(inp.shape[0], 1)
            am = torch.cat([am, torch.ones((inp.shape[0], 2))], dim=1)
            out = m(input_ids=inp, attention_mask=am, output_hidden_states=True)
            hid = out.hidden_states[args.layer][:, keep, :]        # [b, |keep|, H]
            for b in range(hid.shape[0]):
                acts.append(hid[b].cpu().numpy()); gid.extend(keep_sym)
                total += len(keep)
            if total >= args.max_tokens:
                break
    X = np.concatenate(acts, 0); gid = np.array(gid)
    print(f"==> residual {X.shape} at layer {args.layer}")

    sae, stats, log = train_sae(X, SAEConfig(d_sae=args.d_sae, k=args.k, epochs=40),
                                device="cpu", verbose=False)
    F = feature_activations(sae, X, stats, device="cpu")
    print(f"    SAE FVU {log.history[-1]['fvu']:.3f}, dead {log.history[-1]['dead_frac']:.3f}")

    uniq = np.array(sorted(set(gid))); idx = {g: i for i, g in enumerate(uniq)}
    G = np.zeros((len(uniq), F.shape[1]), np.float32); c = np.zeros(len(uniq))
    for row, g in zip(F, gid):
        G[idx[g]] += row; c[idx[g]] += 1
    G /= np.maximum(c[:, None], 1)

    smap = {g: g for g in uniq}                                    # genes are symbols
    gs = load_trrust_genesets(args.trrust).filter_by_size(10)
    gs.add("TF_list", {l.strip().upper() for l in open(args.tfs)})
    ann = annotate_axes(G, list(uniq), gs, smap, min_geneset_size=10)
    ann.to_csv(os.path.join(args.out, f"sae_L{args.layer}_annotations.csv"), index=False)
    top = top_annotations_per_axis(ann, top=1).rename(columns={"axis": "feature"}).sort_values(
        "signed_strength", ascending=False)
    print(f"==> strongest residual-SAE feature→regulon links (layer {args.layer}):")
    print(top.head(18).to_string(index=False))
    print(f"\n  features with clean detector (strength>0.6): {(top['signed_strength']>0.6).sum()} / {len(top)}")
    print(f"  [compare: L2 input-embedding best regulon strength was ~0.4-0.58]")


if __name__ == "__main__":
    main()
