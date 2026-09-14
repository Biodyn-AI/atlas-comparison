#!/usr/bin/env python
"""scPRINT Layer-3 (SAE) step 2: train a TopK SAE on the captured per-gene-token
residual, aggregate feature activations per gene, and annotate the gene-level
features against TRRUST regulons / the TF list (reusing annotate_axes — the same
logic as the Layer-2 SVD axes). Runs in env `bae` (torch + mechaudit).

    conda activate bae
    python scripts/scprint_sae.py --layer 4
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--biomart", default="external/scprint_data/biomart_pos.parquet")
    ap.add_argument("--trrust", default="external/single_cell_mechinterp/external/networks/trrust_human.tsv")
    ap.add_argument("--tfs", default="external/scprint_data/TFs.txt")
    ap.add_argument("--expansion", type=int, default=8)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out", default="outputs/scprint")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    from mechaudit.sae.topk_sae import SAEConfig, train_sae, feature_activations
    from mechaudit.annotate.genesets import (load_trrust_genesets, annotate_axes,
                                             top_annotations_per_axis)

    X = np.load(os.path.join(args.out, f"residual_L{args.layer}.npz"))["X"]
    gid = [l.strip() for l in open(os.path.join(args.out, f"residual_L{args.layer}_genes.txt"))]
    print(f"==> residual {X.shape}; {len(set(gid))} unique genes")

    d_sae = args.expansion * X.shape[1]
    print(f"==> training TopK SAE (d_sae {d_sae}, k {args.k})")
    sae, stats, log = train_sae(X, SAEConfig(d_sae=d_sae, k=args.k, epochs=args.epochs),
                                device="cpu", verbose=False)
    F = feature_activations(sae, X, stats, device="cpu")
    print(f"    final FVU {log.history[-1]['fvu']:.3f}, dead {log.history[-1]['dead_frac']:.3f}")

    # aggregate feature activations per gene -> G [n_unique_genes, n_features]
    gid = np.asarray(gid)
    uniq = np.array(sorted(set(gid)))
    idx = {g: i for i, g in enumerate(uniq)}
    G = np.zeros((len(uniq), F.shape[1]), dtype=np.float32)
    cnt = np.zeros(len(uniq))
    for row, g in zip(F, gid):
        G[idx[g]] += row; cnt[idx[g]] += 1
    G /= np.maximum(cnt[:, None], 1)
    print(f"==> per-gene feature matrix {G.shape}")

    # annotate features vs TRRUST regulons + TF list (reuse the SVD-axis annotator)
    bm = pd.read_parquet(args.biomart)
    smap = {e: str(s) for e, s in bm["hgnc_symbol"].items()}
    gs = load_trrust_genesets(args.trrust).filter_by_size(10)
    gs.add("TF_list", {l.strip().upper() for l in open(args.tfs)})
    ann = annotate_axes(G, list(uniq), gs, smap, min_geneset_size=10)
    ann.to_csv(os.path.join(args.out, f"sae_L{args.layer}_annotations.csv"), index=False)
    top = top_annotations_per_axis(ann, top=1).rename(columns={"axis": "feature"})
    top = top.sort_values("signed_strength", ascending=False)
    print(f"==> {ann['axis'].nunique()} features annotated; strongest feature→geneset links:")
    print(top.head(20).to_string(index=False))
    print(f"\n  features with a clean detector (strength>0.6): "
          f"{(top['signed_strength']>0.6).sum()} / {len(top)}")
    print("==> Layer-3 SAE annotations written to", args.out)


if __name__ == "__main__":
    main()
