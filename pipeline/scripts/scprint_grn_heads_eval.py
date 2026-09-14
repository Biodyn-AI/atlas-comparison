#!/usr/bin/env python
"""Evaluate each (layer, head) attention slice of scPRINT vs TRRUST regulons —
is there a head specialised for regulatory edges? Reads the per-head adjacency
[genes, genes, n_layers*n_heads] from scprint_grn_heads.py.

    conda activate bae
    python scripts/scprint_grn_heads_eval.py --nlayers 8 --nheads 4
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adj", default="outputs/scprint/grn_adj_perhead.npz")
    ap.add_argument("--genes", default="outputs/scprint/grn_perhead_genes.txt")
    ap.add_argument("--biomart", default="external/scprint_data/biomart_pos.parquet")
    ap.add_argument("--trrust", default="external/single_cell_mechinterp/external/networks/trrust_human.tsv")
    ap.add_argument("--nlayers", type=int, default=8)
    ap.add_argument("--nheads", type=int, default=4)
    ap.add_argument("--min-targets", type=int, default=5)
    ap.add_argument("--out", default="outputs/scprint")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score
    from mechaudit.annotate.genesets import load_trrust_genesets

    adj = np.load(args.adj)["adj"]  # [G, G, H]
    genes = [l.strip() for l in open(args.genes) if l.strip()]
    G = len(genes); H = adj.shape[2]
    print(f"==> per-head adjacency {adj.shape}; {G} genes, {H} (layer,head) slices")

    bm = pd.read_parquet(args.biomart)
    ens2sym = {e: str(s).upper() for e, s in bm["hgnc_symbol"].items()}
    syms = [ens2sym.get(g, "").upper() for g in genes]
    sym2idx = {s: i for i, s in enumerate(syms) if s}
    gs = load_trrust_genesets(args.trrust)
    regulons = {k.split(":", 1)[1]: v for k, v in gs.sets.items() if k.startswith("regulon:")}

    # TFs present with enough targets
    tf_eval = []
    for tf, targets in regulons.items():
        if tf not in sym2idx:
            continue
        i = sym2idx[tf]
        y = np.array([(syms[j] in targets) and (j != i) for j in range(G)], dtype=int)
        if y.sum() >= args.min_targets:
            tf_eval.append((i, y))
    print(f"==> {len(tf_eval)} evaluable TFs")

    head_auroc = np.full(H, np.nan)
    for h in range(H):
        A = np.abs(adj[:, :, h])
        aurocs = []
        for i, y in tf_eval:
            mask = np.arange(G) != i
            try:
                aurocs.append(roc_auc_score(y[mask], A[i][mask]))
            except ValueError:
                pass
        if aurocs:
            head_auroc[h] = float(np.mean(aurocs))

    best = int(np.nanargmax(head_auroc))
    print(f"==> per-head mean AUROC: min {np.nanmin(head_auroc):.3f}, "
          f"mean {np.nanmean(head_auroc):.3f}, MAX {np.nanmax(head_auroc):.3f} "
          f"(slice {best} = layer {best // args.nheads}, head {best % args.nheads})")
    np.save(os.path.join(args.out, "grn_head_auroc.npy"), head_auroc)

    # heatmap layers × heads (layer-major ordering assumed)
    grid = head_auroc[: args.nlayers * args.nheads].reshape(args.nlayers, args.nheads)
    fig, ax = plt.subplots(figsize=(5.5, 6))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=0.5 - np.nanmax(np.abs(grid - 0.5)),
                   vmax=0.5 + np.nanmax(np.abs(grid - 0.5)), aspect="auto")
    for l in range(args.nlayers):
        for hh in range(args.nheads):
            ax.text(hh, l, f"{grid[l, hh]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xlabel("head"); ax.set_ylabel("layer")
    ax.set_xticks(range(args.nheads)); ax.set_yticks(range(args.nlayers))
    ax.set_title("scPRINT: per-(layer,head) attention→GRN\nmean AUROC vs TRRUST regulons")
    fig.colorbar(im, ax=ax, label="mean AUROC")
    fig.tight_layout()
    p = os.path.join(args.out, "grn_head_auroc_heatmap.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print("==> wrote", p)


if __name__ == "__main__":
    main()
