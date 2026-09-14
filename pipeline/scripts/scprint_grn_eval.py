#!/usr/bin/env python
"""Evaluate scPRINT's attention→GRN adjacency (Layer 1) against TRRUST regulons.
For each TF present in the inferred network that has a TRRUST regulon, rank the
other genes by the attention edge weight and score recovery of the TF's true
targets (AUROC + AUPRC). Reports the mean vs the random baseline (0.5 / prior).

    conda activate bae     # any env with numpy/pandas/sklearn
    python scripts/scprint_grn_eval.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adj", default="outputs/scprint/grn_adjacency.npz")
    ap.add_argument("--genes", default="outputs/scprint/grn_genes.txt")
    ap.add_argument("--biomart", default="external/scprint_data/biomart_pos.parquet")
    ap.add_argument("--trrust", default="external/single_cell_mechinterp/external/networks/trrust_human.tsv")
    ap.add_argument("--min-targets", type=int, default=5)
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score, average_precision_score
    from mechaudit.annotate.genesets import load_trrust_genesets

    adj = np.load(args.adj)["adj"]
    genes = [l.strip() for l in open(args.genes) if l.strip()]
    n = len(genes)
    print(f"==> adjacency {adj.shape}, {n} genes")

    # Ensembl -> symbol
    bm = pd.read_parquet(args.biomart)
    ens2sym = {e: str(s).upper() for e, s in bm["hgnc_symbol"].items()}
    syms = [ens2sym.get(g, "").upper() for g in genes]
    sym2idx = {s: i for i, s in enumerate(syms) if s}

    gs = load_trrust_genesets(args.trrust)
    regulons = {k.split(":", 1)[1]: v for k, v in gs.sets.items() if k.startswith("regulon:")}

    # symmetric magnitude for ranking (attention adjacency may be near-symmetric)
    A = np.abs(adj)
    rows = []
    for tf, targets in regulons.items():
        if tf not in sym2idx:
            continue
        i = sym2idx[tf]
        # label each OTHER gene: is it a TRRUST target of this TF?
        y = np.array([(syms[j] in targets) and (j != i) for j in range(n)], dtype=int)
        if y.sum() < args.min_targets:
            continue
        score = A[i].copy(); score[i] = -np.inf  # exclude self
        mask = np.arange(n) != i
        try:
            auroc = roc_auc_score(y[mask], score[mask])
            auprc = average_precision_score(y[mask], score[mask])
        except ValueError:
            continue
        rows.append({"tf": tf, "n_targets_present": int(y.sum()),
                     "auroc": auroc, "auprc": auprc,
                     "prior": float(y[mask].mean())})

    df = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    df.to_csv("outputs/scprint/grn_trrust_eval.csv", index=False)
    print(f"==> evaluable TFs (in network, ≥{args.min_targets} targets): {len(df)}")
    if len(df):
        print(df.round(3).head(20).to_string(index=False))
        print("\n===== SUMMARY (attention→GRN vs TRRUST) =====")
        print(f"  mean AUROC : {df['auroc'].mean():.3f}  (random 0.5)")
        print(f"  median AUROC: {df['auroc'].median():.3f}")
        print(f"  mean AUPRC : {df['auprc'].mean():.3f}  (mean prior {df['prior'].mean():.3f})")
        print(f"  AUPRC lift over prior: {(df['auprc']/df['prior']).mean():.2f}x")
        print(f"  TFs with AUROC>0.5: {(df['auroc']>0.5).mean():.0%}")


if __name__ == "__main__":
    main()
