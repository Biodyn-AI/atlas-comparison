#!/usr/bin/env python
"""Visualise a BAE-vs-TopK per-concept comparison CSV as a grouped bar chart
(concept-side best-detector selectivity). Highlights functional concepts.

    python scripts/bae_viz.py --csv outputs/bae/functional_esm2-35m_L10.csv
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(args.csv, index_col=0)
    # order: functional concepts first (fn:), then property (prop:)
    fn = [c for c in df.index if str(c).startswith("fn:")]
    pr = [c for c in df.index if str(c).startswith("prop:")]
    df = df.loc[fn + pr]
    labels = [c.split(":", 1)[1] for c in df.index]

    cols = [c for c in ("bilinear_matched", "topk", "bilinear_full") if c in df.columns]
    colors = {"bilinear_matched": "#d1495b", "topk": "#30638e", "bilinear_full": "#edae49"}
    names = {"bilinear_matched": "bilinear @ top-k", "topk": "TopK", "bilinear_full": "bilinear (full)"}

    n = len(df); x = np.arange(n); w = 0.8 / len(cols)
    fig, ax = plt.subplots(figsize=(max(9, n * 0.55), 5.2))
    for i, c in enumerate(cols):
        ax.bar(x + i * w - 0.4 + w / 2, df[c].values, w, label=names[c], color=colors[c])
    ax.axhline(0.6, ls="--", lw=1, c="gray", alpha=0.7)
    ax.axvline(len(fn) - 0.5, ls=":", c="black", alpha=0.5)
    ax.text(len(fn) / 2 - 0.5, 1.02, "functional / structural", ha="center", fontsize=9, color="#555")
    ax.text(len(fn) + len(pr) / 2 - 0.5, 1.02, "amino-acid property", ha="center", fontsize=9, color="#555")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=9)
    ax.set_ylabel("best feature-detector selectivity  |AUROC-0.5|·2")
    ax.set_ylim(0, 1.08)
    ax.set_title(args.title or f"BAE vs TopK — {os.path.basename(args.csv)}")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    out = args.out or args.csv.replace(".csv", ".png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
