#!/usr/bin/env python
"""Visualise bilinear-SAE eigendecomposition manifolds on ESM-2 protein
activations: for chosen residue concepts, find the best-detecting bilinear
latent, eigendecompose its form, project residues onto the salient subspace and
plot the geometry (coloured by concept membership). Also one composite 3-D
manifold.

    conda activate bae
    python scripts/bae_manifold.py --n-seqs 110 --steps 500
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mechaudit.utils.io import ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esm2-8m")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--fasta", default="data/proteins/swissprot_human_sample.fasta")
    ap.add_argument("--n-seqs", type=int, default=110)
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--expansion", type=int, default=8)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--out", default="outputs/bae")
    args = ap.parse_args()

    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mechaudit.models.hf_wrapper import HFActivationExtractor
    from mechaudit.data.sequences import read_fasta
    from mechaudit.annotate.protein_labels import residue_labels
    from mechaudit.sae.bae_train import BAEConfig, train_bilinear, bilinear_features
    from mechaudit.audit.bae_manifold import (best_latent_for_concept,
                                              composite_weights_for_concept, project)

    out_dir = ensure_dir(args.out)
    seqs = read_fasta(args.fasta, max_seqs=args.n_seqs)
    ex = HFActivationExtractor(args.model, kind="protein").load()
    data = ex.extract(seqs, layer=args.layer, max_tokens=args.max_tokens)
    X, tokens = data["acts"], data["tokens"]
    print(f"==> activations {X.shape}")

    sae, _ = train_bilinear(X, BAEConfig(expansion=args.expansion, steps=args.steps,
                                         lr=0.005), verbose=True)
    Xn = sae.normalize(torch.as_tensor(X, dtype=torch.float32)).numpy()
    F = bilinear_features(sae, X)
    labels = residue_labels(tokens)

    # --- per-latent 2-D manifolds for a few residue concepts -------------
    concepts = ["prop:aromatic", "prop:cysteine", "prop:proline",
                "prop:glycine", "prop:positive", "prop:hydrophobic"]
    concepts = [c for c in concepts if c in labels]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, c in zip(axes.ravel(), concepts):
        y = labels[c]
        j, auroc = best_latent_for_concept(F, y)
        M = sae.latent_form(j)
        _, evecs = sae.eigenspace(M, k=2)          # rank-2 form -> 2 axes
        coords = project(Xn, evecs)                # [N,2]
        ax.scatter(coords[~y, 0], coords[~y, 1], s=3, c="lightgray", alpha=0.4, label="other")
        ax.scatter(coords[y, 0], coords[y, 1], s=6, c="crimson", alpha=0.7, label=c.split(":")[1])
        ax.set_title(f"{c}  (latent {j}, AUROC {auroc:.2f})", fontsize=10)
        ax.set_xlabel("eigvec 1"); ax.set_ylabel("eigvec 2"); ax.legend(fontsize=8)
    fig.suptitle(f"Bilinear-latent manifolds — {args.model} layer {args.layer} "
                 f"(residues coloured by concept)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p1 = os.path.join(out_dir, f"manifolds_{args.model}_L{args.layer}.png")
    fig.savefig(p1, dpi=130); plt.close(fig)
    print("==> wrote", p1)

    # --- composite 3-D manifold for aromatic residues --------------------
    target = "prop:aromatic" if "prop:aromatic" in labels else concepts[0]
    y = labels[target]
    w, idx = composite_weights_for_concept(F, y, top=12)
    M = sae.composite_form(torch.as_tensor(w))
    _, evecs = sae.eigenspace(M, k=3)
    coords = project(Xn, evecs)
    aa = np.array([t.upper() for t in tokens])
    fig = plt.figure(figsize=(8, 7)); ax = fig.add_subplot(111, projection="3d")
    ax.scatter(coords[~y, 0], coords[~y, 1], coords[~y, 2], s=3, c="lightgray", alpha=0.25)
    for a_res, col in [("F", "tab:blue"), ("W", "tab:green"), ("Y", "tab:red"), ("H", "tab:orange")]:
        m = aa == a_res
        if m.any():
            ax.scatter(coords[m, 0], coords[m, 1], coords[m, 2], s=10, c=col, label=a_res)
    ax.set_title(f"Composite manifold — {target} (top-12 latents)\n{args.model} layer {args.layer}")
    ax.legend()
    p2 = os.path.join(out_dir, f"manifold3d_{args.model}_L{args.layer}.png")
    fig.savefig(p2, dpi=130); plt.close(fig)
    print("==> wrote", p2)


if __name__ == "__main__":
    main()
