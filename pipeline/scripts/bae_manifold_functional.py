#!/usr/bin/env python
"""Eigendecomposition manifolds of bilinear-SAE latents for FUNCTIONAL / structural
residue concepts (UniProt). For each concept: the best-detecting bilinear latent,
residues projected onto its eigenspace. Plus a 3-D composite manifold coloured by
secondary structure (helix / strand / coil).

    conda activate bae
    python scripts/bae_manifold_functional.py --model esm2-8m --layer 5
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
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--json", default="data/proteins/annotated_human.json")
    ap.add_argument("--n-seqs", type=int, default=160)
    ap.add_argument("--max-tokens", type=int, default=7000)
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
    from mechaudit.data.uniprot import load_annotated_proteins, functional_labels
    from mechaudit.sae.bae_train import BAEConfig, train_bilinear, bilinear_features
    from mechaudit.audit.bae_manifold import (best_latent_for_concept,
                                              composite_weights_for_concept, project)

    out_dir = ensure_dir(args.out)
    seqs, labs = load_annotated_proteins(args.json)
    idx = np.random.default_rng(1).permutation(len(seqs))[:args.n_seqs]
    seqs = [seqs[i] for i in idx]; labs = [labs[i] for i in idx]

    ex = HFActivationExtractor(args.model, kind="protein").load()
    data = ex.extract(seqs, layer=args.layer, max_tokens=args.max_tokens)
    X = data["acts"]
    labels = functional_labels(data["seq_id"], data["pos"], labs)
    print(f"==> activations {X.shape}")

    sae, _ = train_bilinear(X, BAEConfig(expansion=args.expansion, steps=args.steps,
                                         lr=0.005), verbose=True)
    Xn = sae.normalize(torch.as_tensor(X, dtype=torch.float32)).numpy()
    F = bilinear_features(sae, X)

    # --- per-latent 2-D manifolds for functional concepts ----------------
    concepts = ["fn:transmembrane", "fn:disulfide", "fn:binding",
                "fn:helix", "fn:strand", "fn:modified"]
    concepts = [c for c in concepts if c in labels and labels[c].sum() >= 30]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, c in zip(axes.ravel(), concepts):
        y = labels[c]
        j, auroc = best_latent_for_concept(F, y)
        _, evecs = sae.eigenspace(sae.latent_form(j), k=2)
        co = project(Xn, evecs)
        ax.scatter(co[~y, 0], co[~y, 1], s=3, c="lightgray", alpha=0.35, label="other")
        ax.scatter(co[y, 0], co[y, 1], s=7, c="crimson", alpha=0.75, label=c.split(":")[1])
        ax.set_title(f"{c.split(':')[1]}  (latent {j}, AUROC {auroc:.2f}, n={int(y.sum())})",
                     fontsize=10)
        ax.set_xlabel("eigvec 1"); ax.set_ylabel("eigvec 2"); ax.legend(fontsize=8)
    fig.suptitle(f"Bilinear-latent manifolds — functional concepts — {args.model} layer {args.layer}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p1 = os.path.join(out_dir, f"manifolds_functional_{args.model}_L{args.layer}.png")
    fig.savefig(p1, dpi=130); plt.close(fig); print("==> wrote", p1)

    # --- 3-D secondary-structure manifold (helix/strand/coil) ------------
    w_h, _ = composite_weights_for_concept(F, labels["fn:helix"], top=10)
    w_s, _ = composite_weights_for_concept(F, labels["fn:strand"], top=10)
    w = np.abs(w_h) + np.abs(w_s)
    _, evecs = sae.eigenspace(sae.composite_form(torch.as_tensor(w.astype(np.float32))), k=3)
    co = project(Xn, evecs)
    helix, strand = labels["fn:helix"], labels["fn:strand"]
    coil = ~(helix | strand | labels["fn:turn"])
    fig = plt.figure(figsize=(8, 7)); ax = fig.add_subplot(111, projection="3d")
    ax.scatter(co[coil, 0], co[coil, 1], co[coil, 2], s=3, c="lightgray", alpha=0.25, label="coil")
    ax.scatter(co[helix, 0], co[helix, 1], co[helix, 2], s=7, c="tab:red", alpha=0.6, label="helix")
    ax.scatter(co[strand, 0], co[strand, 1], co[strand, 2], s=7, c="tab:blue", alpha=0.6, label="strand")
    ax.set_title(f"Secondary-structure composite manifold\n{args.model} layer {args.layer}")
    ax.legend()
    p2 = os.path.join(out_dir, f"manifold3d_ss_{args.model}_L{args.layer}.png")
    fig.savefig(p2, dpi=130); plt.close(fig); print("==> wrote", p2)


if __name__ == "__main__":
    main()
