#!/usr/bin/env python
"""Layer sweep: trace BAE vs TopK vs Linear AE vs PCA concept-detection selectivity
across ESM-2 depth, to locate where the quadratic form (BAE) overtakes linear PCA.

    conda activate bae
    python scripts/bae_layer_sweep.py --model esm2-35m --layers 1,3,5,7,9,11
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
    ap.add_argument("--model", default="esm2-35m")
    ap.add_argument("--layers", default="1,3,5,7,9,11")
    ap.add_argument("--json", default="data/proteins/annotated_human.json")
    ap.add_argument("--n-seqs", type=int, default=360)
    ap.add_argument("--test-frac", type=float, default=0.35)
    ap.add_argument("--max-tokens-train", type=int, default=5000)
    ap.add_argument("--max-tokens-test", type=int, default=4000)
    ap.add_argument("--expansion", type=int, default=6)
    ap.add_argument("--steps", type=int, default=350)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--out", default="outputs/bae")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mechaudit.models.hf_wrapper import HFActivationExtractor
    from mechaudit.data.uniprot import load_annotated_proteins, functional_labels
    from mechaudit.annotate.protein_labels import residue_labels
    from mechaudit.sae.bae_train import BAEConfig, train_bilinear, bilinear_features
    from mechaudit.sae.topk_sae import SAEConfig, train_sae, feature_activations
    from mechaudit.sae.linear_ae import LinAEConfig, train_linear_ae, linear_ae_features, pca_features
    from mechaudit.sae.compare import compare_many

    out_dir = ensure_dir(args.out)
    layers = [int(x) for x in args.layers.split(",")]
    seqs, labs = load_annotated_proteins(args.json)
    idx = np.random.default_rng(0).permutation(len(seqs))[:args.n_seqs]
    seqs = [seqs[i] for i in idx]; labs = [labs[i] for i in idx]
    n_test = int(len(seqs) * args.test_frac)
    tr_seqs, tr_labs = seqs[n_test:], labs[n_test:]
    te_seqs, te_labs = seqs[:n_test], labs[:n_test]

    ex = HFActivationExtractor(args.model, kind="protein").load()
    print(f"==> {args.model}: {ex.n_layers} layers; extracting ALL layers once")
    tr = ex.extract_all_layers(tr_seqs, max_tokens=args.max_tokens_train)
    te = ex.extract_all_layers(te_seqs, max_tokens=args.max_tokens_test)
    labels = functional_labels(te["seq_id"], te["pos"], te_labs)
    labels.update(residue_labels(te["tokens"], include_identity=False))
    fn_concepts = [c for c in labels if c.startswith("fn:")]

    rows = []
    for L in layers:
        Xtr, Xte = tr["acts"][L], te["acts"][L]
        d_sae = args.expansion * Xtr.shape[1]
        print(f"\n=== layer {L}: train {Xtr.shape} d_sae {d_sae} ===")
        sae, _ = train_bilinear(Xtr, BAEConfig(expansion=args.expansion, steps=args.steps,
                                               lr=0.005), verbose=False)
        F_bae = bilinear_features(sae, Xte)
        topk, st, _ = train_sae(Xtr, SAEConfig(d_sae=d_sae, k=args.k, epochs=20),
                                device="cpu", verbose=False)
        F_topk = feature_activations(topk, Xte, st, device="cpu")
        lae, _ = train_linear_ae(Xtr, LinAEConfig(d_sae=d_sae, epochs=30), device="cpu")
        F_lae = linear_ae_features(lae, Xte, device="cpu")
        F_pca = pca_features(Xtr, Xte)
        res = compare_many({"BAE": F_bae, "TopK_SAE": F_topk, "Linear_AE": F_lae, "PCA": F_pca},
                           labels, k_match=args.k)
        df = res["per_concept"]
        rec = {"layer": L}
        for c in df.columns:
            rec[f"{c}_all"] = float(df[c].mean())
            rec[f"{c}_fn"] = float(df.loc[df.index.isin(fn_concepts), c].mean())
        rows.append(rec)
        print("  " + "  ".join(f"{c}={df[c].mean():.3f}" for c in df.columns))

    sweep = pd.DataFrame(rows).set_index("layer")
    sweep.to_csv(os.path.join(out_dir, f"layersweep_{args.model}.csv"))
    print("\n===== SWEEP (mean selectivity, all concepts) =====")
    print(sweep[[c for c in sweep.columns if c.endswith("_all")]].round(3).to_string())

    # plot
    dicts = ["BAE", "TopK_SAE", "Linear_AE", "PCA"]
    colors = {"BAE": "#d1495b", "TopK_SAE": "#30638e", "Linear_AE": "#edae49", "PCA": "#66a182"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, suff, ttl in [(axes[0], "_all", "all concepts"),
                          (axes[1], "_fn", "functional/structural only")]:
        for d in dicts:
            ax.plot(sweep.index, sweep[f"{d}{suff}"], "-o", color=colors[d], label=d)
        ax.set_xlabel("ESM-2 layer"); ax.set_title(ttl); ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean best-detector selectivity (L0=32, held-out)")
    axes[0].legend()
    fig.suptitle(f"BAE vs PCA vs TopK vs Linear AE across depth — {args.model}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(out_dir, f"layersweep_{args.model}.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print("==> wrote", p)


if __name__ == "__main__":
    main()
