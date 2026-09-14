#!/usr/bin/env python
"""Error bars: repeat the 6-way comparison over N seeds (each varies the
train/test protein split AND every dictionary's init) at a fixed model/layer, and
report mean ± std of concept-detection selectivity per dictionary, split by concept
type (all / functional / AA-property).

Activations are extracted ONCE for all proteins; each seed resamples the split.

    conda activate bae
    python scripts/bae_seeds.py --model esm2-35m --layer 11 --seeds 5
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mechaudit.utils.io import ensure_dir, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="esm2-35m")
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--json", default="data/proteins/annotated_human.json")
    ap.add_argument("--n-seqs", type=int, default=380)
    ap.add_argument("--test-frac", type=float, default=0.35)
    ap.add_argument("--max-tokens", type=int, default=13000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--expansion", type=int, default=6)
    ap.add_argument("--steps", type=int, default=500)
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
    from mechaudit.sae.sae_variants import (ReLUL1Config, train_relu_l1,
                                            JumpReLUConfig, train_jumprelu, sae_features)
    from mechaudit.sae.linear_ae import LinAEConfig, train_linear_ae, linear_ae_features, pca_features
    from mechaudit.sae.compare import compare_many

    out_dir = ensure_dir(args.out)
    seqs, labs = load_annotated_proteins(args.json)
    seqs, labs = seqs[:args.n_seqs], labs[:args.n_seqs]

    # extract ONCE for all proteins at the target layer
    ex = HFActivationExtractor(args.model, kind="protein").load()
    print(f"==> {args.model} layer {args.layer}: extracting all {len(seqs)} proteins once")
    data = ex.extract(seqs, layer=args.layer, max_tokens=args.max_tokens)
    X, seq_id, pos, tokens = data["acts"], data["seq_id"], data["pos"], data["tokens"]
    uniq = np.unique(seq_id)
    print(f"==> pooled activations {X.shape} over {len(uniq)} proteins")

    per_seed = []
    for s in range(args.seeds):
        rng = np.random.default_rng(1000 + s)
        test_ids = set(rng.choice(uniq, size=int(len(uniq) * args.test_frac), replace=False).tolist())
        te_mask = np.array([sid in test_ids for sid in seq_id])
        tr_mask = ~te_mask
        Xtr, Xte = X[tr_mask], X[te_mask]
        d_sae = args.expansion * Xtr.shape[1]

        sae, _ = train_bilinear(Xtr, BAEConfig(expansion=args.expansion, steps=args.steps,
                                               lr=0.005, seed=s), verbose=False)
        F_bae = bilinear_features(sae, Xte)
        topk, st, _ = train_sae(Xtr, SAEConfig(d_sae=d_sae, k=args.k, epochs=20, seed=s),
                                device="cpu", verbose=False)
        F_topk = feature_activations(topk, Xte, st, device="cpu")
        jr, jst = train_jumprelu(Xtr, JumpReLUConfig(d_sae=d_sae, seed=s))
        F_jump = sae_features(jr, Xte, jst, device="cpu")
        rl, rst = train_relu_l1(Xtr, ReLUL1Config(d_sae=d_sae, seed=s))
        F_relu = sae_features(rl, Xte, rst, device="cpu")
        lae, _ = train_linear_ae(Xtr, LinAEConfig(d_sae=d_sae, seed=s), device="cpu")
        F_lae = linear_ae_features(lae, Xte, device="cpu")
        F_pca = pca_features(Xtr, Xte)

        labels = functional_labels(seq_id[te_mask], pos[te_mask],
                                   [labs[i] for i in range(len(labs))])
        labels.update(residue_labels(tokens[te_mask], include_identity=False))
        res = compare_many({"BAE": F_bae, "JumpReLU": F_jump, "TopK_SAE": F_topk,
                            "ReLU_L1": F_relu, "Linear_AE": F_lae, "PCA": F_pca},
                           labels, k_match=args.k)
        per_seed.append(res["per_concept"])
        print(f"  seed {s}: " + "  ".join(f"{c}={res['per_concept'][c].mean():.3f}"
                                          for c in res["per_concept"].columns))

    # align on concepts present in all seeds
    common = set(per_seed[0].index)
    for df in per_seed[1:]:
        common &= set(df.index)
    common = sorted(common)
    dicts = list(per_seed[0].columns)
    fn = [c for c in common if c.startswith("fn:")]
    pr = [c for c in common if c.startswith("prop:")]

    # stack: [seed, concept, dict]
    stack = np.stack([df.loc[common, dicts].values for df in per_seed])  # [S, C, D]

    def agg(rows_idx):
        sel = np.array([common.index(c) for c in rows_idx])
        per_seed_mean = stack[:, sel, :].mean(axis=1)   # [S, D]
        return per_seed_mean.mean(0), per_seed_mean.std(0)

    groups = {"all": common, "functional": fn, "property": pr}
    summary = {}
    for g, idx in groups.items():
        m, sd = agg(idx)
        summary[g] = {d: [round(float(m[i]), 3), round(float(sd[i]), 3)] for i, d in enumerate(dicts)}

    save_json(os.path.join(out_dir, f"seeds_{args.model}_L{args.layer}.json"),
              {"n_seeds": args.seeds, "summary": summary})
    print("\n===== mean ± std over", args.seeds, "seeds (concept mean selectivity) =====")
    hdr = "  ".join(f"{d:>10}" for d in dicts)
    print(f"{'group':12} {hdr}")
    for g in groups:
        row = "  ".join(f"{summary[g][d][0]:.3f}±{summary[g][d][1]:.2f}" for d in dicts)
        print(f"{g:12} {row}")

    # bar chart with error bars
    colors = {"BAE": "#d1495b", "JumpReLU": "#8338ec", "TopK_SAE": "#30638e",
              "ReLU_L1": "#3a86ff", "Linear_AE": "#edae49", "PCA": "#66a182"}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, g in zip(axes, ["all", "functional", "property"]):
        m = [summary[g][d][0] for d in dicts]; sd = [summary[g][d][1] for d in dicts]
        ax.bar(range(len(dicts)), m, yerr=sd, capsize=4,
               color=[colors[d] for d in dicts])
        ax.set_xticks(range(len(dicts))); ax.set_xticklabels(dicts, rotation=40, ha="right")
        ax.set_title(f"{g} concepts"); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("mean best-detector selectivity (L0=32)")
    fig.suptitle(f"BAE vs SAEs vs linear — {args.model} layer {args.layer}, "
                 f"{args.seeds} seeds (mean ± std)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(out_dir, f"seeds_{args.model}_L{args.layer}.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print("==> wrote", p)


if __name__ == "__main__":
    main()
