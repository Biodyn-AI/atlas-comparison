#!/usr/bin/env python
"""DNA arm (track 3): BAE vs JumpReLU/TopK/ReLU-L1 SAE vs Linear AE vs PCA on
DNABERT-6 activations, scored on functional (coding/promoter/enhancer/intergenic)
and compositional (GC/CpG/…) concepts. Same fair protocol as the protein arm
(held-out sequences, density-matched L0=32).

    conda activate bae-dna
    python scripts/bae_dna.py --layer 7
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
    ap.add_argument("--model", default="dnabert-6")
    ap.add_argument("--layer", type=int, default=7)
    ap.add_argument("--classes", default="coding,intergenic,promoter,enhancer")
    ap.add_argument("--n-per", type=int, default=120)
    ap.add_argument("--test-frac", type=float, default=0.35)
    ap.add_argument("--max-tokens-train", type=int, default=6000)
    ap.add_argument("--max-tokens-test", type=int, default=5000)
    ap.add_argument("--expansion", type=int, default=4)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--out", default="outputs/bae")
    args = ap.parse_args()

    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mechaudit.models.hf_wrapper import HFActivationExtractor
    from mechaudit.data.dna import load_dna_sequences, functional_labels_dna
    from mechaudit.annotate.dna_labels import kmer_labels
    from mechaudit.sae.bae_train import BAEConfig, train_bilinear, bilinear_features
    from mechaudit.sae.topk_sae import SAEConfig, train_sae, feature_activations
    from mechaudit.sae.sae_variants import (ReLUL1Config, train_relu_l1,
                                            JumpReLUConfig, train_jumprelu, sae_features)
    from mechaudit.sae.linear_ae import LinAEConfig, train_linear_ae, linear_ae_features, pca_features
    from mechaudit.sae.compare import compare_many

    out_dir = ensure_dir(args.out)
    classes = args.classes.split(",")
    seqs, seq_class = load_dna_sequences(classes, n_per=args.n_per)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(seqs))
    seqs = [seqs[i] for i in order]; seq_class = [seq_class[i] for i in order]
    n_test = int(len(seqs) * args.test_frac)
    te_seqs, te_cls = seqs[:n_test], seq_class[:n_test]
    tr_seqs = seqs[n_test:]
    print(f"==> {len(tr_seqs)} train / {len(te_seqs)} test sequences; classes {classes}")

    ex = HFActivationExtractor(args.model, kind="dna").load()
    print(f"==> {args.model}: {ex.n_layers} layers, hidden {ex.d_hidden}; layer {args.layer}")
    tr = ex.extract(tr_seqs, layer=args.layer, max_len=512, max_tokens=args.max_tokens_train)
    te = ex.extract(te_seqs, layer=args.layer, max_len=512, max_tokens=args.max_tokens_test)
    Xtr, Xte = tr["acts"], te["acts"]
    d_sae = args.expansion * Xtr.shape[1]
    print(f"==> train {Xtr.shape}, test {Xte.shape}, d_sae {d_sae}")

    print("==> BAE"); sae, _ = train_bilinear(Xtr, BAEConfig(expansion=args.expansion,
                                                             steps=args.steps, lr=0.005), verbose=True)
    F_bae = bilinear_features(sae, Xte)
    print("==> TopK"); topk, st, _ = train_sae(Xtr, SAEConfig(d_sae=d_sae, k=args.k, epochs=20),
                                               device="cpu", verbose=False)
    F_topk = feature_activations(topk, Xte, st, device="cpu")
    print("==> JumpReLU"); jr, jst = train_jumprelu(Xtr, JumpReLUConfig(d_sae=d_sae))
    F_jump = sae_features(jr, Xte, jst, device="cpu")
    print("==> ReLU-L1"); rl, rst = train_relu_l1(Xtr, ReLUL1Config(d_sae=d_sae))
    F_relu = sae_features(rl, Xte, rst, device="cpu")
    print("==> Linear AE"); lae, _ = train_linear_ae(Xtr, LinAEConfig(d_sae=d_sae), device="cpu")
    F_lae = linear_ae_features(lae, Xte, device="cpu")
    print("==> PCA"); F_pca = pca_features(Xtr, Xte)

    labels = functional_labels_dna(te["seq_id"], te_cls)
    labels.update(kmer_labels(te["tokens"]))
    dicts = {"BAE": F_bae, "JumpReLU": F_jump, "TopK_SAE": F_topk,
             "ReLU_L1": F_relu, "Linear_AE": F_lae, "PCA": F_pca}
    res = compare_many(dicts, labels, k_match=args.k)

    df = res["per_concept"]
    df.to_csv(os.path.join(out_dir, f"dna_{args.model}_L{args.layer}.csv"))
    save_json(os.path.join(out_dir, f"dna_summary_{args.model}_L{args.layer}.json"), res["summary"])
    print("\n===== per-concept selectivity (held-out, L0=32) =====")
    print(df.round(3).to_string())
    print("\n===== SUMMARY =====")
    for name, m in res["summary"].items():
        print(f"  {name:11s} concept_mean_sel={m['concept_mean_sel']}  clean={m['concept_clean_frac']}")

    # bar chart
    fn = [c for c in df.index if str(c).startswith("fn:")]
    pr = [c for c in df.index if str(c).startswith("prop:")]
    df = df.loc[fn + pr]
    labs2 = [c.split(":", 1)[1] for c in df.index]
    cols = list(dicts.keys())
    colors = {"BAE": "#d1495b", "JumpReLU": "#8338ec", "TopK_SAE": "#30638e",
              "ReLU_L1": "#3a86ff", "Linear_AE": "#edae49", "PCA": "#66a182"}
    n = len(df); x = np.arange(n); w = 0.8 / len(cols)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.6), 5.4))
    for i, c in enumerate(cols):
        ax.bar(x + i * w - 0.4 + w / 2, df[c].values, w, label=c, color=colors[c])
    ax.axhline(0.6, ls="--", lw=1, c="gray", alpha=0.6)
    ax.axvline(len(fn) - 0.5, ls=":", c="black", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(labs2, rotation=55, ha="right", fontsize=9)
    ax.set_ylabel("best feature-detector selectivity |AUROC-0.5|·2"); ax.set_ylim(0, 1.08)
    ax.set_title(f"DNA — BAE vs SAEs vs linear — {args.model} layer {args.layer} (held-out, L0=32)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(out_dir, f"dna_{args.model}_L{args.layer}.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print("==> wrote", p)


if __name__ == "__main__":
    main()
