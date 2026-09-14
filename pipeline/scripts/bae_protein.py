#!/usr/bin/env python
"""BAE track — FAIR comparison: bilinear SAE vs TopK SAE on ESM-2 protein
activations. SAEs are trained on one set of proteins and probed on HELD-OUT
proteins; bilinear is also evaluated TopK-masked (matched density). Metrics:
concept-side (does a detector exist?) and feature-side (are features monosemantic?).

    conda activate bae
    python scripts/bae_protein.py --model esm2-8m --layer 3 --n-seqs 260
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
    ap.add_argument("--model", default="esm2-8m")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--fasta", default="data/proteins/swissprot_human_sample.fasta")
    ap.add_argument("--n-seqs", type=int, default=260)
    ap.add_argument("--test-frac", type=float, default=0.35)
    ap.add_argument("--max-tokens-train", type=int, default=7000)
    ap.add_argument("--max-tokens-test", type=int, default=5000)
    ap.add_argument("--expansion", type=int, default=8)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--out", default="outputs/bae")
    args = ap.parse_args()

    import numpy as np
    from mechaudit.models.hf_wrapper import HFActivationExtractor
    from mechaudit.data.sequences import read_fasta
    from mechaudit.annotate.protein_labels import residue_labels
    from mechaudit.sae.bae_train import BAEConfig, train_bilinear, bilinear_features
    from mechaudit.sae.topk_sae import SAEConfig, train_sae, feature_activations
    from mechaudit.sae.compare import compare

    out_dir = ensure_dir(args.out)
    seqs = read_fasta(args.fasta, max_seqs=args.n_seqs)
    rng = np.random.default_rng(0); rng.shuffle(seqs)
    n_test = int(len(seqs) * args.test_frac)
    test_seqs, train_seqs = seqs[:n_test], seqs[n_test:]
    print(f"==> {len(train_seqs)} train / {len(test_seqs)} test sequences")

    ex = HFActivationExtractor(args.model, kind="protein").load()
    layer = args.layer if args.layer is not None else ex.n_layers // 2
    print(f"==> {args.model}: {ex.n_layers} layers, hidden {ex.d_hidden}; layer {layer}")
    tr = ex.extract(train_seqs, layer=layer, max_tokens=args.max_tokens_train)
    te = ex.extract(test_seqs, layer=layer, max_tokens=args.max_tokens_test)
    Xtr, Xte, toks_te = tr["acts"], te["acts"], te["tokens"]
    print(f"==> train acts {Xtr.shape}, test acts {Xte.shape}")

    d_sae = args.expansion * Xtr.shape[1]
    print(f"==> BAE (expansion {args.expansion}, lat {d_sae}, Muon lr {args.lr}, {args.steps} steps)")
    sae, _ = train_bilinear(Xtr, BAEConfig(expansion=args.expansion, steps=args.steps,
                                           lr=args.lr), verbose=True)
    F_bae = bilinear_features(sae, Xte)

    print(f"==> TopK (d_sae {d_sae}, k {args.k})")
    topk, stats, tlog = train_sae(Xtr, SAEConfig(d_sae=d_sae, k=args.k, epochs=25),
                                  device="cpu", verbose=False)
    F_topk = feature_activations(topk, Xte, stats, device="cpu")
    print(f"    TopK train FVU={tlog.history[-1]['fvu']:.3f} dead={tlog.history[-1]['dead_frac']:.3f}")

    labels = residue_labels(toks_te)
    print(f"==> {len(labels)} concepts; evaluating on HELD-OUT test tokens")
    res = compare(F_bae, F_topk, labels, k_match=args.k)

    res["per_concept"].to_csv(os.path.join(out_dir, f"faircompare_{args.model}_L{layer}.csv"))
    save_json(os.path.join(out_dir, f"fairsummary_{args.model}_L{layer}.json"), res["summary"])

    print("\n===== Per-concept best-detector selectivity (held-out) =====")
    print(res["per_concept"].round(3).to_string())
    print("\n===== FAIR SUMMARY (L0 = active latents/token) =====")
    for dic, m in res["summary"].items():
        print(f"  {dic:18s} L0={m['L0']:>7}  concept[mean_sel={m['concept_mean_sel']}, "
              f"clean={m['concept_clean_frac']}]  feature[mean_sel={m['feature_mean_sel']}, "
              f"clean={m['feature_clean_frac']}]")
    print("\nRead: 'bilinear_matched' vs 'topk' is the density-matched (same L0) verdict;")
    print("feature_* penalises dense/redundant dictionaries (monosemanticity from the feature side).")


if __name__ == "__main__":
    main()
