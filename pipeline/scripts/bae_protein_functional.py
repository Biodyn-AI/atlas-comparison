#!/usr/bin/env python
"""BAE track — the real test: bilinear vs TopK SAE on a LATE ESM-2 layer, scored
against FUNCTIONAL / STRUCTURAL residue concepts (secondary structure,
transmembrane, binding/active sites, disulfides, modifications) from UniProt.

    conda activate bae
    python scripts/bae_protein_functional.py --model esm2-35m --layer 10
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
    ap.add_argument("--layer", type=int, default=None, help="default = last-but-one")
    ap.add_argument("--json", default="data/proteins/annotated_human.json")
    ap.add_argument("--n-seqs", type=int, default=380)
    ap.add_argument("--test-frac", type=float, default=0.35)
    ap.add_argument("--max-tokens-train", type=int, default=7000)
    ap.add_argument("--max-tokens-test", type=int, default=5000)
    ap.add_argument("--expansion", type=int, default=8)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--out", default="outputs/bae")
    args = ap.parse_args()

    import numpy as np
    from mechaudit.models.hf_wrapper import HFActivationExtractor
    from mechaudit.data.uniprot import load_annotated_proteins, functional_labels
    from mechaudit.annotate.protein_labels import residue_labels
    from mechaudit.sae.bae_train import BAEConfig, train_bilinear, bilinear_features
    from mechaudit.sae.topk_sae import SAEConfig, train_sae, feature_activations
    from mechaudit.sae.compare import compare

    out_dir = ensure_dir(args.out)
    seqs, labs = load_annotated_proteins(args.json)
    order = np.random.default_rng(0).permutation(len(seqs))[:args.n_seqs]
    seqs = [seqs[i] for i in order]; labs = [labs[i] for i in order]
    n_test = int(len(seqs) * args.test_frac)
    tr_seqs, tr_labs = seqs[n_test:], labs[n_test:]
    te_seqs, te_labs = seqs[:n_test], labs[:n_test]
    print(f"==> {len(tr_seqs)} train / {len(te_seqs)} test proteins")

    ex = HFActivationExtractor(args.model, kind="protein").load()
    layer = args.layer if args.layer is not None else ex.n_layers - 2
    print(f"==> {args.model}: {ex.n_layers} layers, hidden {ex.d_hidden}; LATE layer {layer}")
    tr = ex.extract(tr_seqs, layer=layer, max_tokens=args.max_tokens_train)
    te = ex.extract(te_seqs, layer=layer, max_tokens=args.max_tokens_test)
    Xtr, Xte = tr["acts"], te["acts"]
    print(f"==> train acts {Xtr.shape}, test acts {Xte.shape}")

    d_sae = args.expansion * Xtr.shape[1]
    print(f"==> BAE (lat {d_sae}, Muon lr {args.lr}, {args.steps} steps)")
    sae, _ = train_bilinear(Xtr, BAEConfig(expansion=args.expansion, steps=args.steps,
                                           lr=args.lr), verbose=True)
    F_bae = bilinear_features(sae, Xte)

    print(f"==> TopK (d_sae {d_sae}, k {args.k})")
    topk, stats, tlog = train_sae(Xtr, SAEConfig(d_sae=d_sae, k=args.k, epochs=25),
                                  device="cpu", verbose=False)
    F_topk = feature_activations(topk, Xte, stats, device="cpu")
    print(f"    TopK train FVU={tlog.history[-1]['fvu']:.3f}")

    # functional labels (+ keep AA property labels for reference)
    labels = functional_labels(te["seq_id"], te["pos"], te_labs)
    labels.update(residue_labels(te["tokens"], include_identity=False))
    print(f"==> {len(labels)} labels (functional + property); held-out evaluation")
    res = compare(F_bae, F_topk, labels, k_match=args.k, min_pos=30)

    res["per_concept"].to_csv(os.path.join(out_dir, f"functional_{args.model}_L{layer}.csv"))
    save_json(os.path.join(out_dir, f"functional_summary_{args.model}_L{layer}.json"), res["summary"])

    fn = res["per_concept"][res["per_concept"].index.str.startswith("fn:")]
    print("\n===== FUNCTIONAL concepts: best-detector selectivity (held-out) =====")
    print(fn.round(3).to_string())
    print("\n===== full per-concept (functional + property) =====")
    print(res["per_concept"].round(3).to_string())
    print("\n===== FAIR SUMMARY =====")
    for dic, m in res["summary"].items():
        print(f"  {dic:18s} L0={m['L0']:>7}  concept[mean_sel={m['concept_mean_sel']}, "
              f"clean={m['concept_clean_frac']}]  feature[mean_sel={m['feature_mean_sel']}]")


if __name__ == "__main__":
    main()
