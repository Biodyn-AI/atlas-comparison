#!/usr/bin/env python
"""Comparative SAE atlas — step 1: produce a model-agnostic feature_catalog.json
(top-20 genes per SAE feature) from a captured residual + TopK SAE, matching the
bio-sae schema so every model (ours, the ingested prior, new H100 ones) joins on the
SAME annotation vocabulary downstream.

    conda activate bae
    python scripts/atlas_catalog.py --model scPRINT \
        --residual outputs/scprint/residual_L4.npz --genes outputs/scprint/residual_L4_genes.txt \
        --layer 4 --symbol-map biomart
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--residual", required=True, help="npz with key X [n_tokens, d_model]")
    ap.add_argument("--genes", required=True, help="txt: one gene id per token (row-aligned to X)")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--d-sae", type=int, default=2048)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--symbol-map", default="none", choices=["none", "biomart"],
                    help="map gene ids -> HGNC symbols (biomart for scPRINT ENSG)")
    ap.add_argument("--out", default="outputs/atlas")
    args = ap.parse_args()

    import numpy as np
    from mechaudit.sae.topk_sae import SAEConfig, train_sae, feature_activations

    X = np.load(args.residual)["X"].astype(np.float32)
    gid = np.array([l.strip() for l in open(args.genes)])
    assert len(gid) == len(X), f"genes {len(gid)} != tokens {len(X)}"

    if args.symbol_map == "biomart":
        import pandas as pd
        bm = pd.read_parquet(f"{BASE}/external/scprint_data/biomart_pos.parquet")
        e2s = {e: str(s).upper() for e, s in bm["hgnc_symbol"].items()}
        gid = np.array([e2s.get(g, g) for g in gid])
    else:
        gid = np.array([g.upper() for g in gid])

    sae, stats, log = train_sae(X, SAEConfig(d_sae=args.d_sae, k=args.k, epochs=args.epochs),
                                device="cpu", verbose=False)
    F = feature_activations(sae, X, stats, device="cpu")           # [n_tokens, d_sae]
    fvu = log.history[-1]["fvu"]; dead = log.history[-1]["dead_frac"]

    # per-gene mean activation, then top-N genes per feature
    uniq = np.array(sorted(set(gid))); gi = {g: i for i, g in enumerate(uniq)}
    G = np.zeros((len(uniq), F.shape[1]), np.float32); c = np.zeros(len(uniq))
    for row, g in zip(F, gid):
        G[gi[g]] += row; c[gi[g]] += 1
    G /= np.maximum(c[:, None], 1)
    freq = (F > 0).mean(0)                                          # activation frequency
    alive = freq > 0

    feats = {}
    for f in range(F.shape[1]):
        if not alive[f]:
            continue
        order = np.argsort(G[:, f])[::-1][:args.top]
        feats[str(f)] = {
            "top_genes": [str(uniq[i]) for i in order],
            "max_activation": float(G[:, f].max()),
            "activation_frequency": float(freq[f]),
        }

    os.makedirs(f"{args.out}/{args.model}", exist_ok=True)
    catalog = {"model": args.model, "layer": args.layer, "d_model": int(X.shape[1]),
               "d_sae": args.d_sae, "k": args.k, "n_alive": int(alive.sum()),
               "fvu": float(fvu), "dead_frac": float(dead), "n_genes": int(len(uniq)),
               "features": feats}
    path = f"{args.out}/{args.model}/feature_catalog_L{args.layer}.json"
    with open(path, "w") as fh:
        json.dump(catalog, fh)
    print(f"==> {args.model} L{args.layer}: {alive.sum()} alive features, FVU {fvu:.3f}, "
          f"{len(uniq)} genes -> {path}")
    # peek: a few features' top genes
    for f in list(feats)[:3]:
        print(f"   feature {f}: {feats[f]['top_genes'][:8]}")


if __name__ == "__main__":
    main()
