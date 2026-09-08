#!/usr/bin/env python
"""Add t-SNE coordinates for the Layer Explorer's alive features, to compare against the existing UMAP.
Light: reads the alive feature ids per layer from the existing full explorer json, loads the SAE decoder
directions for exactly those features (sae_L*.pt), computes a 2D t-SNE (cosine) — no activation streaming.
Outputs a tiny per-model-per-layer {id: [tx,ty]} that gets merged into the slim explorer locally.

    python tsne_coords.py --all --out out_ts3      -> out_ts3/explorer/tsne_ts3.json
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np

MODELS = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]


def norm01(c):
    x = (c[:, 0] - c[:, 0].min()) / max(np.ptp(c[:, 0]), 1e-9)
    y = (c[:, 1] - c[:, 1].min()) / max(np.ptp(c[:, 1]), 1e-9)
    return x, y


def tsne(V):
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, metric="cosine", init="pca", random_state=0,
                perplexity=min(30, max(5, len(V) // 20))).fit_transform(V)


def main():
    import torch
    from common.sae import TopKSAE
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="out_ts3")
    args = ap.parse_args()
    targets = MODELS if args.all else [args.model]
    res = {}
    for m in targets:
        expp = os.path.join(args.out, "explorer", f"{m}_explorer.json")
        if not os.path.exists(expp):
            print(f"{m}: no {expp}, skip", flush=True); continue
        exp = json.load(open(expp)); mdir = os.path.join(args.out, m)
        per = {}
        for L, ld in exp["layers"].items():
            ids = [f["id"] for f in ld["features"]]
            if len(ids) < 3:
                continue
            sp = os.path.join(mdir, f"sae_L{int(L):02d}.pt")
            ck = torch.load(sp, map_location="cpu"); cfg = ck["cfg"]
            sae = TopKSAE(int(cfg["d_model"]), int(cfg["expansion"]) * int(cfg["d_model"]), int(cfg["k"]))
            sae.load_state_dict(ck["state_dict"])
            W = sae.W_dec.weight.detach().numpy(); W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-9)
            V = W[:, ids].T                                     # [n_alive, d_model] in the SAME id order
            tx, ty = norm01(tsne(V))
            per[L] = {"ids": [int(i) for i in ids],
                      "tx": [round(float(x), 4) for x in tx], "ty": [round(float(y), 4) for y in ty]}
            print(f"  {m} L{L}: tsne on {len(ids)} features", flush=True)
        res[m] = per
    od = os.path.join(args.out, "explorer"); os.makedirs(od, exist_ok=True)
    p = os.path.join(od, "tsne_ts3.json"); json.dump(res, open(p, "w"))
    print(f"==> {p} ({os.path.getsize(p)//1024} KB)", flush=True)


if __name__ == "__main__":
    main()
