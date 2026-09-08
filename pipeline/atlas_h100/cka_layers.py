#!/usr/bin/env python
"""Within-model layer x layer CKA — how each model's representation evolves across its OWN depth.
Uses the saved position-level activations (layer_XX_activations.npy), which are position-aligned across
layers (one forward), so linear CKA between any two layers is valid without re-running the model. A
subsample of shared positions is used for speed. Output: per model an [n_layers x n_layers] CKA matrix
(1 on the diagonal); block structure = stages where the representation is stable, off-diagonal drops =
abrupt transitions.
    python cka_layers.py --all --out out_alllayers --sample 30000    -> out_alllayers/cka_layers.json
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np

MODELS = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="out_alllayers")
    ap.add_argument("--sample", type=int, default=30000)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    targets = MODELS if args.all else [args.model]
    res = {}
    for m in targets:
        mdir = os.path.join(args.out, m)
        layers = sorted(int(re.search(r"layer_(\d+)_activations", p).group(1))
                        for p in glob.glob(f"{mdir}/layer_*_activations.npy"))
        if len(layers) < 2:
            print(f"{m}: <2 activation layers, skip", flush=True); continue
        A0 = np.load(f"{mdir}/layer_{layers[0]:02d}_activations.npy", mmap_mode="r")
        N = A0.shape[0]; K = min(args.sample, N)
        idx = np.sort(np.random.default_rng(0).choice(N, K, replace=False))
        # load + center each layer's subsample on GPU
        X = {}
        for L in layers:
            a = np.load(f"{mdir}/layer_{L:02d}_activations.npy", mmap_mode="r")
            t = torch.as_tensor(np.asarray(a[idx]), dtype=torch.float32, device=dev)
            t -= t.mean(0, keepdim=True)
            X[L] = t
        # pairwise linear CKA
        n = len(layers); Mx = [[1.0] * n for _ in range(n)]
        norm = {L: torch.linalg.norm(X[L].T @ X[L]) for L in layers}
        for i in range(n):
            for j in range(i + 1, n):
                Li, Lj = layers[i], layers[j]
                num = torch.linalg.norm(X[Li].T @ X[Lj]) ** 2
                c = float(num / (norm[Li] * norm[Lj] + 1e-12))
                Mx[i][j] = Mx[j][i] = round(c, 3)
        res[m] = {"layers": layers, "cka": Mx}
        # adjacent-layer drop (min consecutive CKA = sharpest transition)
        adj = [Mx[i][i + 1] for i in range(n - 1)]
        print(f"{m}: {n} layers, adjacent CKA {min(adj):.2f}-{max(adj):.2f} "
              f"(sharpest transition L{layers[int(np.argmin(adj))]}->L{layers[int(np.argmin(adj))+1]})", flush=True)
        for L in layers:
            del X[L]
        torch.cuda.empty_cache() if dev == "cuda" else None
    p = os.path.join(args.out, "cka_layers.json"); json.dump(res, open(p, "w"))
    print(f"==> {p}", flush=True)


if __name__ == "__main__":
    main()
