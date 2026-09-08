#!/usr/bin/env python
"""SVD-vs-SAE superposition test (matches Igor's 'X% of features invisible to SVD'). For a model
layer, computes the SVD/PCA of the residual, then measures how many SAE decoder directions are
NOT aligned with the top SVD axes (max |cos| < thresh = 'novel', i.e. in superposition). Also the
matched-sparsity variance comparison (top-k SVD reconstruction vs the SAE's k-active reconstruction).
Uses the SAVED SAE (sae_L{L}.pt) + activations (layer_{L}_activations.npy) — no model re-run.

    python svd_vs_sae.py --model UCE --layer 16 --out out_ts3
-> out_ts3/svd/UCE_L16_svd.json  { pct_novel, mean_maxcos, svd_var_at_k, sae_var, per-component var }
"""
from __future__ import annotations
import argparse, json, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", default="out_ts3")
    ap.add_argument("--cos-thresh", type=float, default=0.7)   # Igor's alignment cutoff
    ap.add_argument("--n-svd", type=int, default=0, help="top SVD components to compare against (0 = all d_model)")
    ap.add_argument("--sample", type=int, default=150000)
    args = ap.parse_args()

    import torch
    from common.sae import TopKSAE
    LL = f"{args.layer:02d}"
    mdir = os.path.join(args.out, args.model)
    A = np.load(os.path.join(mdir, f"layer_{LL}_activations.npy"), mmap_mode="r")
    ck = torch.load(os.path.join(mdir, f"sae_L{LL}.pt"), map_location="cpu")
    cfg = ck["cfg"]; d_model = int(cfg["d_model"]); k = int(cfg["k"])
    sae = TopKSAE(d_model, int(cfg["expansion"]) * d_model, k)
    sae.load_state_dict(ck["state_dict"])
    sae_var = float(ck.get("stats", {}).get("var_explained", 1 - ck.get("stats", {}).get("fvu", 0)))

    # SVD of the (centered) residual on a subsample
    N = A.shape[0]
    rng = np.random.default_rng(0)
    idx = rng.choice(N, min(args.sample, N), replace=False)
    X = np.asarray(A[idx], dtype=np.float64)
    mu = X.mean(0); Xc = X - mu
    # right singular vectors = principal directions in d_model space
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)      # Vt [min(n,d), d]
    var = (S ** 2) / (S ** 2).sum()                        # variance explained per component
    K = args.n_svd if args.n_svd > 0 else Vt.shape[0]
    comps = Vt[:K]                                         # [K, d_model]

    # SAE decoder feature directions (unit-norm columns of W_dec)
    W = sae.W_dec.weight.detach().numpy()                 # [d_model, d_sae]
    W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-9)
    # max |cos| of each feature to the top-K SVD axes
    cos = np.abs(comps @ W)                                # [K, d_sae]
    maxcos = cos.max(0)                                    # [d_sae]
    novel = maxcos < args.cos_thresh
    pct_novel = float(novel.mean())

    # matched-sparsity variance: top-k SVD cumulative variance vs SAE var_explained
    svd_var_at_k = float(var[:k].sum())
    svd_var_full = float(var.sum())

    out = {"model": args.model, "layer": args.layer, "d_model": d_model, "d_sae": W.shape[1],
           "k": k, "cos_thresh": args.cos_thresh, "n_svd_components": int(K),
           "pct_novel": round(pct_novel, 4), "n_novel": int(novel.sum()),
           "mean_maxcos": round(float(maxcos.mean()), 4),
           "median_maxcos": round(float(np.median(maxcos)), 4),
           "svd_var_at_k": round(svd_var_at_k, 4), "sae_var": round(sae_var, 4),
           "var_top10": [round(float(v), 4) for v in var[:10]]}
    od = os.path.join(args.out, "svd"); os.makedirs(od, exist_ok=True)
    p = os.path.join(od, f"{args.model}_L{LL}_svd.json")
    json.dump(out, open(p, "w"))
    print(f"{args.model} L{args.layer}: novel={100*pct_novel:.1f}% (cos<{args.cos_thresh}) "
          f"mean_maxcos={out['mean_maxcos']} | SVD var@k={svd_var_at_k:.3f} vs SAE var={sae_var:.3f} -> {p}", flush=True)


if __name__ == "__main__":
    main()
