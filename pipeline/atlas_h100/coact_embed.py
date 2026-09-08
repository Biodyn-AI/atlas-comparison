#!/usr/bin/env python
"""Co-activation layout for the Layer Explorer (so its clusters match the co-activation MODULES, the way
Igor's per-layer explorer looks). Instead of embedding features by decoder direction, embed them by how
they FIRE TOGETHER: stream the Pearson correlation of SAE feature activations across gene-token positions,
then reduce the distance D = 1 - corr with BOTH UMAP and t-SNE (same structure, two reductions → the
UMAP↔t-SNE toggle becomes a robustness check on the module clusters).

Outputs compact per-model-per-layer {ids, x,y (coact-UMAP), tx,ty (coact-t-SNE)} to merge into the slim.
    python coact_embed.py --all --out out_ts3     -> out_ts3/explorer/coact_ts3.json
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np

MODELS = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]


def norm01(c):
    x = (c[:, 0] - c[:, 0].min()) / max(np.ptp(c[:, 0]), 1e-9)
    y = (c[:, 1] - c[:, 1].min()) / max(np.ptp(c[:, 1]), 1e-9)
    return x, y


def embed_umap(D):
    import umap
    return umap.UMAP(n_components=2, metric="precomputed", n_neighbors=15, min_dist=0.12,
                     random_state=0).fit_transform(D)


def embed_tsne(D):
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, metric="precomputed", init="random", random_state=0,
                perplexity=min(30, max(5, len(D) // 20))).fit_transform(D)


def main():
    import torch
    from common.sae import TopKSAE
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="out_ts3")
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--max-feat", type=int, default=4500, help="cap alive features embedded (memory of precomputed reducers)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    targets = MODELS if args.all else [args.model]
    res = {}
    for m in targets:
        mdir = os.path.join(args.out, m)
        layers = sorted(int(re.search(r"sae_L(\d+)\.pt", p).group(1)) for p in glob.glob(f"{mdir}/sae_L*.pt"))
        per = {}
        for L in layers:
            LL = f"{L:02d}"
            ck = torch.load(f"{mdir}/sae_L{LL}.pt", map_location="cpu"); cfg = ck["cfg"]
            d_model = int(cfg["d_model"]); d_sae = int(cfg["expansion"]) * d_model
            sae = TopKSAE(d_model, d_sae, int(cfg["k"])).to(dev); sae.load_state_dict(ck["state_dict"]); sae.eval()
            A = np.load(f"{mdir}/layer_{LL}_activations.npy", mmap_mode="r"); N = A.shape[0]
            S1 = torch.zeros(d_sae, dtype=torch.float64, device=dev)
            S2 = torch.zeros(d_sae, d_sae, dtype=torch.float64, device=dev)
            S0 = torch.zeros(d_sae, dtype=torch.float64, device=dev)
            with torch.no_grad():
                for s in range(0, N, args.batch):
                    xb = torch.as_tensor(np.asarray(A[s:s + args.batch]), dtype=torch.float32, device=dev)
                    H = sae.encode(xb).double(); S1 += H.sum(0); S2 += H.T @ H; S0 += (H > 0).sum(0)
            freq = (S0 / N).cpu().numpy(); alive = np.where(freq > 0)[0]
            mean = S1 / N; cov = S2 / N - torch.outer(mean, mean)
            var = torch.clamp(torch.diagonal(cov), min=1e-12)
            corr = (cov / torch.sqrt(torch.outer(var, var))).cpu().numpy()
            if len(alive) > args.max_feat:                       # keep the most active (memory cap)
                keep = alive[np.argsort(-freq[alive])[:args.max_feat]]
                alive = np.sort(keep)
            C = corr[np.ix_(alive, alive)]; np.fill_diagonal(C, 1.0)
            D = np.clip(1.0 - C, 0.0, 2.0).astype(np.float64); D = (D + D.T) / 2; np.fill_diagonal(D, 0.0)
            ux, uy = norm01(embed_umap(D)); tx, ty = norm01(embed_tsne(D))
            per[str(L)] = {"ids": [int(i) for i in alive],
                           "x": [round(float(v) * 100, 1) for v in ux], "y": [round(float(v) * 100, 1) for v in uy],
                           "tx": [round(float(v) * 100, 1) for v in tx], "ty": [round(float(v) * 100, 1) for v in ty]}
            print(f"  {m} L{L}: co-activation embed on {len(alive)} features", flush=True)
        res[m] = per
    od = os.path.join(args.out, "explorer"); os.makedirs(od, exist_ok=True)
    p = os.path.join(od, "coact_ts3.json"); json.dump(res, open(p, "w"))
    print(f"==> {p} ({os.path.getsize(p)//1024} KB)", flush=True)


if __name__ == "__main__":
    main()
