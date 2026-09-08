#!/usr/bin/env python
"""Layer-Explorer data (Igor's per-layer feature map) for ALL depth-matched layers of a model.
For each layer, from the SAVED SAE + activations: (1) a 2D UMAP of the alive features by decoder
direction (cosine) — the scatter; (2) co-activation edges (Pearson corr of feature activations) for
module colouring; (3) SVD-alignment flag per feature (max |cos| to top SVD axes > 0.5); (4) firing
frequency. Genes/concepts/modules are attached locally. Deps: torch + numpy + (umap-learn | sklearn).

    python layer_explorer.py --model UCE --out out_ts3
-> out_ts3/explorer/UCE_explorer.json  { layers: {L: {n_alive, dead, features:[{id,x,y,freq,svd}], edges}} }
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np


def _norm01(coords):
    x = (coords[:, 0] - coords[:, 0].min()) / max(np.ptp(coords[:, 0]), 1e-9)
    y = (coords[:, 1] - coords[:, 1].min()) / max(np.ptp(coords[:, 1]), 1e-9)
    return x, y


def embed_umap(V):
    try:
        import umap
        return umap.UMAP(n_components=2, metric="cosine", n_neighbors=15, min_dist=0.1,
                         random_state=0).fit_transform(V)
    except Exception:
        from sklearn.decomposition import PCA
        print("  [umap unavailable -> PCA]", flush=True)
        return PCA(n_components=2, random_state=0).fit_transform(V)


def embed_tsne(V):
    try:
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, metric="cosine", init="pca", random_state=0,
                    perplexity=min(30, max(5, len(V) // 20))).fit_transform(V)
    except Exception:
        from sklearn.decomposition import PCA
        print("  [tsne unavailable -> PCA]", flush=True)
        return PCA(n_components=2, random_state=0).fit_transform(V)


def main():
    from common.sae import TopKSAE
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="out_ts3")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--min-corr", type=float, default=0.15)
    ap.add_argument("--max-edges", type=int, default=9000)
    ap.add_argument("--svd-cos", type=float, default=0.5)
    ap.add_argument("--sample", type=int, default=120000)
    ap.add_argument("--batch", type=int, default=65536)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdir = os.path.join(args.out, args.model)
    rng = np.random.default_rng(0)
    layers = sorted(int(re.search(r"sae_L(\d+)\.pt", p).group(1)) for p in glob.glob(f"{mdir}/sae_L*.pt"))
    out = {"model": args.model, "layers": {}}
    for L in layers:
        LL = f"{L:02d}"
        ck = torch.load(f"{mdir}/sae_L{LL}.pt", map_location="cpu"); cfg = ck["cfg"]
        d_model = int(cfg["d_model"]); d_sae = int(cfg["expansion"]) * d_model; k = int(cfg["k"])
        sae = TopKSAE(d_model, d_sae, k).to(dev); sae.load_state_dict(ck["state_dict"]); sae.eval()
        A = np.load(f"{mdir}/layer_{LL}_activations.npy", mmap_mode="r"); N = A.shape[0]

        # streaming corr accumulators
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
        corr = (cov / torch.sqrt(torch.outer(var, var))).cpu().numpy(); np.fill_diagonal(corr, 0)

        # decoder directions -> 2D embedding (alive only)
        W = sae.W_dec.weight.detach().cpu().numpy()          # [d_model, d_sae]
        W = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-9)
        Va = W[:, alive].T                                   # [n_alive, d_model]
        cx, cy = _norm01(embed_umap(Va))                     # UMAP
        tx, ty = _norm01(embed_tsne(Va))                     # t-SNE (same directions, for comparison)

        # SVD alignment
        idx = rng.choice(N, min(args.sample, N), replace=False)
        Xc = np.asarray(A[idx], dtype=np.float64); Xc -= Xc.mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        maxcos = np.abs(Vt @ W).max(0)                       # [d_sae]
        svd_al = (maxcos[alive] > args.svd_cos).astype(int)

        # co-activation edges among alive (top-k, dedup, cap)
        aset = set(alive.tolist()); cand = {}
        for i in alive:
            row = corr[i].copy(); row[list(set(range(d_sae)) - aset)] = -1
            for j in np.argpartition(-row, args.topk)[:args.topk]:
                c = float(corr[i, j])
                if c < args.min_corr or j == i:
                    continue
                a, b = (int(i), int(j)) if i < j else (int(j), int(i))
                if (a, b) not in cand or c > cand[(a, b)]:
                    cand[(a, b)] = c
        edges = sorted(cand.items(), key=lambda kv: -kv[1])[:args.max_edges]

        out["layers"][str(L)] = {
            "n_alive": int(len(alive)), "dead": int((freq == 0).sum()),
            "d_sae": d_sae, "n_svd_aligned": int(svd_al.sum()),
            "features": [{"id": int(f), "x": round(float(cx[i]), 4), "y": round(float(cy[i]), 4),
                          "tx": round(float(tx[i]), 4), "ty": round(float(ty[i]), 4),
                          "freq": round(float(freq[f]), 6), "svd": int(svd_al[i])} for i, f in enumerate(alive)],
            "edges": [[a, b, round(c, 3)] for (a, b), c in edges],
        }
        print(f"  L{L}: alive={len(alive)} dead={int((freq==0).sum())} svd_aligned={int(svd_al.sum())} "
              f"edges={len(edges)}", flush=True)
    od = os.path.join(args.out, "explorer"); os.makedirs(od, exist_ok=True)
    p = os.path.join(od, f"{args.model}_explorer.json")
    json.dump(out, open(p, "w"))
    print(f"==> {p} ({os.path.getsize(p)//1024} KB)", flush=True)


if __name__ == "__main__":
    main()
