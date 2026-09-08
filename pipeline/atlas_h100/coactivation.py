#!/usr/bin/env python
"""TRUE feature co-activation graph (Igor-style) from the SAVED SAE + activations.
No re-run of the model: loads out/<MODEL>/sae_L<LL>.pt (exact same SAE as the catalog,
so feature ids map 1:1 to feature_catalog_L<LL>.json) and out/<MODEL>/layer_<LL>_activations.npy,
streams SAE encoding, accumulates feature-feature Pearson correlation over all positions,
and exports a compact edge list. Community detection + layout are done locally (tiny download).

Deps: torch + numpy only (uses common/sae.py).
    python coactivation.py --model AIDO --layer 8
    python coactivation.py --model UCE  --layer 3
    python coactivation.py --model tGPT --layer 8
-> out/coact/coact_<MODEL>_L<LL>.json   ({features:[{id,freq}], edges:[[i,j,corr]]})
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
from common.sae import TopKSAE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--topk", type=int, default=10, help="neighbours kept per feature")
    ap.add_argument("--min-corr", type=float, default=0.15)
    ap.add_argument("--max-edges", type=int, default=9000)
    ap.add_argument("--batch", type=int, default=65536)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    LL = f"{args.layer:02d}"
    mdir = os.path.join(args.out, args.model)
    sae_p = os.path.join(mdir, f"sae_L{LL}.pt")
    act_p = os.path.join(mdir, f"layer_{LL}_activations.npy")
    assert os.path.exists(sae_p), f"missing {sae_p} (re-run run_model.py --model {args.model})"
    assert os.path.exists(act_p), f"missing {act_p} (re-run run_model.py --model {args.model})"

    ck = torch.load(sae_p, map_location="cpu")
    cfg = ck["cfg"]
    d_model = int(cfg["d_model"]); d_sae = int(cfg["expansion"]) * d_model; k = int(cfg["k"])
    sae = TopKSAE(d_model, d_sae, k).to(dev)
    sae.load_state_dict(ck["state_dict"]); sae.eval()
    A = np.load(act_p, mmap_mode="r")
    N = A.shape[0]
    print(f"{args.model} L{args.layer}: {N} positions x {d_model}d, d_sae={d_sae}", flush=True)

    # streaming accumulators (float64 on device) for Pearson corr between feature columns
    S1 = torch.zeros(d_sae, dtype=torch.float64, device=dev)          # sum f
    S2 = torch.zeros(d_sae, d_sae, dtype=torch.float64, device=dev)   # sum f_i f_j
    S0 = torch.zeros(d_sae, dtype=torch.float64, device=dev)          # nonzero count (freq)
    with torch.no_grad():
        for s in range(0, N, args.batch):
            xb = torch.as_tensor(np.asarray(A[s:s + args.batch]), dtype=torch.float32, device=dev)
            H = sae.encode(xb).double()                              # [b, d_sae]
            S1 += H.sum(0)
            S2 += H.T @ H
            S0 += (H > 0).sum(0)
            if (s // args.batch) % 10 == 0:
                print(f"   {min(s+args.batch,N)}/{N}", flush=True)

    freq = (S0 / N).cpu().numpy()
    alive = np.where(freq > 0)[0]
    mean = S1 / N
    cov = S2 / N - torch.outer(mean, mean)
    var = torch.clamp(torch.diagonal(cov), min=1e-12)
    denom = torch.sqrt(torch.outer(var, var))
    corr = (cov / denom).cpu().numpy()
    np.fill_diagonal(corr, 0.0)

    # keep top-k neighbours per alive feature above threshold, dedup i<j, global cap
    aset = set(alive.tolist())
    cand = {}
    for i in alive:
        row = corr[i].copy()
        row[list(set(range(d_sae)) - aset)] = -1  # only alive partners
        nb = np.argpartition(-row, args.topk)[:args.topk]
        for j in nb:
            c = float(corr[i, j])
            if c < args.min_corr or j == i:
                continue
            a, b = (int(i), int(j)) if i < j else (int(j), int(i))
            key = (a, b)
            if key not in cand or c > cand[key]:
                cand[key] = c
    edges = sorted(cand.items(), key=lambda kv: -kv[1])[:args.max_edges]
    out = {
        "model": args.model, "layer": args.layer, "n_positions": int(N),
        "kind": "co-activation (Pearson corr of SAE feature activations across all gene-token positions)",
        "features": [{"id": int(f), "freq": round(float(freq[f]), 6)} for f in alive],
        "edges": [[a, b, round(c, 3)] for (a, b), c in edges],
    }
    od = os.path.join(args.out, "coact"); os.makedirs(od, exist_ok=True)
    p = os.path.join(od, f"coact_{args.model}_L{LL}.json")
    json.dump(out, open(p, "w"))
    print(f"==> {len(out['features'])} alive features, {len(out['edges'])} edges -> {p} "
          f"({os.path.getsize(p)//1024} KB)", flush=True)


if __name__ == "__main__":
    main()
