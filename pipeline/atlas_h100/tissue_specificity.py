#!/usr/bin/env python
"""Per-feature tissue specificity for the TS-3-tissue atlas. For a model, re-runs the forward
pass (cheap: no SAE training, no annotation), encodes each layer's residual with the SAVED SAE
(sae_L{L}.pt from the run — features match the catalog 1:1), and accumulates each SAE feature's
MEAN activation within immune / kidney / lung cells (from obs['compartment']). Answers: which
features (and how many) are tissue-specific, and to which tissue.

    python tissue_specificity.py --model UCE --corpus external/perturb/tabula_3tissue_6k.h5ad --out out_ts3
-> out_ts3/tissue/UCE_tissue.json  { layer: {tissues, means[d_sae][3], spec[d_sae], pref[d_sae]}, summary }

Runs the same adapters as run_model; needs the model's out_ts3/<M>/sae_L*.pt present.
"""
from __future__ import annotations
import argparse, importlib, json, os
import numpy as np


def main():
    from run_model import ADAPTERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(ADAPTERS))
    ap.add_argument("--corpus", default="external/perturb/tabula_3tissue_6k.h5ad")
    ap.add_argument("--out", default="out_ts3")
    ap.add_argument("--tissue-key", default="compartment")
    ap.add_argument("--spec-thresh", type=float, default=0.5, help="feature is tissue-specific if a tissue holds >= this share of its total mean activation")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    import torch, scanpy as sc
    from common.sae import TopKSAE

    mod, cls = ADAPTERS[args.model]
    adapter = getattr(importlib.import_module(mod), cls)()
    adapter.load(device=args.device)
    adata = sc.read_h5ad(args.corpus)

    # load saved SAEs for the adapter's layers
    mdir = os.path.join(args.out, args.model)
    saes = {}
    for L in adapter.layers:
        p = os.path.join(mdir, f"sae_L{L:02d}.pt")
        assert os.path.exists(p), f"missing {p} (run run_model.py --model {args.model} --out {args.out} first)"
        ck = torch.load(p, map_location="cpu"); cfg = ck["cfg"]
        sae = TopKSAE(int(cfg["d_model"]), int(cfg["expansion"]) * int(cfg["d_model"]), int(cfg["k"]))
        sae.load_state_dict(ck["state_dict"]); saes[L] = sae.eval().to(args.device)
    d_sae = {L: saes[L].d_sae for L in adapter.layers}
    print(f"{args.model}: layers {list(adapter.layers)}, d_sae {d_sae}", flush=True)

    # accumulators: per layer, per tissue -> sum of feature activations + position count
    tissues = None
    acc, cnt = {}, {}

    processed = None
    for acts, syms, cell_ids in adapter.iter_activations(adata, batch_size=args.batch_size):
        if processed is None:
            processed = adapter.processed_obs
            tvals = processed[args.tissue_key].astype(str).values
            tissues = sorted(set(tvals))
            tidx = {t: i for i, t in enumerate(tissues)}
            for L in adapter.layers:
                acc[L] = np.zeros((d_sae[L], len(tissues)), np.float64)
                cnt[L] = np.zeros(len(tissues), np.float64)
        tiss_pos = np.array([tidx[tvals[c]] for c in cell_ids])          # tissue index per position
        for L in adapter.layers:
            with torch.no_grad():
                X = torch.as_tensor(acts[L], dtype=torch.float32, device=args.device)
                F = saes[L].encode(X).cpu().numpy()                       # [pos, d_sae]
            for ti in range(len(tissues)):
                m = tiss_pos == ti
                if m.any():
                    acc[L][:, ti] += F[m].sum(0); cnt[L][ti] += m.sum()

    out = {"model": args.model, "tissues": tissues, "layers": {}}
    for L in adapter.layers:
        means = acc[L] / np.maximum(cnt[L][None, :], 1)                   # [d_sae, 3] mean activation per tissue
        tot = means.sum(1)
        alive = tot > 0
        share = np.zeros_like(means); share[alive] = means[alive] / tot[alive, None]
        spec = share.max(1)                                              # 0.33 (uniform) .. 1 (one tissue)
        pref = means.argmax(1)
        # summary: tissue-specific features (spec >= thresh) per tissue
        summary = {}
        for ti, t in enumerate(tissues):
            sel = alive & (spec >= args.spec_thresh) & (pref == ti)
            summary[t] = int(sel.sum())
        out["layers"][str(L)] = {
            "means": np.round(means, 5).tolist(),
            "spec": np.round(spec, 4).tolist(),
            "pref": pref.tolist(),
            "n_alive": int(alive.sum()),
            "tissue_specific": summary,
            "counts": cnt[L].tolist(),
        }
        print(f"  L{L}: alive={int(alive.sum())} tissue-specific(>= {args.spec_thresh}) "
              + " ".join(f"{t}={summary[t]}" for t in tissues), flush=True)

    od = os.path.join(args.out, "tissue"); os.makedirs(od, exist_ok=True)
    p = os.path.join(od, f"{args.model}_tissue.json")
    json.dump(out, open(p, "w"))
    print(f"==> {p} ({os.path.getsize(p)//1024} KB)", flush=True)


if __name__ == "__main__":
    main()
