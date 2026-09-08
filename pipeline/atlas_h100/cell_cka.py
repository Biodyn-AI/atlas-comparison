#!/usr/bin/env python
"""Representational similarity (linear CKA) between the seven models. Each model's residual is
mean-pooled per cell into a cell embedding [n_cells, d_model] (over its own gene tokens), at each
depth-matched layer; then pairwise linear CKA is computed at MATCHED relative depth (0/25/50/75/100%).
CKA handles different d_model. Re-runs the forward (no SAE); saves each model's embeddings so a crash
mid-run is cheap. Exports only the tiny 7x7-per-depth matrix.

    python cell_cka.py --model AIDO --out out_ts3      # per model (compute + cache embeddings)
    python cell_cka.py --cka --out out_ts3             # after all 7: compute CKA -> out_ts3/cka/cka_ts3.json
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np

MODELS = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
# per-model batch cap for a busy shared 80GB GPU (big LM heads OOM at large batch)
BATCH_CAP = {"C2S": 2, "UCE": 8, "MaxToki": 8}


def compute_embeddings(model, corpus, out, device, batch, max_cells=0, all_layers=False):
    """Per-cell mean-pooled embeddings at each layer, in BOTH spaces:
    residual (raw d_model) and SAE-feature (d_sae, encoded activations). Keyed by rel-depth index."""
    import importlib, scanpy as sc, torch
    from run_model import ADAPTERS
    from common.sae import TopKSAE
    cache = os.path.join(out, "cka", f"{model}_emb.npz")
    if os.path.exists(cache):
        try:
            keys = np.load(cache).files
            if any(k.startswith("freq_") for k in keys):
                print(f"{model}: cached with freq ({cache}) — skip", flush=True); return
            print(f"{model}: cached but no freq — recomputing", flush=True)
        except Exception:
            print(f"{model}: cache unreadable — recomputing", flush=True)
    mod, cls = ADAPTERS[model]
    adapter = getattr(importlib.import_module(mod), cls)()
    adapter.load(device=device)
    if all_layers:
        total = getattr(adapter, "n_layers", None) or (max(adapter.layers) + 1)
        adapter.layers = tuple(range(total))
        print(f"{model}: all-layers mode ({total} layers)", flush=True)
    adata = sc.read_h5ad(corpus)
    if max_cells and adata.n_obs > max_cells:
        import numpy as _np
        idx = _np.sort(_np.random.default_rng(0).choice(adata.n_obs, max_cells, replace=False))
        adata = adata[idx].copy()
        print(f"{model}: subsampled to {adata.n_obs} cells (seed 0, shared across models)", flush=True)
    n_cells = adata.n_obs
    layers = list(adapter.layers)
    mdir = os.path.join(out, model)
    # load a SAE per layer if present (skip SAE space for layers without one)
    sae = {}
    for L in layers:
        p = os.path.join(mdir, f"sae_L{L:02d}.pt")
        if os.path.exists(p):
            ck = torch.load(p, map_location=device); cfg = ck["cfg"]
            s = TopKSAE(int(cfg["d_model"]), int(cfg["expansion"]) * int(cfg["d_model"]), int(cfg["k"])).to(device)
            s.load_state_dict(ck["state_dict"]); s.eval(); sae[L] = s
    bs = min(batch, BATCH_CAP.get(model, batch))
    # accumulate per-cell sums on the GPU (index_add_ — far faster than CPU np.add.at, esp. big d_sae)
    r_sum = {L: None for L in layers}; f_sum = {L: None for L in layers}
    f_active = {L: None for L in layers}; n_pos = 0
    cnt = torch.zeros(n_cells, device=device)
    for acts, syms, cell_ids in adapter.iter_activations(adata, batch_size=bs):
        n_pos += len(cell_ids)
        cid = torch.as_tensor(np.asarray(cell_ids), dtype=torch.long, device=device)
        for L in layers:
            A = torch.as_tensor(np.asarray(acts[L]), dtype=torch.float32, device=device)
            if r_sum[L] is None:
                r_sum[L] = torch.zeros(n_cells, A.shape[1], device=device)
            r_sum[L].index_add_(0, cid, A)
            if L in sae:
                with torch.no_grad():
                    H = sae[L].encode(A)
                if f_sum[L] is None:
                    f_sum[L] = torch.zeros(n_cells, H.shape[1], device=device)
                    f_active[L] = torch.zeros(H.shape[1], device=device)
                f_sum[L].index_add_(0, cid, H.float())
                f_active[L] += (H > 0).sum(0).float()
        cnt.index_add_(0, cid, torch.ones_like(cid, dtype=torch.float32))
    c = cnt.clamp(min=1).unsqueeze(1)
    save = {"layers": np.array(layers)}
    for i, L in enumerate(layers):
        save[f"res_{i}"] = (r_sum[L] / c).cpu().numpy().astype(np.float32); r_sum[L] = None
        if f_sum[L] is not None:
            save[f"sae_{i}"] = (f_sum[L] / c).cpu().numpy().astype(np.float32); f_sum[L] = None
            save[f"freq_{i}"] = (f_active[L] / max(n_pos, 1)).cpu().numpy().astype(np.float32)
    od = os.path.join(out, "cka"); os.makedirs(od, exist_ok=True)
    np.savez(os.path.join(od, f"{model}_emb.npz"), **save)
    print(f"{model}: bs={bs} n_pos={n_pos} {sorted(k for k in save)} -> {od}/{model}_emb.npz", flush=True)


def linear_cka(X, Y):
    X = X - X.mean(0, keepdims=True); Y = Y - Y.mean(0, keepdims=True)
    xy = np.linalg.norm(X.T @ Y) ** 2
    xx = np.linalg.norm(X.T @ X); yy = np.linalg.norm(Y.T @ Y)
    return float(xy / (xx * yy + 1e-12))


def build_cka(out):
    od = os.path.join(out, "cka"); os.makedirs(od, exist_ok=True)
    embs = {}
    for p in glob.glob(f"{od}/*_emb.npz"):
        m = re.search(r"([A-Za-z0-9]+)_emb\.npz", p).group(1)
        embs[m] = dict(np.load(p))
    models = [m for m in MODELS if m in embs]
    out_json = {"models": models, "depths": ["0%", "25%", "50%", "75%", "100%"], "residual": {}, "sae": {}}
    for space, keyfn in (("residual", lambda i: f"res_{i}"), ("sae", lambda i: f"sae_{i}")):
        for r in range(5):
            k = keyfn(r)
            if not any(k in embs[m] for m in models):
                continue
            Mx = [[None] * len(models) for _ in models]
            for i, a in enumerate(models):
                for j, b in enumerate(models):
                    if k in embs[a] and k in embs[b]:
                        Mx[i][j] = round(linear_cka(embs[a][k], embs[b][k]), 3)
            out_json[space][str(r)] = Mx
            print(f"{space} rel-depth {r}: CKA done", flush=True)
    json.dump(out_json, open(f"{od}/cka_ts3.json", "w"))
    print(f"==> {od}/cka_ts3.json (residual depths={list(out_json['residual'])}, sae depths={list(out_json['sae'])})", flush=True)
    # firing frequency per feature per layer (tiny) for the annotated-vs-unannotated check
    freq = {}
    for m in models:
        e = embs[m]
        if not any(k.startswith("freq_") for k in e):
            continue
        lay = e["layers"].tolist() if "layers" in e else []
        freq[m] = {"layers": lay,
                   "freq": {k.split("_")[1]: e[k].round(6).tolist() for k in e if k.startswith("freq_")}}
    json.dump(freq, open(f"{od}/freq_ts3.json", "w"))
    print(f"==> {od}/freq_ts3.json (models with freq: {list(freq)})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--cka", action="store_true")
    ap.add_argument("--corpus", default="external/perturb/tabula_3tissue_6k.h5ad")
    ap.add_argument("--out", default="out_ts3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-cells", type=int, default=2000, help="subsample cells for CKA (0 = all)")
    ap.add_argument("--all-layers", action="store_true", help="per-cell embeddings at EVERY layer (for per-layer nonlinearity/tissue)")
    args = ap.parse_args()
    if args.cka:
        build_cka(args.out)
    else:
        compute_embeddings(args.model, args.corpus, args.out, args.device, args.batch_size,
                           args.max_cells, args.all_layers)


if __name__ == "__main__":
    main()
