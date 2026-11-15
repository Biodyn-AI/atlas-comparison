"""Untrained-SAE control: for every layer of a model, a TopK SAE of the same shape as the real one,
initialised exactly as common/sae.py:TopKSAE (torch seed 0, Kaiming-uniform encoder, zero biases,
mu = mean activation of the layer) but NEVER trained, is applied to the same stored activations, and
the feature catalogue is built exactly as common/pipeline.py:build_catalog (per-gene mean activation,
stable descending sort, zero-score genes dropped, top 20).

usage: ATLAS_GPU_BUNDLE=<bundle> python random_sae_catalogs.py MODEL [MODEL ...]
output: $ATLAS_CACHE/random_sae/<MODEL>/feature_catalog_L<L>.json
"""
import glob, json, os, re, sys, time
import numpy as np
import torch
BUNDLE = os.path.abspath(os.environ["ATLAS_GPU_BUNDLE"])
sys.path.insert(0, BUNDLE)
from common.sae import SAECfg, TopKSAE

SRC = os.environ.get("ATLAS_FIXED_ACTIVATIONS", os.path.join(BUNDLE, "out_fixed"))
CACHE = os.environ.get("ATLAS_CACHE", os.path.join(os.path.dirname(__file__), "..", ".cache"))
OUT = os.environ.get("ATLAS_RANDOM_SAE_OUTPUT", os.path.join(CACHE, "random_sae"))
dev = "cuda"
t0 = time.time()
for m in sys.argv[1:]:
    src = f"{SRC}/{m}"; out = f"{OUT}/{m}"; os.makedirs(out, exist_ok=True)
    syms = np.load(f"{src}/gene_symbols.npy", allow_pickle=True).astype(str)
    uniq = np.array(sorted(set(syms))); gi = {g: i for i, g in enumerate(uniq)}
    sym_idx = torch.as_tensor(np.array([gi[g] for g in syms]), device=dev)
    cnt = torch.zeros(len(uniq), device=dev).index_add_(0, sym_idx, torch.ones(len(syms), device=dev))
    layers = sorted(int(re.search(r"layer_(\d+)_activations", p).group(1)) for p in glob.glob(f"{src}/layer_*_activations.npy"))
    for L in layers:
        dst = f"{out}/feature_catalog_L{L:02d}.json"
        if os.path.exists(dst):
            continue
        A = np.load(f"{src}/layer_{L:02d}_activations.npy", mmap_mode="r")
        cfg = SAECfg(d_model=A.shape[1])
        torch.manual_seed(cfg.seed)                       # the same init as train_sae (seed 0)
        sae = TopKSAE(cfg.d_model, cfg.d_sae, cfg.k).to(dev)
        X = np.asarray(A, dtype=np.float32)
        sae.mu.copy_(torch.as_tensor(X.mean(0), dtype=torch.float32, device=dev))
        sae.eval()
        G = torch.zeros(len(uniq) * cfg.d_sae, dtype=torch.float64); fired = torch.zeros(cfg.d_sae, dtype=torch.float64)
        si = sym_idx.cpu().long()
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                xb = torch.as_tensor(X[i:i + 8192], device=dev)
                h = sae.W_enc(xb - sae.mu)                       # = TopKSAE.encode, kept sparse:
                vals, idx = torch.topk(h, sae.k, dim=-1); vals = torch.relu(vals)
                vals, idx = vals.cpu().double(), idx.cpu().long()
                flat = si[i:i + 8192, None] * cfg.d_sae + idx
                G.index_add_(0, flat.reshape(-1), vals.reshape(-1))
                fired.index_add_(0, idx[vals > 0], torch.ones(int((vals > 0).sum()), dtype=torch.float64))
        G = G.view(len(uniq), cfg.d_sae).float(); cntc = cnt.cpu()
        G = (G / cntc.clamp(min=1)[:, None]).numpy(); freq = (fired / len(X)).numpy()
        feats = {}
        for f in np.where(freq > 0)[0]:
            order = np.argsort(-G[:, f], kind="stable")
            order = order[G[order, f] > 0][:20]
            feats[str(int(f))] = {"top_genes": [str(uniq[i]) for i in order], "max_activation": float(G[:, f].max()),
                                  "activation_frequency": float(freq[f])}
        json.dump({"model": m, "layer": int(L), "d_model": int(A.shape[1]), "d_sae": int(cfg.d_sae), "k": int(cfg.k),
                   "n_alive": int((freq > 0).sum()), "n_genes": int(len(uniq)), "untrained_sae": True, "features": feats},
                  open(dst, "w"))
        print(f"{m} L{L}: {len(X)} positions, {int((freq > 0).sum())}/{cfg.d_sae} alive [{time.time() - t0:.0f}s]", flush=True)
        del X, A, G
