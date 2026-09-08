"""Model-agnostic pipeline: adapter -> per-layer activations -> TopK SAE -> feature
catalog (top-20 genes/feature). One model, all requested layers. Outputs match the
bio-sae schema so every model joins on the same annotation vocabulary downstream.
"""
from __future__ import annotations
import json, os
import numpy as np
from .sae import SAECfg, train_sae, feature_activations


def extract(adapter, adata, out_dir: str, batch_size: int = 8, max_positions: int = 4_000_000, log=print):
    """Run the adapter over the corpus, store per-layer activation memmaps + gene symbols."""
    os.makedirs(out_dir, exist_ok=True)
    buffers = {L: [] for L in adapter.layers}
    syms_all = []
    total = 0
    for acts, syms, *_ in adapter.iter_activations(adata, batch_size=batch_size):
        for L in adapter.layers:
            buffers[L].append(acts[L])
        syms_all.append(syms)
        total += len(syms)
        if total >= max_positions:
            break
    syms = np.concatenate(syms_all)
    paths = {}
    for L in adapter.layers:
        A = np.concatenate(buffers[L], 0).astype(np.float32)
        p = os.path.join(out_dir, f"layer_{L:02d}_activations.npy")
        np.save(p, A); paths[L] = p
    np.save(os.path.join(out_dir, "gene_symbols.npy"), syms)
    log(f"    extracted {total} positions x {adapter.d_model}d over layers {list(adapter.layers)}")
    return paths, syms


def build_catalog(adapter, out_dir: str, device: str = "cuda", top: int = 20,
                  expansion: int = 4, k: int = 32, epochs: int = 4, log=print):
    """Train a TopK SAE per layer and emit feature_catalog_L{L}.json (top-N genes/feature)."""
    syms = np.load(os.path.join(out_dir, "gene_symbols.npy"), allow_pickle=True).astype(str)
    uniq = np.array(sorted(set(syms))); gi = {g: i for i, g in enumerate(uniq)}
    sym_idx = np.array([gi[g] for g in syms])
    for L in adapter.layers:
        A = np.load(os.path.join(out_dir, f"layer_{L:02d}_activations.npy"), mmap_mode="r")
        cfg = SAECfg(d_model=A.shape[1], expansion=expansion, k=k, epochs=epochs)
        log(f"    [L{L}] SAE d_sae={cfg.d_sae} on {A.shape[0]} positions")
        sae, stats = train_sae(np.asarray(A), cfg, device=device, log=log)
        import torch
        torch.save({"state_dict": sae.state_dict(), "cfg": cfg.__dict__, "stats": stats},
                   os.path.join(out_dir, f"sae_L{L:02d}.pt"))
        F = feature_activations(sae, np.asarray(A), device=device)
        # per-gene mean activation -> top-N genes
        G = np.zeros((len(uniq), F.shape[1]), np.float32); c = np.zeros(len(uniq))
        np.add.at(G, sym_idx, F); np.add.at(c, sym_idx, 1)
        G /= np.maximum(c[:, None], 1)
        freq = (F > 0).mean(0)
        feats = {}
        for f in np.where(freq > 0)[0]:
            order = np.argsort(G[:, f])[::-1][:top]
            feats[str(int(f))] = {"top_genes": [str(uniq[i]) for i in order],
                                  "max_activation": float(G[:, f].max()),
                                  "activation_frequency": float(freq[f])}
        catalog = {"model": adapter.name, "layer": int(L), "d_model": int(A.shape[1]),
                   "d_sae": int(cfg.d_sae), "k": int(k), "n_alive": int((freq > 0).sum()),
                   "n_genes": int(len(uniq)), **stats, "features": feats}
        with open(os.path.join(out_dir, f"feature_catalog_L{L:02d}.json"), "w") as fh:
            json.dump(catalog, fh)
        log(f"    [L{L}] {catalog['n_alive']} alive, VarExpl {stats['var_explained']:.3f}, "
            f"dead {stats['dead_frac']:.3f}, dec-cos {stats['mean_abs_decoder_cos']:.3f}")
