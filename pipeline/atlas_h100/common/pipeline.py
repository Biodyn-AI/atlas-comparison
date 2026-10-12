"""Model-agnostic pipeline: adapter -> per-layer activations -> TopK SAE -> feature
catalog (top-20 genes/feature). One model, all requested layers. Outputs match the
bio-sae schema so every model joins on the same annotation vocabulary downstream.
"""
from __future__ import annotations
import json, os
import numpy as np
from .sae import SAECfg, train_sae, feature_activations


def extract(adapter, adata, out_dir: str, batch_size: int = 8, max_positions: int = 4_000_000,
            per_cell: int = 0, seed: int = 0, log=print):
    """Run the adapter over the corpus, store per-layer activation memmaps + gene symbols.

    per_cell = 0: the original behaviour, kept so the published runs can be reproduced: positions
        are taken in corpus order until max_positions is reached. Because models read very different
        numbers of genes per cell, this caps cells unequally (AIDO.Cell 32 cells, tGPT all 6,000).
    per_cell > 0: every cell of the corpus is read, and at most per_cell of its positions are kept,
        drawn uniformly without replacement (seeded). Every model then sees the same cells, each cell
        weighted equally; max_positions is ignored. Rows are streamed to disk while the model runs, so
        all layers of a 3B model fit, and each layer is then stored as float16 when its values fit that
        range (|a| < 65,000) and as float32 otherwise (tGPT has outliers above 150,000).
    """
    os.makedirs(out_dir, exist_ok=True)
    stream = per_cell > 0
    buffers = {L: [] for L in adapter.layers}
    raw = {L: open(os.path.join(out_dir, f"layer_{L:02d}.f32"), "wb") for L in adapter.layers} if stream else {}
    maxabs = {L: 0.0 for L in adapter.layers}
    syms_all = []; cells_all = []
    total = 0; d = None
    rng = np.random.default_rng(seed)
    for acts, syms, *rest in adapter.iter_activations(adata, batch_size=batch_size):
        cids = np.asarray(rest[0]) if rest and rest[0] is not None else None
        if stream:
            if cids is None:
                raise ValueError(f"{adapter.name}: per-cell sampling needs per-position cell ids")
            keep = []
            for c in np.unique(cids):
                pos = np.flatnonzero(cids == c)
                keep.append(pos if len(pos) <= per_cell else np.sort(rng.choice(pos, per_cell, replace=False)))
            keep = np.concatenate(keep)
            syms = np.asarray(syms)[keep]; cids = cids[keep]
            for L in adapter.layers:
                a = np.asarray(acts[L][keep], dtype=np.float32)
                d = a.shape[1]
                if len(a):
                    maxabs[L] = max(maxabs[L], float(np.abs(a).max()))
                raw[L].write(a.tobytes())
        else:
            for L in adapter.layers:
                buffers[L].append(acts[L])
        syms_all.append(syms)
        if cids is not None:                        # per-position cell ids (enables split-half by cell for held-out control)
            cells_all.append(cids)
        total += len(syms)
        if not stream and total >= max_positions:
            break
    syms = np.concatenate(syms_all)
    paths = {}
    for L in adapter.layers:
        p = os.path.join(out_dir, f"layer_{L:02d}_activations.npy")
        if stream:                                  # raw float32 rows -> .npy, one layer at a time
            raw[L].close()
            src = os.path.join(out_dir, f"layer_{L:02d}.f32")
            R = np.memmap(src, dtype=np.float32, mode="r", shape=(total, d))
            dt = np.float16 if maxabs[L] < 65000 else np.float32
            A = np.lib.format.open_memmap(p, mode="w+", dtype=dt, shape=(total, d))
            for i in range(0, total, 200_000):
                A[i:i + 200_000] = R[i:i + 200_000]
            A.flush(); del A, R
            os.remove(src)
        else:
            np.save(p, np.concatenate(buffers[L], 0).astype(np.float32))
        paths[L] = p
    np.save(os.path.join(out_dir, "gene_symbols.npy"), syms)
    n_cells = None
    if cells_all:
        cells = np.concatenate(cells_all)
        np.save(os.path.join(out_dir, "cell_ids.npy"), cells)
        n_cells = int(len(np.unique(cells)))
    with open(os.path.join(out_dir, "extraction.json"), "w") as fh:
        json.dump({"positions": int(total), "cells": n_cells, "per_cell": int(per_cell), "seed": int(seed),
                   "max_positions": None if per_cell else int(max_positions),
                   "dtype": {str(L): ("float16" if stream and maxabs[L] < 65000 else "float32") for L in adapter.layers},
                   "max_abs": {str(L): round(v, 2) for L, v in maxabs.items()} if stream else None,
                   "layers": list(map(int, adapter.layers))}, fh)
    log(f"    extracted {total} positions from {n_cells} cells x {adapter.d_model}d over layers {list(adapter.layers)}")
    return paths, syms


def build_catalog(adapter, out_dir: str, device: str = "cuda", top: int = 20,
                  expansion: int = 4, k: int = 32, epochs: int = 4, seed: int = 0, log=print):
    """Train a TopK SAE per layer and emit feature_catalog_L{L}.json (top-N genes/feature)."""
    syms = np.load(os.path.join(out_dir, "gene_symbols.npy"), allow_pickle=True).astype(str)
    uniq = np.array(sorted(set(syms))); gi = {g: i for i, g in enumerate(uniq)}
    sym_idx = np.array([gi[g] for g in syms])
    for L in adapter.layers:
        A = np.load(os.path.join(out_dir, f"layer_{L:02d}_activations.npy"), mmap_mode="r")
        cfg = SAECfg(d_model=A.shape[1], expansion=expansion, k=k, epochs=epochs, seed=seed)
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
            # stable descending sort, genes with zero mean activation dropped: a feature that fires on
            # fewer than `top` genes gets a shorter list instead of arbitrary zero-valued filler
            order = np.argsort(-G[:, f], kind="stable")
            order = order[G[order, f] > 0][:top]
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
