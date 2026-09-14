#!/usr/bin/env python
"""L4 edge SIGN — do we reproduce Igor's inhibitory dominance (65-89% of causal
edges inhibitory)? Ablate each of the top source SAE features at layer L, read the
continuous pre-activation of a target SAE at a deeper layer L', and record the sign
of the downstream shift: ablate source -> target DOWN = excitatory edge; target UP =
inhibitory edge (the source normally suppresses it). Reconstruction error preserved
(as in scprint_circuits) so effects are feature-specific, not norm artifacts.

    conda activate scprint
    python scripts/scprint_edge_sign.py --layer 4 --target-layer 6
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ckpt_scprint/medium-v1.5.ckpt")
    ap.add_argument("--layer", type=int, default=4)          # source
    ap.add_argument("--target-layer", type=int, default=6)   # downstream
    ap.add_argument("--n-genes", type=int, default=600)
    ap.add_argument("--n-cells", type=int, default=48)
    ap.add_argument("--d-sae", type=int, default=2048)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--n-source", type=int, default=50)
    ap.add_argument("--out", default="outputs/scprint")
    args = ap.parse_args()

    import numpy as np
    import scanpy as sc
    import torch
    from scdataloader import Preprocessor, SimpleAnnDataset, Collator
    from torch.utils.data import DataLoader
    from scprint import scPrint
    from mechaudit.sae.topk_sae import SAEConfig, train_sae, feature_activations, apply_norm, compute_norm_stats

    # source SAE from the layer-4 residual we already captured
    X = np.load(os.path.join(args.out, f"residual_L{args.layer}.npz"))["X"]
    sae_s, stats_s, _ = train_sae(X, SAEConfig(d_sae=args.d_sae, k=args.k, epochs=40), device="cpu", verbose=False)
    Fs = feature_activations(sae_s, X, stats_s, device="cpu")
    src_feats = list(np.argsort(Fs.mean(0))[::-1][:args.n_source])   # most-active hub features
    print(f"==> source layer {args.layer}: testing {len(src_feats)} hub features")

    # model + one batch of cells (same setup as scprint_circuits)
    adata = sc.datasets.pbmc3k(); adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
    adata = Preprocessor(is_symbol=True, skip_validate=True, min_valid_genes_id=1000,
                         min_nnz_genes=100, filter_gene_by_counts=False)(adata)
    m = scPrint.load_from_checkpoint(args.checkpoint, precpt_gene_emb=None, transformer="normal"); m.eval()
    sc.pp.highly_variable_genes(adata, n_top_genes=args.n_genes, flavor="seurat_v3")
    fixed = [g for g in adata.var.index[adata.var.highly_variable].tolist() if g in set(m.genes)]
    ds = SimpleAnnDataset(adata[: args.n_cells], obs_to_output=["organism_ontology_term_id"])
    col = Collator(organisms=m.organisms, valid_genes=m.genes, how="some", genelist=fixed, max_len=0)
    batch = next(iter(DataLoader(ds, collate_fn=col, batch_size=args.n_cells, shuffle=False)))
    gp, expr = batch["genes"], batch["x"]; ng = gp.shape[1]
    mean_t = torch.as_tensor(stats_s.mean); scale = float(stats_s.scale)

    # patch hook at the source block; capture hook at the target block
    state = {"ablate": None}; cap = {}

    def patch(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        g0 = h.shape[1] - ng; hg = h[:, g0:, :]
        xn = (hg - mean_t) / scale
        f, _ = sae_s.encode(xn); recon = sae_s.decode(f); err = xn - recon
        if state["ablate"] is not None:
            f = f.clone(); f[..., state["ablate"]] = 0.0
        hg_new = (sae_s.decode(f) + err) * scale + mean_t
        h = h.clone(); h[:, g0:, :] = hg_new
        return (h,) + out[1:] if isinstance(out, tuple) else h

    def capture(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        cap["h"] = h[:, h.shape[1]-ng:, :].detach()

    hp = m.transformer.blocks[args.layer].register_forward_hook(patch)
    hc = m.transformer.blocks[args.target_layer].register_forward_hook(capture)

    def fwd():
        with torch.no_grad():
            m(gene_pos=gp, expression=expr, req_depth=batch["depth"], depth_mult=expr.sum(1))
        return cap["h"].reshape(-1, cap["h"].shape[-1]).numpy()      # [tokens, d]

    # baseline: train target SAE on layer-L' residual, get per-token preactivations
    state["ablate"] = None; H0 = fwd()
    sae_t, stats_t, _ = train_sae(H0, SAEConfig(d_sae=args.d_sae, k=args.k, epochs=40), device="cpu", verbose=False)

    def preact(H):
        xn = apply_norm(torch.as_tensor(H, dtype=torch.float32), stats_t)
        with torch.no_grad():
            return sae_t.preactivation(xn).numpy()                    # [tokens, d_sae]

    P0 = preact(H0)
    base_mean = P0.mean(0); sigma = P0.std(0) + 1e-6
    live = sigma > 1e-4                                                # target features with variability
    print(f"==> target layer {args.target_layer}: {live.sum()} live features")

    # ablate each source feature -> standardized downstream shift
    Z = np.zeros((len(src_feats), P0.shape[1]), np.float32)
    for r, i in enumerate(src_feats):
        state["ablate"] = int(i)
        Z[r] = (preact(fwd()).mean(0) - base_mean) / sigma
    hp.remove(); hc.remove()

    # edges = |z|>tau on live targets; sign(z<0)=excitatory (ablate src -> target down),
    #         sign(z>0)=inhibitory (ablate src -> target up)
    Zl = Z[:, live]
    print("\n==> edge sign (Igor: 65-89% inhibitory):")
    for tau in (2, 3, 4, 5):
        mask = np.abs(Zl) > tau
        n = int(mask.sum())
        inhib = int((Zl[mask] > 0).sum())
        if n:
            print(f"   |z|>{tau}: {n:>6} edges | inhibitory {100*inhib/n:5.1f}% | excitatory {100*(1-inhib/n):5.1f}%")
    # net direction sanity
    print(f"   overall mean z over all source×target = {Zl.mean():+.3f} "
          f"({'net inhibitory' if Zl.mean()>0 else 'net excitatory'})")
    np.savez_compressed(os.path.join(args.out, "edge_sign.npz"), Z=Z, live=live, src_feats=np.array(src_feats))
    print("==> saved to", os.path.join(args.out, "edge_sign.npz"))


if __name__ == "__main__":
    main()
