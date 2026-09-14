#!/usr/bin/env python
"""scPRINT Layer-4 (causal circuits). Ablate a Layer-3 SAE feature by activation
patching at transformer block L, run the forward to scPRINT's NB decoder, and
measure the causal shift in predicted gene expression (out['mean']). Test whether
ablating the E2F1-detecting feature preferentially moves E2F1 target genes —
causal confirmation of the correlational Layer-3 result.

    conda activate scprint
    python scripts/scprint_circuits.py --layer 4 --concept E2F1
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ckpt_scprint/medium-v1.5.ckpt")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--concept", default="E2F1")     # TRRUST regulon to target
    ap.add_argument("--n-genes", type=int, default=600)
    ap.add_argument("--n-cells", type=int, default=48)
    ap.add_argument("--d-sae", type=int, default=2048)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--biomart", default="external/scprint_data/biomart_pos.parquet")
    ap.add_argument("--trrust", default="external/single_cell_mechinterp/external/networks/trrust_human.tsv")
    ap.add_argument("--out", default="outputs/scprint")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import scanpy as sc
    import torch
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score
    from scdataloader import Preprocessor, SimpleAnnDataset, Collator
    from torch.utils.data import DataLoader
    from scprint import scPrint
    from mechaudit.sae.topk_sae import SAEConfig, train_sae, feature_activations, apply_norm

    # ---- SAE from the captured residual (Layer 3) -----------------------
    X = np.load(os.path.join(args.out, f"residual_L{args.layer}.npz"))["X"]
    gid = np.array([l.strip() for l in open(os.path.join(args.out, f"residual_L{args.layer}_genes.txt"))])
    sae, stats, _ = train_sae(X, SAEConfig(d_sae=args.d_sae, k=args.k, epochs=40),
                              device="cpu", verbose=False)
    F = feature_activations(sae, X, stats, device="cpu")

    # find the feature that best detects the concept's TRRUST regulon (per gene)
    bm = pd.read_parquet(args.biomart)
    ens2sym = {e: str(s).upper() for e, s in bm["hgnc_symbol"].items()}
    trrust = pd.read_csv(args.trrust, sep="\t", header=None, names=["tf", "tg", "m", "p"])
    targets = {t.upper() for t in trrust[trrust.tf.str.upper() == args.concept]["tg"]}
    uniq = np.array(sorted(set(gid)))
    idx = {g: i for i, g in enumerate(uniq)}
    Gp = np.zeros((len(uniq), F.shape[1]), np.float32); cnt = np.zeros(len(uniq))
    for row, g in zip(F, gid):
        Gp[idx[g]] += row; cnt[idx[g]] += 1
    Gp /= np.maximum(cnt[:, None], 1)
    y = np.array([ens2sym.get(g, "") in targets for g in uniq])
    print(f"==> concept {args.concept}: {y.sum()} target genes in panel")
    R = np.apply_along_axis(rankdata, 0, Gp)
    npos = y.sum(); auroc = (R[y].sum(0) - npos*(npos+1)/2) / (npos*(len(uniq)-npos))
    feat = int(np.abs(auroc - 0.5).argmax())
    print(f"==> target SAE feature {feat} (E2F1-detector AUROC {auroc[feat]:.3f})")

    # ---- model + a batch of cells ---------------------------------------
    adata = sc.datasets.pbmc3k(); adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
    adata = Preprocessor(is_symbol=True, skip_validate=True, min_valid_genes_id=1000,
                         min_nnz_genes=100, filter_gene_by_counts=False)(adata)
    m = scPrint.load_from_checkpoint(args.checkpoint, precpt_gene_emb=None, transformer="normal"); m.eval()
    sc.pp.highly_variable_genes(adata, n_top_genes=args.n_genes, flavor="seurat_v3")
    fixed = [g for g in adata.var.index[adata.var.highly_variable].tolist() if g in set(m.genes)]
    ds = SimpleAnnDataset(adata[: args.n_cells], obs_to_output=["organism_ontology_term_id"])
    col = Collator(organisms=m.organisms, valid_genes=m.genes, how="some", genelist=fixed, max_len=0)
    batch = next(iter(DataLoader(ds, collate_fn=col, batch_size=args.n_cells, shuffle=False)))
    gp, expr = batch["genes"], batch["x"]
    ng = gp.shape[1]
    mean_t = torch.as_tensor(stats.mean); scale = float(stats.scale)

    # ---- activation-patching hook at block L ----------------------------
    state = {"ablate": None}

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        seq = h.shape[1]; g0 = seq - ng
        hg = h[:, g0:, :]
        xn = (hg - mean_t) / scale
        f, _ = sae.encode(xn)
        recon = sae.decode(f); err = xn - recon
        if state["ablate"] is not None:
            f = f.clone(); f[..., state["ablate"]] = 0.0
        hg_new = (sae.decode(f) + err) * scale + mean_t
        h = h.clone(); h[:, g0:, :] = hg_new
        return (h,) + out[1:] if isinstance(out, tuple) else h

    handle = m.transformer.blocks[args.layer].register_forward_hook(hook)

    def run():
        with torch.no_grad():
            o = m(gene_pos=gp, expression=expr, req_depth=batch["depth"], depth_mult=expr.sum(1))
        return o["mean"].cpu().numpy()  # [B, ng] predicted expression

    state["ablate"] = None; base = run()
    state["ablate"] = feat; abl = run()
    handle.remove()

    # ---- causal effect: which genes shift? vs E2F1 targets --------------
    delta = np.abs(abl - base).mean(0)               # [ng] mean |Δ| per gene position
    gene_at_pos = np.array([ens2sym.get(str(g), "") for g in
                            np.array(m.genes)[gp[0].cpu().numpy()]])
    ytgt = np.array([s in targets for s in gene_at_pos])
    order = np.argsort(delta)[::-1]
    print(f"\n==> mean |Δ predicted expr| over {args.n_cells} cells; {ytgt.sum()} E2F1 targets in {ng} genes")
    print("   top-15 most-shifted genes:", [g for g in gene_at_pos[order][:15] if g][:15])
    if ytgt.sum() >= 3 and ytgt.sum() < ng:
        au = roc_auc_score(ytgt, delta)
        print(f"   AUROC(|Δ| separates E2F1 targets): {au:.3f}  (random 0.5)")
        print(f"   mean |Δ| targets {delta[ytgt].mean():.4f} vs non-targets {delta[~ytgt].mean():.4f} "
              f"({delta[ytgt].mean()/max(delta[~ytgt].mean(),1e-9):.2f}×)")
    np.savez_compressed(os.path.join(args.out, f"circuit_L{args.layer}_{args.concept}.npz"),
                        delta=delta, is_target=ytgt, genes=gene_at_pos)
    print("==> saved circuit effect to", args.out)


if __name__ == "__main__":
    main()
