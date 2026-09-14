#!/usr/bin/env python
"""AIDO.Cell-10M audit — Layer 4 (causal circuits). Ablate the L3 SAE feature that
detects the RFX regulon by activation-patching the layer-4 residual, run the forward
to the masked-LM head, and measure the causal shift in predicted gene expression.
Test: does ablating the RFX feature preferentially move RFX target genes? — causal
confirmation of L3, symmetric to the scPRINT E2F1 result.

    conda activate scprint
    python scripts/aido_circuits.py --layer 4
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ckpt_aido")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--concept", default="RFXANK")   # TRRUST regulon to target
    ap.add_argument("--n-cells", type=int, default=32)
    ap.add_argument("--d-sae", type=int, default=2048)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--trrust", default="external/single_cell_mechinterp/external/networks/trrust_human.tsv")
    ap.add_argument("--out", default="outputs/aido")
    args = ap.parse_args()

    import numpy as np
    import pandas as pd
    import scanpy as sc
    import torch
    import torch.nn as nn
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score
    from gb_cell.models import CellFoundationConfig
    from gb_cell.models.modeling_cellfoundation import CellFoundationForMaskedLM
    from gb_cell.utils import align_adata, preprocess_counts
    from mechaudit.sae.topk_sae import SAEConfig, train_sae, feature_activations

    genes = [l.split("\t")[0] for l in open("external/scprint_data/aido_genes.tsv").read().splitlines()[1:]]
    sym2pos = {g: i for i, g in enumerate(genes)}
    tr = pd.read_csv(args.trrust, sep="\t", header=None, names=["tf", "tg", "m", "p"])
    tr_genes = set(tr.tf.str.upper()) | set(tr.tg.str.upper())
    keep = sorted(sym2pos[g] for g in tr_genes if g in sym2pos)
    keep_sym = np.array([genes[i] for i in keep])
    targets = {t.upper() for t in tr[tr.tf.str.upper() == args.concept]["tg"]}

    cfg = CellFoundationConfig.from_pretrained(args.model)
    m = CellFoundationForMaskedLM.from_pretrained(args.model, config=cfg).eval()

    # locate the transformer block list (ModuleList of length num_hidden_layers)
    blocks = None
    for name, mod in m.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == cfg.num_hidden_layers:
            blocks = mod; print(f"==> transformer blocks at '{name}' ({len(mod)})"); break
    assert blocks is not None, "could not find transformer block list"

    adata = sc.datasets.pbmc3k()[: args.n_cells].copy()
    adata_al, attn = align_adata(adata)
    xb = adata_al.X.toarray() if hasattr(adata_al.X, "toarray") else adata_al.X
    inp = preprocess_counts(xb, device="cpu")
    am = torch.cat([torch.from_numpy(attn).unsqueeze(0).repeat(inp.shape[0], 1),
                    torch.ones((inp.shape[0], 2))], dim=1)
    bin_vals = torch.arange(cfg.bin_num).float()

    # ---- capture residual (pass 1) + train SAE + find RFX feature --------
    cap = {}
    h1 = blocks[args.layer].register_forward_hook(
        lambda mod, i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    with torch.no_grad():
        m(input_ids=inp, attention_mask=am)
    h1.remove()
    Hg = cap["h"][:, keep, :]                              # [B, |keep|, d]
    X = Hg.reshape(-1, Hg.shape[-1]).numpy()
    gid = np.tile(keep_sym, Hg.shape[0])
    sae, stats, _ = train_sae(X, SAEConfig(d_sae=args.d_sae, k=args.k, epochs=40), device="cpu", verbose=False)
    F = feature_activations(sae, X, stats, device="cpu")
    uniq = np.array(sorted(set(gid))); idx = {g: i for i, g in enumerate(uniq)}
    Gp = np.zeros((len(uniq), F.shape[1]), np.float32); c = np.zeros(len(uniq))
    for row, g in zip(F, gid):
        Gp[idx[g]] += row; c[idx[g]] += 1
    Gp /= np.maximum(c[:, None], 1)
    y = np.array([g in targets for g in uniq])
    R = np.apply_along_axis(rankdata, 0, Gp); npos = y.sum()
    au = (R[y].sum(0) - npos*(npos+1)/2)/(npos*(len(uniq)-npos))
    feat = int(np.abs(au - 0.5).argmax())
    print(f"==> {args.concept}: {y.sum()} targets in panel; RFX SAE feature {feat} (AUROC {au[feat]:.3f})")
    fa = F[:, feat]
    print(f"    feature {feat} activation on batch: nonzero {100*(fa>0).mean():.1f}% of tokens, "
          f"mean {fa.mean():.4f}, max {fa.max():.4f}")

    # ---- activation-patching hook + baseline/ablated logits -------------
    mean_t = torch.as_tensor(stats.mean); scale = float(stats.scale)
    state = {"ablate": None}

    def patch(mod, i, o):
        h = o[0] if isinstance(o, tuple) else o
        hg = h[:, keep, :]
        xn = (hg - mean_t) / scale
        f, _ = sae.encode(xn); recon = sae.decode(f); err = xn - recon
        if state["ablate"] is not None:
            f = f.clone(); f[..., state["ablate"]] = 0.0
        hg_new = (sae.decode(f) + err) * scale + mean_t
        h = h.clone(); h[:, keep, :] = hg_new
        return (h,) + o[1:] if isinstance(o, tuple) else h

    hp = blocks[args.layer].register_forward_hook(patch)

    def pred_logits():
        with torch.no_grad():
            out = m(input_ids=inp, attention_mask=am)
        logits = out.logits if hasattr(out, "logits") else out[0]   # [B, seq, bin_num]
        return logits.float()[:, keep, :].cpu().numpy()             # [B, |keep|, bins]

    state["ablate"] = None; base = pred_logits()
    state["ablate"] = feat; abl = pred_logits()
    hp.remove()

    # sensitivity: full change in the predicted bin-distribution per gene (not just its mean)
    delta = np.abs(abl - base).mean(axis=(0, 2))          # [|keep|]
    print(f"    total |Δ logit| summed over genes: {np.abs(abl-base).sum():.4e} "
          f"(0 => ablation had no downstream effect)")
    ytgt = np.array([g in targets for g in keep_sym])
    order = np.argsort(delta)[::-1]
    print(f"\n==> mean |Δ predicted expr| over {inp.shape[0]} cells; {ytgt.sum()} RFX targets in {len(keep_sym)} genes")
    print("   top-15 most-shifted genes:", list(keep_sym[order][:15]))
    if 3 <= ytgt.sum() < len(ytgt):
        auc = roc_auc_score(ytgt, delta)
        print(f"   AUROC(|Δ| separates RFX targets): {auc:.3f}  (random 0.5)")
        print(f"   mean |Δ| targets {delta[ytgt].mean():.4f} vs non-targets {delta[~ytgt].mean():.4f} "
              f"({delta[ytgt].mean()/max(delta[~ytgt].mean(),1e-9):.2f}×)")
    np.savez_compressed(os.path.join(args.out, f"circuit_L{args.layer}_{args.concept}.npz"),
                        delta=delta, is_target=ytgt, genes=keep_sym)
    print("==> saved circuit effect to", args.out)


if __name__ == "__main__":
    main()
