#!/usr/bin/env python
"""Mechinterp on scGPT + Kang (IFN host response) — the differentiator. Capture scGPT's
per-gene residual (layer 6) via the prior work's ScGPTAdapter (CPU, use_fast_transformer=False),
train a TopK SAE, and look for a RESPONSE feature at two levels:
  (state)  a feature whose per-CELL activation separates IFN-stimulated from control
  (gene)   a feature that fires on interferon-stimulated genes (ISG set)
If the same feature does both, scGPT carries a conjunctive gene x state 'IFN program'
feature (like the RFX-in-APC feature) — an interpretable host-response circuit.

    conda activate scprint
    python scripts/kang_scgpt_mechinterp.py
"""
from __future__ import annotations
import os, sys
import numpy as np, scanpy as sc
sys.path.insert(0, "atlas_h100")
sys.path.insert(0, "external/single_cell_mechinterp/external/scGPT")
if not hasattr(os, "sched_getaffinity"):
    os.sched_getaffinity = lambda pid=0: set(range(os.cpu_count() or 1))
from mechaudit.sae.topk_sae import SAEConfig, train_sae, feature_activations
from scipy.stats import rankdata

CKPT = os.path.abspath("external/single_cell_mechinterp/external/scGPT_checkpoints/whole-human")
ISG = set("ISG15 IFI6 IFI27 IFI44 IFI44L IFIT1 IFIT2 IFIT3 IFITM1 IFITM3 MX1 MX2 OAS1 OAS2 OAS3 OASL "
          "RSAD2 IRF7 STAT1 STAT2 USP18 LY6E IFI35 XAF1 BST2 ISG20 HERC5 DDX58 DDX60 EIF2AK2 GBP1 "
          "CMPK2 SIGLEC1 CXCL10 IFI16 SAMD9 SAMD9L PARP9 PARP14".split())


def auroc_vec(scores2d, y):
    y = np.asarray(y, bool); R = np.apply_along_axis(rankdata, 0, scores2d)
    npos = y.sum(); n = len(y)
    return (R[y].sum(0) - npos*(npos+1)/2) / (npos*(n-npos))


def main():
    from adapters.scgpt import ScGPTAdapter
    ad = sc.read_h5ad("data/kang/kang_batch2.h5ad")
    ad.var_names = ad.var["sym"].astype(str).values           # adapter matches vocab by symbol
    ad.var_names_make_unique()
    rng = np.random.default_rng(0); idx = []
    for (s, d), g in ad.obs.groupby(["stim", "ind"], observed=True):
        idx += list(rng.choice(g.index.values, min(len(g), 16), replace=False))   # ~256 cells
    ad = ad[idx].copy()
    stim_cell = (ad.obs["stim"].values == "stim").astype(int)
    print(f"==> {ad.n_obs} cells ({stim_cell.sum()} stim), capturing scGPT layer-6 residual on CPU")

    adp = ScGPTAdapter(ckpt_dir=CKPT, layers=[6], max_seq_len=1200)
    adp.load(device="cpu")
    acts, syms, cids = [], [], []
    for a, s, c in adp.iter_activations(ad, batch_size=16):
        acts.append(a[6] if isinstance(a, dict) else a); syms.append(np.asarray(s)); cids.append(np.asarray(c))
    X = np.concatenate(acts, 0).astype(np.float32)
    syms = np.concatenate(syms); cids = np.concatenate(cids)
    print(f"==> residual {X.shape} over {len(np.unique(syms))} genes")

    sae, stats, log = train_sae(X, SAEConfig(d_sae=2048, k=32, epochs=40), device="cpu", verbose=False)
    F = feature_activations(sae, X, stats, device="cpu")
    print(f"    SAE FVU {log.history[-1]['fvu']:.3f}")

    # (state) per-cell feature profile -> which feature separates stim vs ctrl
    ncell = ad.n_obs
    P = np.zeros((ncell, F.shape[1]), np.float32); cnt = np.zeros(ncell)
    for row, c in zip(F, cids):
        P[c] += row; cnt[c] += 1
    P /= np.maximum(cnt[:, None], 1)
    au_state = auroc_vec(P, stim_cell)
    f_state = int(np.abs(au_state - 0.5).argmax())

    # (gene) per-gene feature profile -> which feature fires on ISG genes
    ug = np.array(sorted(set(syms))); gi = {g: i for i, g in enumerate(ug)}
    G = np.zeros((len(ug), F.shape[1]), np.float32); gc = np.zeros(len(ug))
    for row, g in zip(F, syms):
        G[gi[g]] += row; gc[gi[g]] += 1
    G /= np.maximum(gc[:, None], 1)
    y_isg = np.array([g in ISG for g in ug]); n_isg = y_isg.sum()
    au_gene = auroc_vec(G, y_isg)
    f_gene = int(np.abs(au_gene - 0.5).argmax())

    print(f"\n==> RESPONSE feature (state): feature {f_state} separates stim cells, AUROC {au_state[f_state]:.3f}")
    print(f"==> ISG feature (gene):       feature {f_gene} detects ISGs ({n_isg} in panel), AUROC {au_gene[f_gene]:.3f}")
    print(f"    is the state feature also ISG-enriched?  feature {f_state} gene-AUROC vs ISG = {au_gene[f_state]:.3f}")
    print(f"    is the ISG feature also stim-selective?  feature {f_gene} state-AUROC vs stim = {au_state[f_gene]:.3f}")
    conj = abs(au_state[f_state]-0.5) > 0.15 and abs(au_gene[f_state]-0.5) > 0.1
    print(f"    -> conjunctive IFN-program feature (gene x state)? {'YES' if conj else 'partial/no'}")
    # top ISG genes by the state feature (interpretability)
    order = np.argsort(G[:, f_state])[::-1]
    print("    top genes on the response feature:", list(ug[order][:12]))
    np.savez_compressed("outputs/singlecell/kang_scgpt_sae.npz",
                        au_state=au_state, au_gene=au_gene, f_state=f_state, f_gene=f_gene,
                        genes=ug, isg=y_isg, G_state=G[:, f_state])
    print("==> saved kang_scgpt_sae.npz")


if __name__ == "__main__":
    main()
