#!/usr/bin/env python
"""Causal -> prediction (Kang / scGPT) — the narrowed, open angle. Does the host-response
predictor CAUSALLY rely on one interpretable IFN feature? Capture scGPT layer-6 residual,
train a TopK SAE, build a stim-vs-ctrl predictor on the pooled residual, then ABLATE the
IFN feature's contribution from the representation and measure the drop in predicted
stim-probability — vs ablating random features (specificity).

    conda activate scprint
    python scripts/kang_causal_prediction.py
"""
from __future__ import annotations
import os, sys
import numpy as np, scanpy as sc
sys.path.insert(0, "atlas_h100"); sys.path.insert(0, "external/single_cell_mechinterp/external/scGPT")
if not hasattr(os, "sched_getaffinity"):
    os.sched_getaffinity = lambda pid=0: set(range(os.cpu_count() or 1))
import torch
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from mechaudit.sae.topk_sae import SAEConfig, train_sae, feature_activations, apply_norm

CKPT = os.path.abspath("external/single_cell_mechinterp/external/scGPT_checkpoints/whole-human")
ISG = set("ISG15 IFI6 IFI27 IFI44 IFI44L IFIT1 IFIT2 IFIT3 IFITM1 IFITM3 MX1 MX2 OAS1 OAS2 OAS3 OASL "
          "RSAD2 IRF7 STAT1 STAT2 USP18 LY6E IFI35 XAF1 BST2 ISG20 HERC5 DDX58 DDX60 EIF2AK2 GBP1 "
          "CMPK2 SIGLEC1 CXCL10 IFI16 SAMD9 SAMD9L PARP9 PARP14".split())


def auroc_col(M, y):
    y = np.asarray(y, bool); R = np.apply_along_axis(rankdata, 0, M)
    npos = y.sum(); n = len(y)
    return (R[y].sum(0) - npos*(npos+1)/2) / (npos*(n-npos))


def main():
    from adapters.scgpt import ScGPTAdapter
    ad = sc.read_h5ad("data/kang/kang_batch2.h5ad")
    ad.var_names = ad.var["sym"].astype(str).values; ad.var_names_make_unique()
    rng = np.random.default_rng(0); idx = []
    for (s, d), g in ad.obs.groupby(["stim", "ind"], observed=True):
        idx += list(rng.choice(g.index.values, min(len(g), 18), replace=False))
    ad = ad[idx].copy()
    stim = (ad.obs["stim"].values == "stim").astype(int)
    print(f"==> {ad.n_obs} cells ({stim.sum()} stim); capturing scGPT L6 residual")

    adp = ScGPTAdapter(ckpt_dir=CKPT, layers=[6], max_seq_len=1200); adp.load(device="cpu")
    acts, syms, cids = [], [], []
    for a, s, c in adp.iter_activations(ad, batch_size=16):
        acts.append(a[6] if isinstance(a, dict) else a); syms.append(np.asarray(s)); cids.append(np.asarray(c))
    X = np.concatenate(acts, 0).astype(np.float32); syms = np.concatenate(syms); cids = np.concatenate(cids)
    print(f"==> residual {X.shape}")

    sae, stats, _ = train_sae(X, SAEConfig(d_sae=2048, k=32, epochs=40), device="cpu", verbose=False)
    F = feature_activations(sae, X, stats, device="cpu")                  # [tokens, d_sae] post-TopK
    Wdec = sae.W_dec.detach().cpu().numpy()
    if Wdec.shape[0] != F.shape[1]: Wdec = Wdec.T                          # -> [d_sae, d_in]
    xn = apply_norm(torch.as_tensor(X), stats).numpy()                    # normalized residual

    ncell = ad.n_obs
    def pool(mat):
        P = np.zeros((ncell, mat.shape[1]), np.float32); c = np.zeros(ncell)
        for row, ci in zip(mat, cids): P[ci] += row; c[ci] += 1
        return P / np.maximum(c[:, None], 1)
    P = pool(xn)                                                          # cell rep = pooled residual

    # host-response predictor on the representation (leave-one-donor-out AUROC)
    don = ad.obs["ind"].values.astype(str); oof = np.zeros(ncell)
    for d in np.unique(don):
        te = don == d
        c = LogisticRegression(max_iter=2000, class_weight="balanced").fit(P[~te], stim[~te])
        oof[te] = c.predict_proba(P[te])[:, 1]
    print(f"==> predictor (pooled scGPT residual, stim vs ctrl, LODO): AUROC {roc_auc_score(stim, oof):.3f}")

    # IFN feature: separates stim at cell level AND detects ISG genes
    Pf = pool(F)                                                          # per-cell feature profile
    au_state = auroc_col(Pf, stim)
    ug = np.array(sorted(set(syms))); gi = {g: i for i, g in enumerate(ug)}
    G = np.zeros((len(ug), F.shape[1]), np.float32); gc = np.zeros(len(ug))
    for row, g in zip(F, syms): G[gi[g]] += row; gc[gi[g]] += 1
    G /= np.maximum(gc[:, None], 1)
    y_isg = np.array([g in ISG for g in ug]); au_gene = auroc_col(G, y_isg)
    fj = int(np.abs(au_state - 0.5).argmax())
    print(f"==> IFN feature {fj}: stim-AUROC {au_state[fj]:.3f} | ISG-AUROC {au_gene[fj]:.3f} "
          f"({y_isg.sum()} ISGs in panel)")

    # single classifier for the causal counterfactual
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(P, stim)
    base = clf.predict_proba(P)[:, 1]

    def ablate_prob(j):
        xn_ab = xn - F[:, j:j+1] * Wdec[j][None, :]                       # remove feature j's contribution
        return clf.predict_proba(pool(xn_ab))[:, 1]

    ab = ablate_prob(fj)
    sm = stim == 1
    dprob = (base[sm] - ab[sm]).mean()                                    # drop in stim-prob on stim cells
    flip = ((base[sm] >= .5) & (ab[sm] < .5)).mean()
    # specificity: 20 random active features
    active = np.where(F.max(0) > 0)[0]; rndf = rng.choice(active[active != fj], 20, replace=False)
    rnd_d = [ (base[sm] - ablate_prob(int(j))[sm]).mean() for j in rndf ]
    print(f"\n==> CAUSAL -> PREDICTION:")
    print(f"   ablating the IFN feature: mean stim-prob drop {dprob:+.3f}, {100*flip:.1f}% of stim cells flip to ctrl")
    print(f"   random features (n=20):   mean drop {np.mean(rnd_d):+.3f} ± {np.std(rnd_d):.3f}")
    z = (dprob - np.mean(rnd_d)) / (np.std(rnd_d) + 1e-9)
    print(f"   specificity: IFN-feature effect is {z:+.1f}σ vs random features "
          f"-> {'SPECIFIC causal driver' if z>3 else 'distributed / not a single driver'}")
    np.savez_compressed("outputs/singlecell/kang_causal.npz", base=base, ab=ab, stim=stim,
                        dprob=dprob, flip=flip, rnd_d=np.array(rnd_d), fj=fj,
                        au_state=au_state, au_gene=au_gene)
    print("==> saved kang_causal.npz")


if __name__ == "__main__":
    main()
