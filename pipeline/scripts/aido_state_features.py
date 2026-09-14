#!/usr/bin/env python
"""State-dependent features — does an SAE feature's activation depend on the CELL TYPE,
not just the gene? Clean prediction: the RFX feature fires on RFX/HLA gene tokens only
in antigen-presenting cells (monocytes/B), not T cells (validates the L4 caveat with
cell-type data). Also quantifies, across all features, how much activation is explained
by cell type (eta^2). AIDO L4 residual, PBMC with marker-based cell types.

    conda activate scprint
    python scripts/aido_state_features.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MARKERS = {
    "T": ["CD3D", "CD3E", "CD3G", "TRAC", "IL7R"],
    "B": ["MS4A1", "CD79A", "CD79B", "CD19"],
    "Mono": ["CD14", "LYZ", "FCN1", "S100A8", "S100A9"],
    "NK": ["NKG7", "GNLY", "KLRD1", "KLRF1"],
    "DC": ["FCER1A", "CST3", "CLEC10A"],
}
APC = {"Mono", "B", "DC"}


def main():
    import numpy as np, pandas as pd, scanpy as sc, torch
    from scipy.stats import rankdata, f_oneway
    from gb_cell.models import CellFoundationModel, CellFoundationConfig
    from gb_cell.utils import align_adata, preprocess_counts
    from mechaudit.sae.topk_sae import SAEConfig, train_sae, feature_activations

    N = 160
    genes = [l.split("\t")[0].upper() for l in open(f"{BASE}/external/scprint_data/aido_genes.tsv").read().splitlines()[1:]]
    sym2pos = {g: i for i, g in enumerate(genes)}
    tr = pd.read_csv(f"{BASE}/external/single_cell_mechinterp/external/networks/trrust_human.tsv",
                     sep="\t", header=None, names=["tf", "tg", "m", "p"])
    tr_genes = set(tr.tf.str.upper()) | set(tr.tg.str.upper())
    keep = sorted(sym2pos[g] for g in tr_genes if g in sym2pos)
    keep_sym = np.array([genes[i] for i in keep])
    rfx = {t.upper() for tf in ("RFXANK", "RFXAP", "RFX5") for t in tr[tr.tf.str.upper() == tf]["tg"]}
    y_rfx = np.array([g in rfx for g in keep_sym])

    # ---- cell types from markers (on log-norm expression) --------------
    adata = sc.datasets.pbmc3k()[:N].copy()
    raw = adata.copy(); sc.pp.normalize_total(raw, target_sum=1e4); sc.pp.log1p(raw)
    def zscore(v): return (v - v.mean()) / (v.std() + 1e-9)
    scores = {}
    for ct, ms in MARKERS.items():
        present = [g for g in ms if g in raw.var_names]
        X = raw[:, present].X
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        scores[ct] = np.mean([zscore(X[:, j]) for j in range(X.shape[1])], axis=0)
    S = np.stack([scores[ct] for ct in MARKERS], 1)
    cts = np.array(list(MARKERS))
    cell_type = np.where(S.max(1) > 0.2, cts[S.argmax(1)], "other")
    print("==> cell types:", {c: int((cell_type == c).sum()) for c in np.unique(cell_type)})

    # ---- AIDO L4 residual per (cell, gene) -----------------------------
    cfg = CellFoundationConfig.from_pretrained(f"{BASE}/ckpt_aido")
    m = CellFoundationModel.from_pretrained(f"{BASE}/ckpt_aido", config=cfg).eval()
    ad, attn = align_adata(adata)
    attn_t = torch.from_numpy(attn).unsqueeze(0)
    acts, cellid = [], []
    with torch.no_grad():
        for s in range(0, ad.n_obs, 8):
            xb = ad.X[s:s+8]; xb = xb.toarray() if hasattr(xb, "toarray") else xb
            inp = preprocess_counts(xb, device="cpu")
            am = torch.cat([attn_t.repeat(inp.shape[0], 1), torch.ones((inp.shape[0], 2))], 1)
            out = m(input_ids=inp, attention_mask=am, output_hidden_states=True)
            hid = out.hidden_states[4][:, keep, :]
            for b in range(hid.shape[0]):
                acts.append(hid[b].numpy()); cellid.extend([s + b] * len(keep))
    X = np.concatenate(acts, 0); cellid = np.array(cellid)
    gid = np.tile(keep_sym, ad.n_obs)
    print(f"==> residual {X.shape}")

    sae, stats, _ = train_sae(X, SAEConfig(d_sae=2048, k=32, epochs=40), device="cpu", verbose=False)
    F = feature_activations(sae, X, stats, device="cpu")

    # RFX feature = best RFX-target detector (per gene)
    uniq = np.array(sorted(set(gid))); gi = {g: i for i, g in enumerate(uniq)}
    G = np.zeros((len(uniq), F.shape[1])); c = np.zeros(len(uniq))
    for row, g in zip(F, gid):
        G[gi[g]] += row; c[gi[g]] += 1
    G /= np.maximum(c[:, None], 1)
    yy = np.array([g in rfx for g in uniq]); R = np.apply_along_axis(rankdata, 0, G); npos = yy.sum()
    au = (R[yy].sum(0) - npos*(npos+1)/2) / (npos*(len(uniq)-npos)); feat = int(np.abs(au-0.5).argmax())
    print(f"==> RFX feature {feat} (AUROC {au[feat]:.3f})")

    # RFX feature activation on RFX-target tokens, by cell type
    tok_ct = cell_type[cellid]; is_rfx_tok = np.isin(gid, list(rfx))
    print("\n==> RFX feature activation on RFX-target gene tokens, by cell type:")
    order = ["Mono", "B", "DC", "NK", "T", "other"]
    rows = {}
    for ct in order:
        msk = is_rfx_tok & (tok_ct == ct)
        if msk.sum() > 5:
            rows[ct] = F[msk, feat].mean()
            print(f"   {ct:<6} {rows[ct]:.4f}   (n={int(msk.sum())})  {'← APC' if ct in APC else ''}")

    # overall state-dependence: per-cell mean feature profile -> eta^2 by cell type
    ncell = ad.n_obs
    P = np.zeros((ncell, F.shape[1]))
    for ci in range(ncell):
        P[ci] = F[cellid == ci].mean(0)
    groups = [P[cell_type == ct] for ct in np.unique(cell_type) if (cell_type == ct).sum() >= 3]
    eta2 = np.zeros(F.shape[1])
    gt = np.array(cell_type)
    for j in range(F.shape[1]):
        gs = [P[gt == ct, j] for ct in np.unique(gt) if (gt == ct).sum() >= 3]
        grand = np.concatenate(gs).mean()
        ssb = sum(len(g)*(g.mean()-grand)**2 for g in gs)
        sst = sum(((np.concatenate(gs)-grand)**2)); eta2[j] = ssb/(sst+1e-9)
    live = P.std(0) > 1e-6
    print(f"\n==> state-dependence across {int(live.sum())} live features (eta^2 = variance explained by cell type):")
    print(f"    median eta^2 {np.median(eta2[live]):.3f}; features with eta^2>0.3: "
          f"{int((eta2[live]>0.3).sum())} ({100*(eta2[live]>0.3).mean():.0f}%); RFX feature eta^2 = {eta2[feat]:.3f}")
    np.savez_compressed(f"{BASE}/outputs/singlecell/state_features.npz",
                        eta2=eta2, live=live, feat=feat, rfx_by_ct=rows,
                        order=np.array([k for k in rows]))
    print("==> saved state_features.npz")


if __name__ == "__main__":
    main()
