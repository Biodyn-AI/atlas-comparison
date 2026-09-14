#!/usr/bin/env python
"""scPRINT Layer-5 (algorithm extraction). Distil a cell-cycle readout from
scPRINT's internals — the E2F1/cell-cycle signal that recurred in L2/L3/L4 — and
validate it against an INDEPENDENT ground truth (scanpy cell-cycle scoring with the
Tirosh S/G2M gene lists). Two extractions:
  (a) a single SAE feature (the E2F1 detector), aggregated per cell -> a 1-D readout;
  (b) a tiny linear probe on the per-cell SAE profile -> cell-cycle score.

    conda activate scprint
    python scripts/scprint_extract.py --layer 4
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tirosh et al. 2016 cell-cycle gene lists (symbols)
S_GENES = ("MCM5 PCNA TYMS FEN1 MCM2 MCM4 RRM1 UNG GINS2 MCM6 CDCA7 DTL PRIM1 UHRF1 "
           "HELLS RFC2 RPA2 NASP RAD51AP1 GMNN WDR76 SLBP CCNE2 UBR7 POLD3 MSH2 ATAD2 "
           "RAD51 RRM2 CDC45 CDC6 EXO1 TIPIN DSCC1 BLM CASP8AP2 USP1 CLSPN POLA1 CHAF1B "
           "BRIP1 E2F8").split()
G2M_GENES = ("HMGB2 CDK1 NUSAP1 UBE2C BIRC5 TPX2 TOP2A NDC80 CKS2 NUF2 CKS1B MKI67 TMPO "
             "CENPF TACC3 SMC4 CCNB2 CKAP2L CKAP2 AURKB BUB1 KIF11 ANP32E TUBB4B GTSE1 "
             "KIF20B HJURP CDCA3 CDC20 TTK CDC25C KIF2C RANGAP1 NCAPD2 DLGAP5 CDCA2 CDCA8 "
             "ECT2 KIF23 HMMR AURKA PSRC1 ANLN LBR CKAP5 CENPE NEK2 G2E3 GAS2L3 CBX5 CENPA").split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="ckpt_scprint/medium-v1.5.ckpt")
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--n-genes", type=int, default=600)
    ap.add_argument("--n-cells", type=int, default=400)
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
    from scipy.stats import rankdata, spearmanr
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_predict
    from scdataloader import Preprocessor, SimpleAnnDataset, Collator
    from torch.utils.data import DataLoader
    from scprint import scPrint
    from mechaudit.sae.topk_sae import SAEConfig, train_sae, feature_activations

    # ---- SAE from the L3 residual + E2F1 feature ------------------------
    X = np.load(os.path.join(args.out, f"residual_L{args.layer}.npz"))["X"]
    gid = np.array([l.strip() for l in open(os.path.join(args.out, f"residual_L{args.layer}_genes.txt"))])
    sae, stats, _ = train_sae(X, SAEConfig(d_sae=args.d_sae, k=args.k, epochs=40), device="cpu", verbose=False)
    F = feature_activations(sae, X, stats, device="cpu")
    bm = pd.read_parquet(args.biomart)
    ens2sym = {e: str(s2).upper() for e, s2 in bm["hgnc_symbol"].items()}
    trrust = pd.read_csv(args.trrust, sep="\t", header=None, names=["tf", "tg", "m", "p"])
    e2f1_targets = {t.upper() for t in trrust[trrust.tf.str.upper() == "E2F1"]["tg"]}
    uniq = np.array(sorted(set(gid))); idx = {g: i for i, g in enumerate(uniq)}
    Gp = np.zeros((len(uniq), F.shape[1]), np.float32); c = np.zeros(len(uniq))
    for row, g in zip(F, gid):
        Gp[idx[g]] += row; c[idx[g]] += 1
    Gp /= np.maximum(c[:, None], 1)
    y = np.array([ens2sym.get(g, "") in e2f1_targets for g in uniq])
    R = np.apply_along_axis(rankdata, 0, Gp); npos = y.sum()
    au = (R[y].sum(0) - npos*(npos+1)/2)/(npos*(len(uniq)-npos))
    e2f1_feat = int(np.abs(au - 0.5).argmax())
    print(f"==> E2F1 SAE feature = {e2f1_feat}")

    # ---- per-cell SAE profile via scPRINT forward -----------------------
    adata = sc.datasets.pbmc3k(); adata.obs["organism_ontology_term_id"] = "NCBITaxon:9606"
    adata = Preprocessor(is_symbol=True, skip_validate=True, min_valid_genes_id=1000,
                         min_nnz_genes=100, filter_gene_by_counts=False)(adata)
    adata = adata[: args.n_cells].copy()
    # ground-truth cell cycle computed ON THIS adata (symbol-renamed copy) — no barcode
    # matching needed; independent of scPRINT.
    cc_ad = adata.copy()
    cc_ad.var_names = adata.var["symbol"].astype(str).values
    cc_ad.var_names_make_unique()
    cc_ad.X = cc_ad.layers["norm"] if "norm" in cc_ad.layers else cc_ad.X
    sc.pp.log1p(cc_ad)
    s = [g for g in S_GENES if g in cc_ad.var_names]
    g2 = [g for g in G2M_GENES if g in cc_ad.var_names]
    sc.tl.score_genes_cell_cycle(cc_ad, s_genes=s, g2m_genes=g2)
    adata.obs[["S_score", "G2M_score", "phase"]] = cc_ad.obs[["S_score", "G2M_score", "phase"]].values
    print(f"==> cell-cycle GT ({len(s)} S + {len(g2)} G2M genes): "
          f"phases {dict(adata.obs['phase'].value_counts())}")

    m = scPrint.load_from_checkpoint(args.checkpoint, precpt_gene_emb=None, transformer="normal"); m.eval()
    sc.pp.highly_variable_genes(adata, n_top_genes=args.n_genes, flavor="seurat_v3")
    fixed = [g for g in adata.var.index[adata.var.highly_variable].tolist() if g in set(m.genes)]
    ds = SimpleAnnDataset(adata, obs_to_output=["organism_ontology_term_id"])
    col = Collator(organisms=m.organisms, valid_genes=m.genes, how="some", genelist=fixed, max_len=0)
    dl = DataLoader(ds, collate_fn=col, batch_size=16, shuffle=False)

    cap = {}
    hh = m.transformer.blocks[args.layer].register_forward_hook(
        lambda mod, i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    cell_profiles, e2f1_scores, cell_embs = [], [], []
    with torch.no_grad():
        for batch in dl:
            gp, expr = batch["genes"], batch["x"]; ng = gp.shape[1]
            cap.clear()
            o = m(gene_pos=gp, expression=expr, req_depth=batch["depth"], depth_mult=expr.sum(1))
            cell_embs.append(np.asarray(o["cell_emb"].cpu()))   # scPRINT cell embedding [B, d]
            hg = cap["h"][:, cap["h"].shape[1] - ng:, :]
            fb = feature_activations(sae, hg.reshape(-1, hg.shape[-1]).numpy(), stats, device="cpu")
            fb = fb.reshape(hg.shape[0], ng, -1)          # [B, ng, d_sae]
            cell_profiles.append(fb.mean(1))               # per-cell mean feature profile
            e2f1_scores.append(fb[:, :, e2f1_feat].mean(1))
    hh.remove()
    P = np.concatenate(cell_profiles); e2f1 = np.concatenate(e2f1_scores)
    CE = np.concatenate(cell_embs)                         # [n_cells, d] scPRINT cell embedding
    g2m_all = adata.obs["G2M_score"].values[: len(e2f1)].astype(float)
    phase_all = adata.obs["phase"].values[: len(e2f1)]
    valid = np.isfinite(g2m_all)
    print(f"==> {len(e2f1)} cells; {valid.sum()} with a cell-cycle score")
    e2f1, P, CE = e2f1[valid], P[valid], CE[valid]
    g2m = g2m_all[valid]
    phase = phase_all[valid]
    cycling = np.array([p in ("S", "G2M") for p in phase])
    print(f"==> {cycling.sum()} cycling (S/G2M) of {len(g2m)}")

    # (a) single-feature readout
    rho, _ = spearmanr(e2f1, g2m)
    auc = roc_auc_score(cycling, e2f1) if 0 < cycling.sum() < len(cycling) else np.nan
    print(f"\n[extraction a] single E2F1 SAE feature (per cell):")
    print(f"   Spearman(feature, G2M score) = {rho:.3f}")
    print(f"   AUROC(feature separates cycling cells) = {auc:.3f}  (random 0.5)")

    # (b) tiny linear probe on the SAE profile -> G2M score (5-fold CV R^2)
    pred = cross_val_predict(Ridge(alpha=10.0), P, g2m, cv=5)
    ss_res = ((g2m - pred) ** 2).sum(); ss_tot = ((g2m - g2m.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    print(f"\n[extraction b] tiny Ridge probe (per-cell SAE profile -> G2M score), 5-fold CV:")
    print(f"   R^2 = {r2:.3f}; params = {P.shape[1]} (vs scPRINT ~12M) → ~{12e6/P.shape[1]:.0f}× smaller")

    # (c) the RIGHT substrate: probe scPRINT's CELL EMBEDDING -> cell cycle
    predc = cross_val_predict(Ridge(alpha=10.0), CE, g2m, cv=5)
    r2c = 1 - ((g2m - predc) ** 2).sum() / ((g2m - g2m.mean()) ** 2).sum()
    from scipy.stats import spearmanr as _sp
    rhoc, _ = _sp(predc, g2m)
    aucc = roc_auc_score(cycling, predc) if 0 < cycling.sum() < len(cycling) else np.nan
    print(f"\n[extraction c] tiny Ridge probe on scPRINT CELL EMBEDDING -> G2M score, 5-fold CV:")
    print(f"   R^2 = {r2c:.3f}, Spearman = {rhoc:.3f}, AUROC(cycling) = {aucc:.3f}; "
          f"params = {CE.shape[1]} → ~{12e6/CE.shape[1]:.0f}× smaller than scPRINT")
    np.savez_compressed(os.path.join(args.out, "extract_cellcycle.npz"),
                        e2f1=e2f1, g2m=g2m, phase=phase, profile=P)
    print("==> saved extraction to", args.out)


if __name__ == "__main__":
    main()
