#!/usr/bin/env python
"""Depth profile of a regulon across AIDO.Cell layers — where does RFX / E2F1 become
decodable? Aggregates each TRRUST gene's residual per layer, linear-probes regulon-
target membership, plots AUROC vs depth. Cross-model claim: AIDO builds regulon
structure DEEP (flat input -> rises), vs scPRINT front-loading it at the input.

    conda activate scprint
    python scripts/aido_depth.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    import numpy as np, pandas as pd, scanpy as sc, torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    from gb_cell.models import CellFoundationModel, CellFoundationConfig
    from gb_cell.utils import align_adata, preprocess_counts

    genes = [l.split("\t")[0] for l in open(f"{BASE}/external/scprint_data/aido_genes.tsv").read().splitlines()[1:]]
    sym2pos = {g: i for i, g in enumerate(genes)}
    tr = pd.read_csv(f"{BASE}/external/single_cell_mechinterp/external/networks/trrust_human.tsv",
                     sep="\t", header=None, names=["tf", "tg", "m", "p"])
    tr_genes = set(tr.tf.str.upper()) | set(tr.tg.str.upper())
    keep = sorted(sym2pos[g] for g in tr_genes if g in sym2pos)
    keep_sym = np.array([genes[i] for i in keep])
    rfx = {t.upper() for tf in ("RFXANK", "RFXAP", "RFX5") for t in tr[tr.tf.str.upper() == tf]["tg"]}
    e2f1 = {t.upper() for t in tr[tr.tf.str.upper() == "E2F1"]["tg"]}
    y_rfx = np.array([g in rfx for g in keep_sym]); y_e2f = np.array([g in e2f1 for g in keep_sym])
    print(f"==> {len(keep)} TRRUST genes; RFX targets {y_rfx.sum()}, E2F1 targets {y_e2f.sum()}")

    cfg = CellFoundationConfig.from_pretrained(f"{BASE}/ckpt_aido")
    m = CellFoundationModel.from_pretrained(f"{BASE}/ckpt_aido", config=cfg).eval()
    adata = sc.datasets.pbmc3k()[:32].copy()
    ad, attn = align_adata(adata)
    xb = ad.X.toarray() if hasattr(ad.X, "toarray") else ad.X
    inp = preprocess_counts(xb, device="cpu")
    am = torch.cat([torch.from_numpy(attn).unsqueeze(0).repeat(inp.shape[0], 1),
                    torch.ones((inp.shape[0], 2))], 1)
    with torch.no_grad():
        out = m(input_ids=inp, attention_mask=am, output_hidden_states=True)
    hs = out.hidden_states                                   # tuple: embedding + each layer

    def probe(G, y):
        G = (G - G.mean(0)) / (G.std(0) + 1e-9)
        pr = cross_val_predict(LogisticRegression(max_iter=1000, C=1.0), G, y, cv=5,
                               method="predict_proba")[:, 1]
        return roc_auc_score(y, pr)

    rfx_au, e2f_au = [], []
    for L, h in enumerate(hs):
        G = h[:, keep, :].mean(0).numpy()                    # [n_genes, d] mean over cells
        rfx_au.append(probe(G, y_rfx)); e2f_au.append(probe(G, y_e2f))
        print(f"   layer {L:>2}: RFX {rfx_au[-1]:.3f}  E2F1 {e2f_au[-1]:.3f}")
    np.savez_compressed(f"{BASE}/outputs/singlecell/depth_AIDO.npz",
                        rfx=np.array(rfx_au), e2f1=np.array(e2f_au))
    print("==> saved depth_AIDO.npz")


if __name__ == "__main__":
    main()
