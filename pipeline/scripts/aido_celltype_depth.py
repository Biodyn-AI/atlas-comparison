#!/usr/bin/env python
"""The cell manifold (L5 lens) — is CELL-TYPE identity encoded, and how does it emerge
with depth? All our probes so far are gene-token level; this is the cell representation
(mean-pooled residual per cell). Closes the L5 story: the manifold lens gave a null on
cell cycle (resting PBMC), but cell TYPE is present in blood and should be extractable.
Probes T/B/Mono/NK cell type from AIDO's pooled residual at layers 0/4/8.

    conda activate scprint
    python scripts/aido_celltype_depth.py
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MARKERS = {"T": ["CD3D", "CD3E", "CD3G", "TRAC", "IL7R"], "B": ["MS4A1", "CD79A", "CD79B", "CD19"],
           "Mono": ["CD14", "LYZ", "FCN1", "S100A8", "S100A9"], "NK": ["NKG7", "GNLY", "KLRD1", "KLRF1"],
           "DC": ["FCER1A", "CST3", "CLEC10A"]}


def main():
    import numpy as np, scanpy as sc, torch, torch.nn as nn
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    from gb_cell.models import CellFoundationModel, CellFoundationConfig
    from gb_cell.utils import align_adata, preprocess_counts

    N = 64
    adata = sc.datasets.pbmc3k()[:N].copy()
    raw = adata.copy(); sc.pp.normalize_total(raw, target_sum=1e4); sc.pp.log1p(raw)
    def z(v): return (v - v.mean()) / (v.std() + 1e-9)
    S = []
    for ct, ms in MARKERS.items():
        pres = [g for g in ms if g in raw.var_names]
        X = raw[:, pres].X; X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        S.append(np.mean([z(X[:, j]) for j in range(X.shape[1])], 0))
    S = np.stack(S, 1); cts = np.array(list(MARKERS))
    cell_type = np.where(S.max(1) > 0.2, cts[S.argmax(1)], "other")
    print("==> cell types:", {c: int((cell_type == c).sum()) for c in np.unique(cell_type)})

    cfg = CellFoundationConfig.from_pretrained(f"{BASE}/ckpt_aido")
    m = CellFoundationModel.from_pretrained(f"{BASE}/ckpt_aido", config=cfg).eval()
    blocks = next(mod for _, mod in m.named_modules()
                  if isinstance(mod, nn.ModuleList) and len(mod) == cfg.num_hidden_layers)
    ad, attn = align_adata(adata)
    ng = len(attn)
    present = torch.as_tensor(np.where(attn[:ng] > 0)[0])           # attended (present) gene positions
    attn_t = torch.from_numpy(attn).unsqueeze(0)

    pooled = {0: [], 4: [], 8: []}
    cap = {}
    def mk(L):
        def hook(mod, i, o=None):
            h = (i[0] if o is None else o)
            h = h[0] if isinstance(h, tuple) else h
            cap[L] = h[:, present, :].mean(1).detach()               # [b, d] pooled over present genes
        return hook
    with torch.no_grad():
        for s in range(0, ad.n_obs, 8):
            xb = ad.X[s:s+8]; xb = xb.toarray() if hasattr(xb, "toarray") else xb
            inp = preprocess_counts(xb, device="cpu")
            am = torch.cat([attn_t.repeat(inp.shape[0], 1), torch.ones((inp.shape[0], 2))], 1)
            h0 = blocks[0].register_forward_pre_hook(lambda mod, i: mk(0)(mod, i))
            h4 = blocks[3].register_forward_hook(lambda mod, i, o: mk(4)(mod, i, o))
            h8 = blocks[7].register_forward_hook(lambda mod, i, o: mk(8)(mod, i, o))
            cap.clear(); m(input_ids=inp, attention_mask=am)
            for L in (0, 4, 8):
                pooled[L].append(cap[L].numpy())
            for h in (h0, h4, h8):
                h.remove()

    np.savez_compressed(f"{BASE}/outputs/singlecell/celltype_pooled.npz",
                        L0=np.concatenate(pooled[0]), L4=np.concatenate(pooled[4]),
                        L8=np.concatenate(pooled[8]), cell_type=cell_type)
    print("==> saved pooled residuals -> celltype_pooled.npz (probes recomputable offline)")

    keep = np.array([c != "other" for c in cell_type])
    y = cell_type[keep]
    labs = [c for c in np.unique(y) if (y == c).sum() >= 10]
    print(f"==> probing cell types {labs} from pooled residual (n={keep.sum()})")
    print(f"    {'layer':<6} macro-AUROC  per-type")
    for L in (0, 4, 8):
        M = np.concatenate(pooled[L])[keep]
        M = (M - M.mean(0)) / (M.std(0) + 1e-9)
        aus = {}
        for c in labs:
            yy = (y == c)
            pr = cross_val_predict(LogisticRegression(max_iter=1000), M, yy, cv=5,
                                   method="predict_proba")[:, 1]
            aus[c] = roc_auc_score(yy, pr)
        print(f"    L{L:<5} {np.mean(list(aus.values())):.3f}       " +
              "  ".join(f"{c} {a:.2f}" for c, a in aus.items()))
    print("\n[expect] cell-TYPE identity is present in blood -> should be strong and sharpen with depth "
          "(contrast with the cell-CYCLE null on resting PBMC)")


if __name__ == "__main__":
    main()
