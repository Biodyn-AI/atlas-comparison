#!/usr/bin/env python
"""Closing the caveat — do co-regulated clusters emerge in AIDO's DEEP residual?
The input embedding showed AIDO clusters only HLA (co-reg), not the protein-sequence
families. Co-regulation is contextual, so it should appear DEEPER — but only for
clusters actually co-expressed in the data (PBMC = blood): predict HLA rises with
depth, while HOX/keratins (not expressed in blood) stay flat. Captures AIDO residual
at layers 0/4/8 for cluster genes + background, intra-cluster AUROC per layer.

    conda activate scprint
    python scripts/aido_cluster_depth.py
"""
from __future__ import annotations
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HIST_PREF = ("HIST", "H1-", "H2AC", "H2BC", "H3C", "H4C")
CLUSTERS = {
    "HLA": lambda g: g.startswith("HLA-"),
    "Histones": lambda g: g.startswith(HIST_PREF),
    "HOX": lambda g: bool(re.match(r"^HOX[ABCD]\d+$", g)),
    "Keratins": lambda g: bool(re.match(r"^KRT\d+$", g)),
    "IFN-alpha": lambda g: bool(re.match(r"^IFNA\d+$", g)) or g in {"IFNB1", "IFNW1", "IFNE", "IFNK"},
}


def main():
    import numpy as np, scanpy as sc, torch, torch.nn as nn
    from scipy.stats import rankdata
    from gb_cell.models import CellFoundationModel, CellFoundationConfig
    from gb_cell.utils import align_adata, preprocess_counts

    genes = [l.split("\t")[0].upper() for l in open(f"{BASE}/external/scprint_data/aido_genes.tsv").read().splitlines()[1:]]
    rng = np.random.default_rng(0)
    memb = {name: [i for i, g in enumerate(genes) if fn(g)] for name, fn in CLUSTERS.items()}
    clu_pos = sorted({i for v in memb.values() for i in v})
    bg = rng.choice([i for i in range(len(genes)) if i not in set(clu_pos)], 3000, replace=False)
    keep = sorted(set(clu_pos) | set(bg.tolist()))
    pos2k = {p: k for k, p in enumerate(keep)}
    cl_idx = {name: [pos2k[i] for i in v] for name, v in memb.items()}
    for name, v in memb.items():
        print(f"   {name:<10} {len(v):>3} genes")

    cfg = CellFoundationConfig.from_pretrained(f"{BASE}/ckpt_aido")
    m = CellFoundationModel.from_pretrained(f"{BASE}/ckpt_aido", config=cfg).eval()
    blocks = None
    for _, mod in m.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == cfg.num_hidden_layers:
            blocks = mod; break

    adata = sc.datasets.pbmc3k()[:32].copy()
    ad, attn = align_adata(adata)
    xb = ad.X.toarray() if hasattr(ad.X, "toarray") else ad.X
    inp = preprocess_counts(xb, device="cpu")
    am = torch.cat([torch.from_numpy(attn).unsqueeze(0).repeat(inp.shape[0], 1),
                    torch.ones((inp.shape[0], 2))], 1)

    cap = {}
    kt = torch.as_tensor(keep)
    hks = [blocks[0].register_forward_pre_hook(
        lambda mod, i: cap.__setitem__(0, (i[0] if isinstance(i, tuple) else i)[:, kt, :].detach()))]
    for L, blk in ((4, blocks[3]), (8, blocks[7])):
        hks.append(blk.register_forward_hook(
            lambda mod, i, o, L=L: cap.__setitem__(L, (o[0] if isinstance(o, tuple) else o)[:, kt, :].detach())))
    with torch.no_grad():
        m(input_ids=inp, attention_mask=am)
    for h in hks:
        h.remove()

    def auroc(s, y):
        y = np.asarray(y, bool); r = rankdata(s); npos = y.sum(); n = len(y)
        return (r[y].sum() - npos*(npos+1)/2) / (npos*(n-npos))

    bg_k = [pos2k[i] for i in bg.tolist()]
    print("\n==> intra-cluster proximity AUROC by AIDO layer (0=input .. 8=deep):")
    print(f"    {'cluster':<10} " + "  ".join(f"L{L}" for L in (0, 4, 8)))
    for name, idx in cl_idx.items():
        if len(idx) < 4:
            print(f"    {name:<10} (too few)"); continue
        row = []
        for L in (0, 4, 8):
            Mn = cap[L].mean(0).numpy()
            Mn = Mn / (np.linalg.norm(Mn, axis=1, keepdims=True) + 1e-9)
            sub = Mn[idx]; pc = (sub @ sub.T)[np.triu_indices(len(idx), 1)]
            nb = Mn[bg_k]; nn_ = rng.integers(0, len(bg_k), (4000, 2))
            nc = (nb[nn_[:, 0]] * nb[nn_[:, 1]]).sum(1)
            row.append(auroc(np.concatenate([pc, nc]), [1]*len(pc)+[0]*len(nc)))
        print(f"    {name:<10} " + "  ".join(f"{a:.3f}" for a in row))
    print("\n[predict] HLA (immune, expressed in PBMC) rises with depth = co-regulation; "
          "HOX/keratins (not in blood) stay flat = no context to build on")


if __name__ == "__main__":
    main()
