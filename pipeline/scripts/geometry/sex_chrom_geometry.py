#!/usr/bin/env python
"""Identity-axis control — sex-linked genes (XIST/Y) and chromosome geometry, on the
INPUT gene embedding (layer 0), for scPRINT vs AIDO. Unlike the three protein probes,
chromosomal/sex identity is NOT a protein-sequence property, so the ESM prior should
NOT help here (predict scPRINT ~ AIDO) — a specificity control for the ESM-advantage.

    conda activate bae
    python scripts/sex_chrom_geometry.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 0
Y_GENES = ("RPS4Y1 RPS4Y2 DDX3Y UTY KDM5D EIF1AY USP9Y NLGN4Y ZFY TMSB4Y PRKY TXLNGY "
           "TBL1Y PCDH11Y NLGN4Y UTY").split()
SEX_EXTRA = ["XIST", "TSIX"]


def load_embeddings():
    out = {}
    E = np.load(f"{BASE}/outputs/scprint/gene_embedding.npz")["embedding"]
    ids = [l.strip() for l in open(f"{BASE}/outputs/scprint/gene_ids.txt")]
    bm = pd.read_parquet(f"{BASE}/external/scprint_data/biomart_pos.parquet")
    ens2sym = {e: str(s).upper() for e, s in bm["hgnc_symbol"].items()}
    sym2chr = {str(s).upper(): str(c) for s, c in zip(bm["hgnc_symbol"], bm["chromosome_name"])}
    d = {}
    for i, e in enumerate(ids):
        s = ens2sym.get(e)
        if s and not s.startswith("ENSG") and s not in d:
            d[s] = E[i]
    out["scPRINT"] = d
    Ea = np.load(f"{BASE}/outputs/aido/gene_embedding.npz")["embedding"]
    ag = [l.split("\t")[0].upper() for l in open(f"{BASE}/external/scprint_data/aido_genes.tsv").read().splitlines()[1:]]
    out["AIDO"] = {g: Ea[i] for i, g in enumerate(ag) if i < len(Ea)}
    return out, sym2chr


def auroc(scores, labels):
    labels = np.asarray(labels, bool); r = rankdata(scores)
    npos = labels.sum(); n = len(labels)
    return (r[labels].sum() - npos*(npos+1)/2) / (npos*(n-npos))


def main():
    rng = np.random.default_rng(SEED)
    emb, sym2chr = load_embeddings()
    panel = sorted(set(emb["scPRINT"]) & set(emb["AIDO"]))
    pset = set(panel); sym2idx = {s: i for i, s in enumerate(panel)}

    sex_genes = [g for g in (Y_GENES + SEX_EXTRA) if g in pset]
    print(f"==> {len(sex_genes)} sex-linked genes in panel: {sex_genes}")

    # chromosome labels for the panel
    chrom = np.array([sym2chr.get(g, "NA") for g in panel])
    autos = [str(i) for i in range(1, 23)]
    # same-chromosome pair test on a subsample (autosomes with >=50 genes)
    big = [c for c in autos if (chrom == c).sum() >= 50]

    for model in ("scPRINT", "AIDO"):
        M = np.stack([emb[model][g] for g in panel]).astype(np.float64)
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)

        # (1) sex-linked separability: cosine-to-sex-centroid separates sex genes
        cen = Mn[[sym2idx[g] for g in sex_genes]].mean(0)
        cen /= np.linalg.norm(cen) + 1e-9
        score = Mn @ cen
        y = np.array([g in set(sex_genes) for g in panel])
        au_sex = auroc(score, y)
        # mutual clustering: mean pairwise cosine among sex genes vs random sets
        idx = [sym2idx[g] for g in sex_genes]
        sub = Mn[idx]; intra = (sub @ sub.T)[np.triu_indices(len(idx), 1)].mean()
        rand_intra = []
        for _ in range(200):
            r = rng.choice(len(panel), len(idx), replace=False)
            s = Mn[r]; rand_intra.append((s @ s.T)[np.triu_indices(len(idx), 1)].mean())
        rand_m, rand_s = np.mean(rand_intra), np.std(rand_intra)
        z = (intra - rand_m) / (rand_s + 1e-9)

        # (2) same-chromosome pairs closer than random? (identity backbone)
        pos, neg = [], []
        for _ in range(20000):
            i, j = rng.integers(0, len(panel), 2)
            if i == j or chrom[i] not in big:
                continue
            (pos if chrom[i] == chrom[j] else neg).append((i, j))
        pc = np.array([(Mn[i]*Mn[j]).sum() for i, j in pos])
        nc = np.array([(Mn[i]*Mn[j]).sum() for i, j in neg])
        n = min(len(pc), len(nc))
        au_chr = auroc(np.concatenate([pc[:n], nc[:n]]), [1]*n + [0]*n)

        print(f"\n===== {model} =====")
        print(f"  SEX-linked: AUROC(sep. sex genes) = {au_sex:.3f} | "
              f"intra-cosine {intra:+.3f} vs random {rand_m:+.3f} (z={z:+.1f})")
        print(f"  CHROMOSOME: AUROC(same-chr pairs closer) = {au_chr:.3f}  ({n} pos/neg pairs)")
    print("\n[control expectation] sex/chromosome are NOT protein-sequence properties -> "
          "ESM prior should give scPRINT no special advantage here (unlike complexes/PPI/localization)")


if __name__ == "__main__":
    main()
