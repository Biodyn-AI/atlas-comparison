#!/usr/bin/env python
"""Resolving the chromosome paradox — co-regulated GENOMIC CLUSTERS (HLA / histones /
HOX / protocadherins / keratins / IFN-alpha). Whole chromosomes are NOT encoded (~0.5,
they aren't functionally coherent), but genes in a co-regulated cluster should sit
close because they ARE functionally/regulatorily coherent. Tests each cluster's intra-
cosine + AUROC in the gene embedding, scPRINT vs AIDO. (Static input embedding, so PBMC
expression is irrelevant — this is what each model learned about the genes overall.)

    conda activate bae
    python scripts/cluster_geometry.py
"""
from __future__ import annotations
import os, re, sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 0

HIST_PREF = ("HIST", "H1-", "H2AC", "H2BC", "H3C", "H4C")
def is_hist(g): return g.startswith(HIST_PREF)
def is_hla(g):  return g.startswith("HLA-")
def is_hox(g):  return bool(re.match(r"^HOX[ABCD]\d+$", g))
def is_pcdh(g): return bool(re.match(r"^PCDH[ABG]", g))
def is_krt(g):  return bool(re.match(r"^KRT\d+$", g))
def is_ifna(g): return bool(re.match(r"^IFNA\d+$", g)) or g in {"IFNB1", "IFNW1", "IFNE", "IFNK"}
CLUSTERS = {"HLA (MHC, chr6)": is_hla, "Histones (chr6/1)": is_hist, "HOX (chr7/17/12/2)": is_hox,
            "Protocadherins (chr5)": is_pcdh, "Keratins (chr17/12)": is_krt, "IFN-alpha (chr9)": is_ifna}


def load_embeddings():
    out = {}
    E = np.load(f"{BASE}/outputs/scprint/gene_embedding.npz")["embedding"]
    ids = [l.strip() for l in open(f"{BASE}/outputs/scprint/gene_ids.txt")]
    bm = pd.read_parquet(f"{BASE}/external/scprint_data/biomart_pos.parquet")
    ens2sym = {e: str(s).upper() for e, s in bm["hgnc_symbol"].items()}
    d = {}
    for i, e in enumerate(ids):
        s = ens2sym.get(e)
        if s and not s.startswith("ENSG") and s not in d:
            d[s] = E[i]
    out["scPRINT"] = d
    Ea = np.load(f"{BASE}/outputs/aido/gene_embedding.npz")["embedding"]
    ag = [l.split("\t")[0].upper() for l in open(f"{BASE}/external/scprint_data/aido_genes.tsv").read().splitlines()[1:]]
    out["AIDO"] = {g: Ea[i] for i, g in enumerate(ag) if i < len(Ea)}
    return out


def auroc(scores, labels):
    labels = np.asarray(labels, bool); r = rankdata(scores)
    npos = labels.sum(); n = len(labels)
    return (r[labels].sum() - npos*(npos+1)/2) / (npos*(n-npos))


def main():
    rng = np.random.default_rng(SEED)
    emb = load_embeddings()
    panel = sorted(set(emb["scPRINT"]) & set(emb["AIDO"]))
    sym2idx = {s: i for i, s in enumerate(panel)}
    members = {name: [g for g in panel if fn(g)] for name, fn in CLUSTERS.items()}
    for name, ms in members.items():
        print(f"   {name:<24} {len(ms):>3} genes in panel")

    # shared random negative pairs
    negs = set()
    while len(negs) < 20000:
        i, j = rng.integers(0, len(panel), 2)
        if i != j:
            negs.add((min(i, j), max(i, j)))
    negs = list(negs)

    for model in ("scPRINT", "AIDO"):
        M = np.stack([emb[model][g] for g in panel]).astype(np.float64)
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        nc = np.array([(Mn[i]*Mn[j]).sum() for i, j in negs])
        print(f"\n===== {model} =====   (random-pair mean cosine {nc.mean():+.3f})")
        all_pos = []
        for name, ms in members.items():
            if len(ms) < 4:
                print(f"   {name:<24} — too few"); continue
            idx = [sym2idx[g] for g in ms]
            sub = Mn[idx]; pc = (sub @ sub.T)[np.triu_indices(len(idx), 1)]
            all_pos.append(pc)
            au = auroc(np.concatenate([pc, nc]), [1]*len(pc) + [0]*len(nc))
            print(f"   {name:<24} intra-cos {pc.mean():+.3f}  AUROC {au:.3f}  (vs whole-chromosome ~0.50)")
        pooled = np.concatenate(all_pos)
        au_all = auroc(np.concatenate([pooled, nc]), [1]*len(pooled) + [0]*len(nc))
        print(f"   {'ALL clusters pooled':<24} AUROC {au_all:.3f}")
    print("\n[contrast] whole-chromosome proximity was ~0.50 (chrom not encoded) — "
          "co-regulated clusters should be >> that if the geometry is functional, not positional")


if __name__ == "__main__":
    main()
