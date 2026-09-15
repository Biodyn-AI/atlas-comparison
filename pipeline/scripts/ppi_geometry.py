#!/usr/bin/env python
"""L2 extension beyond TF — PPI geometry vs STRING (the SV2-4 tier).
Tests whether physical protein-interaction confidence (STRING) is encoded in the
geometric proximity of gene embeddings, for scPRINT (ESM-2-augmented) and AIDO.Cell
(from-scratch). Hypothesis: the ESM protein prior should make PPI geometry STRONGER
in scPRINT than AIDO — even more than the TF-regulon tier — and PPI vs TF should live
on different SVD axes (reproducing the tiered SV1>SV2-4>SV5-7 structure). CPU, reuses
the already-captured gene_embedding.npz for both models.

    conda activate bae
    python scripts/ppi_geometry.py
"""
from __future__ import annotations
import gzip, os, sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRING_INFO = f"{BASE}/external/string/9606.protein.info.v12.0.txt.gz"
STRING_LINKS = f"{BASE}/external/string/9606.protein.links.v12.0.txt.gz"
HI = 700           # STRING combined-score threshold for a high-confidence PPI positive
SEED = 0


def load_embeddings():
    """Return {model: (symbol->vec dict)} for scPRINT and AIDO."""
    out = {}
    # scPRINT: ENSG rows -> symbol via biomart
    E = np.load(f"{BASE}/outputs/scprint/gene_embedding.npz")["embedding"]
    ids = [l.strip() for l in open(f"{BASE}/outputs/scprint/gene_ids.txt")]
    bm = pd.read_parquet(f"{BASE}/external/scprint_data/biomart_pos.parquet")
    ens2sym = {e: str(s).upper() for e, s in bm["hgnc_symbol"].items()}
    d = {}
    for i, e in enumerate(ids):
        s = ens2sym.get(e)
        if s and s.startswith("ENSG") is False and s not in d:
            d[s] = E[i]
    out["scPRINT"] = d
    # AIDO: rows already symbols
    Ea = np.load(f"{BASE}/outputs/aido/gene_embedding.npz")["embedding"]
    ag = [l.split("\t")[0].upper() for l in open(f"{BASE}/external/scprint_data/aido_genes.tsv").read().splitlines()[1:]]
    out["AIDO"] = {g: Ea[i] for i, g in enumerate(ag) if i < len(Ea)}
    return out


def load_string(panel_syms):
    ensp2sym = {}
    with gzip.open(STRING_INFO, "rt") as f:
        next(f)
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            ensp2sym[p[0]] = p[1].upper()
    keep = {e for e, s in ensp2sym.items() if s in panel_syms}
    edges = {}
    with gzip.open(STRING_LINKS, "rt") as f:
        next(f)
        for ln in f:
            a, b, sc = ln.split()
            if a in keep and b in keep:
                sa, sb = ensp2sym[a], ensp2sym[b]
                if sa != sb:
                    edges[(min(sa, sb), max(sa, sb))] = int(sc)
    return edges


def auroc(scores, labels):
    labels = np.asarray(labels, bool)
    r = rankdata(scores); npos = labels.sum(); n = len(labels)
    return (r[labels].sum() - npos*(npos+1)/2) / (npos*(n-npos))


def main():
    rng = np.random.default_rng(SEED)
    emb = load_embeddings()
    panel = sorted(set(emb["scPRINT"]) & set(emb["AIDO"]))
    print(f"==> shared gene panel (scPRINT ∩ AIDO): {len(panel)} symbols")
    edges = load_string(set(panel))
    hi_pairs = [p for p, s in edges.items() if s >= HI]
    print(f"==> STRING edges within panel: {len(edges)} scored, {len(hi_pairs)} high-conf (>={HI})")

    # TRRUST TF-target pairs within panel (for the TF-vs-PPI axis contrast)
    tr = pd.read_csv(f"{BASE}/external/single_cell_mechinterp/external/networks/trrust_human.tsv",
                     sep="\t", header=None, names=["tf", "tg", "m", "p"])
    pset = set(panel)
    tf_pairs = {(min(a, b), max(a, b)) for a, b in zip(tr.tf.str.upper(), tr.tg.str.upper())
                if a in pset and b in pset and a != b}
    tf_pairs = list(tf_pairs)

    # sampled non-edge negatives (shared across models & concepts)
    edge_set = set(edges) | set(tf_pairs)
    negs = []
    while len(negs) < max(len(hi_pairs), len(tf_pairs)) * 2:
        i, j = rng.integers(0, len(panel), 2)
        if i == j:
            continue
        key = (min(panel[i], panel[j]), max(panel[i], panel[j]))
        if key not in edge_set:
            negs.append(key)
    negs = list(set(negs))
    sym2idx = {s: i for i, s in enumerate(panel)}

    for model in ("scPRINT", "AIDO"):
        M = np.stack([emb[model][s] for s in panel]).astype(np.float64)
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)   # for cosine
        Mc = M - M.mean(0)
        U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
        coords = U * S                                                # per-gene SVD coords

        def cos(pairs):
            ii = [sym2idx[a] for a, b in pairs]; jj = [sym2idx[b] for a, b in pairs]
            return (Mn[ii] * Mn[jj]).sum(1)

        # (1) PPI: cosine proximity separates high-conf STRING pairs from non-edges
        pos_c = cos(hi_pairs); neg_c = cos(negs)
        au_ppi = auroc(np.concatenate([pos_c, neg_c]),
                       [1]*len(pos_c) + [0]*len(neg_c))
        # graded: spearman(cosine, STRING score) over all scored panel pairs
        scored = list(edges.items())
        sc_cos = cos([p for p, _ in scored]); sc_val = np.array([s for _, s in scored])
        rho = spearmanr(sc_cos, sc_val).statistic
        # (2) TF: same, cosine separates TRRUST pairs from non-edges
        tf_c = cos(tf_pairs)
        au_tf = auroc(np.concatenate([tf_c, neg_c]), [1]*len(tf_c) + [0]*len(neg_c))

        print(f"\n===== {model} =====")
        print(f"  PPI  : AUROC(cosine sep. high-conf STRING) = {au_ppi:.3f} | "
              f"Spearman(cosine, STRING score) = {rho:.3f}")
        print(f"  TF   : AUROC(cosine sep. TRRUST pairs)      = {au_tf:.3f}")

        # monotonic grading by STRING-confidence bin (reproduces 'graded proximity')
        bins = [(150, 300), (300, 500), (500, 700), (700, 900), (900, 1000)]
        print("  STRING-confidence bin -> mean cosine proximity:")
        for lo, hi in bins:
            pr = [p for p, s in edges.items() if lo <= s < hi]
            if pr:
                print(f"     {lo:>3}-{hi:<3}: {cos(pr).mean():+.3f}  (n={len(pr)})")

        # which SVD axes carry PPI vs TF (per-axis AUROC using 1-D proximity)
        def axis_auroc(pairs, a):
            ii = [sym2idx[x] for x, y in pairs]; jj = [sym2idx[y] for x, y in pairs]
            prox_pos = -np.abs(coords[ii, a] - coords[jj, a])
            ni = [sym2idx[x] for x, y in negs]; nj = [sym2idx[y] for x, y in negs]
            prox_neg = -np.abs(coords[ni, a] - coords[nj, a])
            return auroc(np.concatenate([prox_pos, prox_neg]),
                         [1]*len(prox_pos) + [0]*len(prox_neg))
        ppi_ax = np.array([axis_auroc(hi_pairs, a) for a in range(12)])
        tf_ax = np.array([axis_auroc(tf_pairs, a) for a in range(12)])
        print("  top PPI axes (SV#):", [int(x)+1 for x in np.argsort(ppi_ax)[::-1][:3]],
              "vals", [f"{ppi_ax[x]:.2f}" for x in np.argsort(ppi_ax)[::-1][:3]])
        print("  top TF  axes (SV#):", [int(x)+1 for x in np.argsort(tf_ax)[::-1][:3]],
              "vals", [f"{tf_ax[x]:.2f}" for x in np.argsort(tf_ax)[::-1][:3]])
        np.savez_compressed(f"{BASE}/outputs/singlecell/ppi_{model}.npz",
                            ppi_axis_auroc=ppi_ax, tf_axis_auroc=tf_ax,
                            au_ppi=au_ppi, au_tf=au_tf, rho=rho)
    print("\n==> saved per-model PPI/TF axis profiles to outputs/singlecell/")


if __name__ == "__main__":
    main()
