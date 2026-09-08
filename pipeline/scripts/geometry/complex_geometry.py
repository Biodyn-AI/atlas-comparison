#!/usr/bin/env python
"""Sharper PPI probe — curated protein COMPLEXES (EBI Complex Portal) vs embedding
geometry. Do subunits of the SAME physical complex sit closer than random pairs?
Cleaner than STRING (no co-expression/text channels). Hypothesis: the ESM-augmented
scPRINT encodes complex geometry, from-scratch AIDO does not.

    conda activate bae
    python scripts/complex_geometry.py
"""
from __future__ import annotations
import os, re, sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CP = f"{BASE}/external/corum/cp.tsv"
HGNC = f"{BASE}/external/corum/hgnc.txt"
ACC = re.compile(r"^[A-NR-Z][0-9][A-Z0-9]{3}[0-9]$|^[OPQ][0-9][A-Z0-9]{3}[0-9]$")
SEED = 0


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


def load_complexes(panel):
    hg = pd.read_csv(HGNC, sep="\t", dtype=str, low_memory=False)
    uni2sym = {}
    for sym, ups in zip(hg["symbol"], hg["uniprot_ids"]):
        if isinstance(ups, str):
            for u in ups.split("|"):
                uni2sym[u.strip()] = str(sym).upper()
    cp = pd.read_csv(CP, sep="\t", dtype=str)
    complexes = {}
    for name, mol in zip(cp["Recommended name"], cp["Identifiers (and stoichiometry) of molecules in complex"]):
        if not isinstance(mol, str):
            continue
        syms = set()
        for tok in mol.split("|"):
            acc = tok.split("(")[0].strip()
            if ACC.match(acc) and acc in uni2sym:
                s = uni2sym[acc]
                if s in panel:
                    syms.add(s)
        if 3 <= len(syms) <= 60:
            complexes[name] = syms
    return complexes


def auroc(scores, labels):
    labels = np.asarray(labels, bool)
    r = rankdata(scores); npos = labels.sum(); n = len(labels)
    return (r[labels].sum() - npos*(npos+1)/2) / (npos*(n-npos))


def main():
    rng = np.random.default_rng(SEED)
    emb = load_embeddings()
    panel = sorted(set(emb["scPRINT"]) & set(emb["AIDO"]))
    pset = set(panel); sym2idx = {s: i for i, s in enumerate(panel)}
    complexes = load_complexes(pset)
    print(f"==> panel {len(panel)} genes; {len(complexes)} complexes (3-60 subunits) with >=3 members in panel")

    pos = set()
    for syms in complexes.values():
        ss = sorted(syms)
        for a in range(len(ss)):
            for b in range(a+1, len(ss)):
                pos.add((ss[a], ss[b]))
    pos = list(pos)
    negs = []
    while len(negs) < len(pos):
        i, j = rng.integers(0, len(panel), 2)
        if i == j:
            continue
        key = (min(panel[i], panel[j]), max(panel[i], panel[j]))
        if key not in pos:
            negs.append(key)
    negs = list(set(negs))
    print(f"==> {len(pos)} same-complex pairs vs {len(negs)} random pairs")

    for model in ("scPRINT", "AIDO"):
        M = np.stack([emb[model][s] for s in panel]).astype(np.float64)
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        Mc = M - M.mean(0)
        U, S, _ = np.linalg.svd(Mc, full_matrices=False); coords = U * S

        def cos(pairs):
            ii = [sym2idx[a] for a, b in pairs]; jj = [sym2idx[b] for a, b in pairs]
            return (Mn[ii] * Mn[jj]).sum(1)

        pc, nc = cos(pos), cos(negs)
        au = auroc(np.concatenate([pc, nc]), [1]*len(pc) + [0]*len(nc))
        print(f"\n===== {model} =====")
        print(f"  AUROC(cosine sep. same-complex pairs) = {au:.3f}   "
              f"mean cos: complex {pc.mean():+.3f} vs random {nc.mean():+.3f}")

        # which complexes are geometrically tightest (interpretability)
        rows = []
        for name, syms in complexes.items():
            ss = sorted(syms)
            pr = [(ss[a], ss[b]) for a in range(len(ss)) for b in range(a+1, len(ss))]
            rows.append((name, len(ss), cos(pr).mean()))
        rows.sort(key=lambda r: -r[2])
        print("  tightest complexes (mean intra-cosine):")
        for name, n, c in rows[:6]:
            print(f"     {c:+.3f}  [{n:>2}] {name[:60]}")

        # which SVD axes carry complex structure
        def axis_auroc(pairs, a):
            ii = [sym2idx[x] for x, y in pairs]; jj = [sym2idx[y] for x, y in pairs]
            ni = [sym2idx[x] for x, y in negs]; nj = [sym2idx[y] for x, y in negs]
            pp = -np.abs(coords[ii, a]-coords[jj, a]); pn = -np.abs(coords[ni, a]-coords[nj, a])
            return auroc(np.concatenate([pp, pn]), [1]*len(pp)+[0]*len(pn))
        ax = np.array([axis_auroc(pos, a) for a in range(12)])
        top = np.argsort(ax)[::-1][:3]
        print("  top complex axes (SV#):", [int(t)+1 for t in top], "vals", [f"{ax[t]:.2f}" for t in top])
        np.savez_compressed(f"{BASE}/outputs/singlecell/complex_{model}.npz", auroc=au, axis_auroc=ax)
    print("\n==> saved to outputs/singlecell/")


if __name__ == "__main__":
    main()
