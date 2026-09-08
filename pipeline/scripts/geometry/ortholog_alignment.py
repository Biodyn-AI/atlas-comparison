#!/usr/bin/env python
"""Ortholog alignment — unique to scPRINT (human + mouse in one vocab). Does a human
gene sit near its mouse ortholog once the species offset is removed? Tests biological
fidelity of the cross-species representation: does the model 'know' human GENE X = mouse
Gene x. Metrics: AUROC(cosine separates ortholog vs random cross-species pairs) and
retrieval rank of the true ortholog — raw vs species-mean-centered.

    conda activate bae
    python scripts/ortholog_alignment.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    from scipy.stats import rankdata
    E = np.load(f"{BASE}/outputs/scprint/gene_embedding.npz")["embedding"].astype(np.float64)
    ids = [l.strip() for l in open(f"{BASE}/outputs/scprint/gene_ids.txt")]
    id2row = {g: i for i, g in enumerate(ids)}

    # human ENSG -> symbol ; mouse ENSMUSG -> symbol
    bm = pd.read_parquet(f"{BASE}/external/scprint_data/biomart_pos.parquet")
    hsym2ens = {}
    for ens, s in bm["hgnc_symbol"].items():
        s = str(s).upper()
        if not s.startswith("ENSG"):
            hsym2ens.setdefault(s, ens)
    msym2ens = {}
    for ln in open(f"{BASE}/external/ortholog/mrk_ensembl.rpt"):
        p = ln.rstrip("\n").split("\t")
        if len(p) > 6 and p[5].startswith("ENSMUSG"):
            msym2ens.setdefault(p[1].upper(), p[5])

    # 1:1 ortholog pairs from MGI homology classes
    cls = {}
    with open(f"{BASE}/external/ortholog/hom.rpt") as f:
        next(f)
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 4:
                continue
            key, org, sym = p[0], p[1], p[3].upper()
            cls.setdefault(key, {"human": [], "mouse": []})
            if org == "human":
                cls[key]["human"].append(sym)
            elif org.startswith("mouse"):
                cls[key]["mouse"].append(sym)

    pairs = []
    for k, d in cls.items():
        if len(d["human"]) == 1 and len(d["mouse"]) == 1:
            he, me = hsym2ens.get(d["human"][0]), msym2ens.get(d["mouse"][0])
            if he in id2row and me in id2row:
                pairs.append((id2row[he], id2row[me]))
    pairs = list(dict.fromkeys(pairs))
    hrows = np.array([h for h, _ in pairs]); mrows = np.array([m for _, m in pairs])
    print(f"==> {len(pairs)} 1:1 ortholog pairs present in scPRINT vocab")

    # all human / mouse rows in vocab (for retrieval + species means)
    hset = np.array([i for g, i in id2row.items() if g.startswith("ENSG")])
    mset = np.array([i for g, i in id2row.items() if g.startswith("ENSMUSG")])
    hmean, mmean = E[hset].mean(0), E[mset].mean(0)

    def norm(X): return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    def evaluate(Emb, tag):
        Hn = norm(Emb[hrows]); Mn = norm(Emb[mrows])
        orth_cos = (Hn * Mn).sum(1)
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(mrows))
        rand_cos = (Hn * norm(Emb[mrows[perm]])).sum(1)
        s = np.concatenate([orth_cos, rand_cos]); y = np.r_[np.ones(len(orth_cos)), np.zeros(len(rand_cos))]
        r = rankdata(s); npos = y.sum(); au = (r[y == 1].sum() - npos*(npos+1)/2)/(npos*(len(y)-npos))
        # retrieval: rank true ortholog among ALL mouse-vocab genes
        Mall = norm(Emb[mset]); m_pos = {m: k for k, m in enumerate(mset)}
        samp = rng.choice(len(pairs), min(1500, len(pairs)), replace=False)
        ranks = []
        for si in samp:
            h, m = hrows[si], mrows[si]
            sims = Mall @ norm(Emb[h:h+1])[0]
            ranks.append(int((sims > sims[m_pos[m]]).sum()) + 1)
        ranks = np.array(ranks)
        print(f"  [{tag}] AUROC(ortholog vs random x-species) {au:.3f} | "
              f"median rank {int(np.median(ranks))}/{len(mset)} | "
              f"top-1 {100*(ranks==1).mean():.1f}% | top-10 {100*(ranks<=10).mean():.1f}%")
        np.savez_compressed(f"{BASE}/outputs/singlecell/ortholog_{tag}.npz",
                            ranks=ranks, auroc=au, n_mouse=len(mset))

    print("orthologs vs random cross-species pairs, and retrieval of the true mouse ortholog:")
    evaluate(E, "raw")
    Ec = E.copy(); Ec[hset] -= hmean; Ec[mset] -= mmean
    evaluate(Ec, "species-centered")
    print(f"\n(species offset ||hmean-mmean|| = {np.linalg.norm(hmean-mmean):.3f}; "
          f"mean |emb| = {np.linalg.norm(E, axis=1).mean():.3f})")


if __name__ == "__main__":
    main()
