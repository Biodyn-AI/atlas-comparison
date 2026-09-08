#!/usr/bin/env python
"""Does the model encode the SIGN of regulation (activator vs repressor)? For TFs that
TRRUST annotates with both Activation and Repression targets, test whether a TF's
activated vs repressed targets separate in the gene embedding (linear probe, per TF,
5-fold CV AUROC). Careful prior: regulatory sign is a property of the (TF, target)
PAIR and context, not of the target gene alone — so this may be a clean null (the
embedding encodes gene identity/function, not signed pairwise relations).

    conda activate bae
    python scripts/regulation_sign.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TFS = ["TP53", "RELA", "NFKB1", "SP1", "STAT3", "AR", "PPARG"]


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


def main():
    tr = pd.read_csv(f"{BASE}/external/single_cell_mechinterp/external/networks/trrust_human.tsv",
                     sep="\t", header=None, names=["tf", "tg", "mode", "p"])
    emb = load_embeddings()

    for model in ("scPRINT", "AIDO"):
        D = emb[model]
        print(f"\n===== {model} =====")
        aucs = []
        for tf in TFS:
            sub = tr[(tr.tf.str.upper() == tf) & (tr["mode"].isin(["Activation", "Repression"]))]
            act = [g.upper() for g in sub[sub["mode"] == "Activation"]["tg"] if g.upper() in D]
            rep = [g.upper() for g in sub[sub["mode"] == "Repression"]["tg"] if g.upper() in D]
            act = list(dict.fromkeys(act)); rep = list(dict.fromkeys(rep))
            if len(act) < 12 or len(rep) < 12:
                print(f"   {tf:<7} act {len(act):>3} rep {len(rep):>3}  — too few"); continue
            X = np.stack([D[g] for g in act + rep]).astype(np.float64)
            X = (X - X.mean(0)) / (X.std(0) + 1e-9)
            y = np.array([1]*len(act) + [0]*len(rep))
            pr = cross_val_predict(LogisticRegression(max_iter=1000, C=0.5), X, y, cv=5,
                                   method="predict_proba")[:, 1]
            au = roc_auc_score(y, pr); aucs.append(au)
            print(f"   {tf:<7} act {len(act):>3} rep {len(rep):>3}  AUROC(activation vs repression) {au:.3f}")
        if aucs:
            print(f"   {'MEAN':<7} {'':>15}  AUROC {np.mean(aucs):.3f}  (0.5 = sign not encoded in the embedding)")


if __name__ == "__main__":
    main()
