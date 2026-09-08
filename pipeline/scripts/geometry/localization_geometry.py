#!/usr/bin/env python
"""Third leg of the ESM-prior story — subcellular LOCALIZATION geometry (HPA).
A purely protein-level property (signal peptides, transmembrane domains, targeting
sequences) that ESM-2 knows but expression-only training cannot see. Linear probe
(logistic, 5-fold CV) from each model's gene embedding to HPA compartments; per
compartment AUROC. Hypothesis: scPRINT (ESM-augmented) >> AIDO (from scratch),
sharpest for membrane / ER / mitochondria / secretory compartments.

    conda activate bae
    python scripts/localization_geometry.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HPA = f"{BASE}/external/hpa/subcellular_location.tsv"

# fine HPA main-locations -> coarse compartments
COARSE = {
    "Nucleus": ["Nucleoplasm", "Nuclear bodies", "Nuclear membrane", "Nuclear speckles",
                "Nucleoli", "Nucleoli fibrillar center", "Nucleoli rim", "Kinetochore",
                "Mitotic chromosome"],
    "Cytosol": ["Cytosol", "Cytoplasmic bodies", "Rods & Rings", "Aggresome"],
    "Plasma membrane": ["Plasma membrane", "Cell Junctions"],
    "Mitochondria": ["Mitochondria"],
    "ER": ["Endoplasmic reticulum"],
    "Golgi": ["Golgi apparatus"],
    "Vesicles": ["Vesicles", "Lysosomes", "Peroxisomes", "Endosomes", "Lipid droplets"],
    "Cytoskeleton": ["Actin filaments", "Centrosome", "Centriolar satellite", "Microtubules",
                     "Intermediate filaments", "Microtubule ends", "Cytokinetic bridge",
                     "Midbody", "Midbody ring", "Focal adhesion sites", "Cleavage furrow"],
}
FINE2COARSE = {f: c for c, fs in COARSE.items() for f in fs}


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
    hpa = pd.read_csv(HPA, sep="\t")
    gene2comp = {}
    for g, main in zip(hpa["Gene name"], hpa["Main location"]):
        if not isinstance(main, str):
            continue
        comps = {FINE2COARSE[x] for x in main.split(";") if x in FINE2COARSE}
        if comps:
            gene2comp[str(g).upper()] = comps

    emb = load_embeddings()
    panel = sorted(set(emb["scPRINT"]) & set(emb["AIDO"]) & set(gene2comp))
    comps = list(COARSE)
    Y = np.array([[c in gene2comp[g] for c in comps] for g in panel])
    keep = Y.sum(0) >= 40
    comps = [c for c, k in zip(comps, keep) if k]; Y = Y[:, keep]
    print(f"==> panel {len(panel)} labelled genes; compartments: "
          + ", ".join(f"{c}({int(Y[:,i].sum())})" for i, c in enumerate(comps)))

    res = {}
    for model in ("scPRINT", "AIDO"):
        M = np.stack([emb[model][g] for g in panel]).astype(np.float64)
        M = (M - M.mean(0)) / (M.std(0) + 1e-9)
        aus = []
        for i, c in enumerate(comps):
            y = Y[:, i]
            pr = cross_val_predict(LogisticRegression(max_iter=1000, C=1.0),
                                   M, y, cv=5, method="predict_proba")[:, 1]
            aus.append(roc_auc_score(y, pr))
        res[model] = np.array(aus)
        print(f"\n===== {model} =====  macro-AUROC {res[model].mean():.3f}")
        for c, a in sorted(zip(comps, res[model]), key=lambda t: -t[1]):
            print(f"   {c:<16} AUROC {a:.3f}")

    print("\n==> scPRINT − AIDO per compartment (ESM-prior advantage):")
    for c, a, b in sorted(zip(comps, res["scPRINT"], res["AIDO"]), key=lambda t: -(t[1]-t[2])):
        print(f"   {c:<16} {a:.3f} vs {b:.3f}   Δ {a-b:+.3f}")
    np.savez_compressed(f"{BASE}/outputs/singlecell/localization.npz",
                        comps=np.array(comps), scprint=res["scPRINT"], aido=res["AIDO"])
    print("\n==> saved to outputs/singlecell/localization.npz")


if __name__ == "__main__":
    main()
