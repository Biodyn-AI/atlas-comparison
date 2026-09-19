#!/usr/bin/env python
"""How NONLINEAR is each model's representation, and does the SAE linearise it? Decodes a known
biological concept (tissue = obs['compartment']: immune/kidney/lung) from the per-cell embeddings that
cell_cka already saved (res_* residual, sae_* SAE features), via cross-validated probes:

  lin_res : linear (logreg) probe on the raw residual         -> linear baseline
  mlp_res : nonlinear (MLP)   probe on the raw residual        -> nonlinear ceiling
  lin_sae : linear (logreg) probe on the SAE features          -> does the SAE expose it linearly?

PRIMARY, confound-free metric = nonlinearity gap = mlp_res - lin_res (same dimensionality): large gap =
the concept is encoded NONLINEARLY in the residual. SECONDARY = lin_sae vs mlp_res: if the SAE's LINEAR
readout matches the residual's NONLINEAR readout, the SAE has linearised the structure (features are the
right linear units) — noting sae is higher-dim, so read lin_sae relative to mlp_res, not to lin_res.

CPU only, from saved embeddings. Balanced accuracy, 5-fold stratified CV, standardized features.
    python nonlinearity.py --all --out out_ts3       -> out_ts3/nonlinearity/nonlinearity_ts3.json
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from collections import Counter

MODELS = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]


CT_CANDIDATES = ["cell_type", "cell_ontology_class", "celltype", "free_annotation", "cell_type_ontology_term"]


def concepts_for(corpus, max_cells, tissue_key, min_per_class=25):
    """Return {concept_name: y} for tissue + (auto-detected) cell type, on cell_cka's seed-0 subsample.
    Rare classes (< min_per_class cells) are dropped so 5-fold stratified CV is valid; returns a mask too."""
    import scanpy as sc
    ad = sc.read_h5ad(corpus)
    if max_cells and ad.n_obs > max_cells:
        idx = np.sort(np.random.default_rng(0).choice(ad.n_obs, max_cells, replace=False))
    else:
        idx = np.arange(ad.n_obs)
    cols = list(ad.obs.columns)
    keys = {"tissue": tissue_key}
    ct = next((c for c in CT_CANDIDATES if c in cols), None)
    if ct:
        keys["cell_type"] = ct
    print(f"obs columns: {cols}\nusing: {keys}", flush=True)
    out = {}
    for name, k in keys.items():
        y = ad.obs[k].astype(str).values[idx]
        vc = Counter(y); keep = {c for c, n in vc.items() if n >= min_per_class}
        mask = np.array([v in keep for v in y])
        out[name] = (y[mask], mask, sorted(keep))
    return out


def cv_probe(X, y, kind, seed=0):
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score
    yi = LabelEncoder().fit_transform(y)                     # int labels (string labels break MLP early_stopping)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, yi):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        if kind == "lin":
            clf = LogisticRegression(max_iter=2000, C=1.0)
        else:
            clf = MLPClassifier(hidden_layer_sizes=(128,), alpha=1e-3, max_iter=300,
                                early_stopping=True, random_state=seed)
        clf.fit(Xtr, yi[tr])
        accs.append(balanced_accuracy_score(yi[te], clf.predict(Xte)))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--all", action="store_true")
    ap.add_argument("--corpus", default="external/perturb/tabula_3tissue_6k.h5ad")
    ap.add_argument("--out", default="out_ts3")
    ap.add_argument("--tissue-key", default="compartment")
    ap.add_argument("--max-cells", type=int, default=2000)
    ap.add_argument("--layers", default="mid", help="'mid' (rel 2) or 'all'")
    args = ap.parse_args()

    concepts = concepts_for(args.corpus, args.max_cells, args.tissue_key)
    targets = MODELS if args.all else [args.model]
    res = {"concepts": {}}
    for cname, (y, mask, classes) in concepts.items():
        chance = 1.0 / len(classes)
        print(f"\n=== concept '{cname}': {mask.sum()} cells, {len(classes)} classes (chance={chance:.2f}) ===", flush=True)
        cres = {"classes": classes, "n_classes": len(classes), "n_cells": int(mask.sum()),
                "chance": round(chance, 3), "models": {}}
        for m in targets:
            p = os.path.join(args.out, "cka", f"{m}_emb.npz")
            if not os.path.exists(p):
                print(f"{m}: no {p}, skip", flush=True); continue
            e = dict(np.load(p))
            rels = [2] if args.layers == "mid" else sorted(int(k.split("_")[1]) for k in e if k.startswith("res_"))
            per = {}
            for r in rels:
                if f"res_{r}" not in e:
                    continue
                Xr = e[f"res_{r}"][mask]
                lin_res = cv_probe(Xr, y, "lin"); mlp_res = cv_probe(Xr, y, "mlp")
                rec = {"lin_res": round(lin_res[0], 3), "mlp_res": round(mlp_res[0], 3),
                       "gap": round(mlp_res[0] - lin_res[0], 3)}
                if f"sae_{r}" in e:
                    lin_sae = cv_probe(e[f"sae_{r}"][mask], y, "lin")
                    rec["lin_sae"] = round(lin_sae[0], 3)
                per[str(r)] = rec
                print(f"  [{cname}] {m} rel{r}: lin_res={rec['lin_res']} mlp_res={rec['mlp_res']} "
                      f"gap={rec['gap']} lin_sae={rec.get('lin_sae','-')}", flush=True)
            cres["models"][m] = per
        res["concepts"][cname] = cres
    od = os.path.join(args.out, "nonlinearity"); os.makedirs(od, exist_ok=True)
    fp = os.path.join(od, "nonlinearity_ts3.json"); json.dump(res, open(fp, "w"))
    print(f"==> {fp}", flush=True)


if __name__ == "__main__":
    main()
