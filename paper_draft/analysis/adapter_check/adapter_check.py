"""Adapter check: do our adapters read the models the way the model authors' own code does?

For the 2,000 corpus cells of the atlas's CKA analysis (seed 0), the cell embeddings produced by each
model's OFFICIAL inference code are compared with the per-cell representations our adapter produced
(mean over the cell's gene positions, every layer; out_fixed/cka/<M>_emb.npz). Linear CKA and the
overlap of 15-nearest-neighbour sets are reported for every layer; as a reference, the same scores
for PCA-50 of the raw expression of the same cells.

usage: ATLAS_GPU_BUNDLE=<bundle> python adapter_check.py MODEL
"""
import json, os, subprocess, sys
import numpy as np
import scanpy as sc

BUNDLE = os.path.abspath(os.environ["ATLAS_GPU_BUNDLE"])
OUT = os.path.abspath(os.environ.get("ATLAS_ADAPTER_OUTPUT", os.path.dirname(__file__)))
CACHE = os.path.abspath(os.environ.get("ATLAS_CACHE", os.path.join(os.path.dirname(__file__), "..", ".cache")))
TMP = os.path.join(CACHE, "adapter_check")
CORPUS = os.path.abspath(os.environ.get("ATLAS_GPU_CORPUS", os.path.join(BUNDLE, "external/perturb/tabula_3tissue_6k.h5ad")))
CKA = os.path.abspath(os.environ.get("ATLAS_CKA_EMBEDDINGS", os.path.join(BUNDLE, "out_fixed/cka")))
os.makedirs(OUT, exist_ok=True); os.chdir(BUNDLE); sys.path.insert(0, BUNDLE)
M = sys.argv[1]


def cells():
    a = sc.read_h5ad(CORPUS)
    idx = np.sort(np.random.default_rng(0).choice(a.n_obs, 2000, replace=False))   # = cell_cka.py
    return a[idx].copy()


def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float(np.linalg.norm(X.T @ Y) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def knn_overlap(X, Y, k=15):
    from sklearn.neighbors import NearestNeighbors
    nx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X, return_distance=False)[:, 1:]
    ny = NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(nx, ny)]))


def official_scgpt(ad):
    import adapters.scgpt as A
    A.ScGPTAdapter()._stub_optional_deps()
    for k in [k for k in sys.modules if k == "torchtext" or k.startswith("torchtext.")]:
        del sys.modules[k]
    sys.path.insert(0, os.path.join(BUNDLE, "adapters", "_shims"))
    import scgpt.tasks.cell_emb as ce
    e = ce.embed_data(ad, "external/scGPT_checkpoints/whole-human", gene_col="feature_name", max_length=1200,
                      batch_size=32, device="cuda", use_fast_transformer=False, return_new_adata=True)
    return np.asarray(e.X, dtype=np.float32), "scgpt.tasks.embed_data (CLS embedding, whole-human)"


def official_uce(ad):
    tmp = os.path.join(TMP, "uce_tmp"); os.makedirs(tmp, exist_ok=True)
    p = os.path.join(tmp, "cells2000.h5ad")
    a2 = ad.copy(); a2.var_names = a2.var["feature_name"].astype(str).values; a2.var_names_make_unique(); a2.write_h5ad(p)
    uce = os.path.join(BUNDLE, "external", "UCE")
    cmd = [sys.executable, "eval_single_anndata.py", "--adata_path", p, "--dir", tmp + "/", "--species", "human",
           "--model_loc", os.path.join(uce, "model_files", "33l_8ep_1024t_1280.torch"), "--nlayers", "33",
           "--batch_size", "25"]
    subprocess.run(cmd, cwd=uce, check=True)
    out = [f for f in os.listdir(tmp) if f.endswith("_uce_adata.h5ad")][0]
    e = sc.read_h5ad(os.path.join(tmp, out))
    return np.asarray(e.obsm["X_uce"], dtype=np.float32), "UCE eval_single_anndata.py (33-layer, X_uce)"


def probe(X, y):
    """balanced accuracy of a linear classifier, 5-fold stratified CV (classes with >= 25 cells)"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    keep = np.isin(y, [c for c, n in zip(*np.unique(y, return_counts=True)) if n >= 25])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    pred = cross_val_predict(clf, X[keep], y[keep], cv=StratifiedKFold(5, shuffle=True, random_state=0))
    return round(float(balanced_accuracy_score(y[keep], pred)), 4)


def official_tahoe(ad):
    tmp = os.path.join(TMP, "tahoe_tmp"); os.makedirs(tmp, exist_ok=True)
    p = os.path.join(tmp, "cells2000.h5ad"); ad.write_h5ad(p); o = os.path.join(tmp, "emb.npy")
    code = f"""
import numpy as np, scanpy as sc, torch
from tahoe_x1.model import ComposerTX
from tahoe_x1.tasks.emb_extractor import get_batch_embeddings
model, vocab, model_cfg, collator_cfg = ComposerTX.from_hf(repo_id="tahoebio/Tahoe-x1", model_size="3b")
model = model.eval().to(device="cuda", dtype=torch.bfloat16)
a = sc.read_h5ad("{p}")
stoi = vocab.get_stoi() if hasattr(vocab, "get_stoi") else dict(vocab)
a.var["id_in_vocab"] = [stoi.get(g, -1) for g in a.var["ensembl_id"].astype(str)]
a = a[:, a.var["id_in_vocab"] >= 0].copy()
with torch.autocast("cuda", dtype=torch.bfloat16):
    E = get_batch_embeddings(a, model.model, vocab, model_cfg, collator_cfg, batch_size=8, num_workers=4, max_length=2048)
np.save("{o}", np.asarray(E, dtype=np.float32))
"""
    py = os.environ.get("ATLAS_TAHOE_PYTHON", sys.executable)
    subprocess.run([py, "-c", code], check=True)
    return np.load(o), "tahoe_x1.tasks.emb_extractor.get_batch_embeddings (3b, CLS embedding)"


def official_scfoundation(ad):
    tmp = os.path.join(TMP, "scf_tmp"); os.makedirs(tmp, exist_ok=True)
    d = os.path.join(BUNDLE, "external", "scFoundation", "model")
    import pandas as pd, anndata as an
    from scipy import sparse
    panel = list(pd.read_csv(os.path.join(d, "OS_scRNA_gene_index.19264.tsv"), sep="\t")["gene_name"])
    sym = ad.var["feature_name"].astype(str).values; col = {}
    for j, g in enumerate(sym): col.setdefault(g, j)
    X = sparse.csr_matrix(ad.X); M = np.zeros((ad.n_obs, len(panel)), np.float32)
    for k, g in enumerate(panel):          # the authors' main_gene_selection: their panel order, zeros where absent
        if g in col: M[:, k] = X[:, col[g]].toarray().ravel()
    a2 = an.AnnData(M, obs=pd.DataFrame(index=ad.obs_names), var=pd.DataFrame(index=panel))
    p = os.path.join(tmp, "cells2000.h5ad"); a2.write_h5ad(p)
    subprocess.run([sys.executable, "get_embedding.py", "--task_name", "check", "--input_type", "singlecell",
                    "--output_type", "cell", "--pool_type", "all", "--tgthighres", "t4", "--data_path", p,
                    "--save_path", tmp, "--pre_normalized", "F", "--version", "ce"], cwd=d, check=True)
    f = [x for x in os.listdir(tmp) if x.endswith("_resolution.npy")][0]
    return np.load(os.path.join(tmp, f)).astype(np.float32), "scFoundation get_embedding.py (cell, version ce, pool all, t4)"


os.makedirs(TMP, exist_ok=True)
ad = cells()
cache = os.path.join(TMP, f"{M}_official.npz")
if os.path.exists(cache):
    zc = np.load(cache, allow_pickle=True); E_off, how = zc["E"], str(zc["how"])
else:
    E_off, how = {"scGPT": official_scgpt, "UCE": official_uce, "Tahoe": official_tahoe,
                  "scFoundation": official_scfoundation}[M](ad)
    np.savez(cache, E=E_off, how=how)
z = np.load(os.path.join(CKA, f"{M}_emb.npz"), allow_pickle=True)
nL = len([k for k in z.files if k.startswith("res_")])
# reference: PCA-50 of the raw expression (log1p of counts per 10k, 2,000 most variable genes)
r = ad.copy(); sc.pp.normalize_total(r, 1e4); sc.pp.log1p(r)
sc.pp.highly_variable_genes(r, n_top_genes=2000); r = r[:, r.var.highly_variable]; sc.pp.pca(r, n_comps=50)
P = r.obsm["X_pca"].astype(np.float32)
res = {"model": M, "official": how, "n_cells": int(E_off.shape[0]), "official_dim": int(E_off.shape[1]),
       "reference_pca50": {"cka": round(cka(E_off, P), 4), "knn15": round(knn_overlap(E_off, P), 4)}, "layers": {}}
for L in range(nL):
    X = z[f"res_{L}"].astype(np.float32)
    res["layers"][L] = {"cka": round(cka(E_off, X), 4), "knn15": round(knn_overlap(E_off, X), 4)}
best = max(res["layers"], key=lambda L: res["layers"][L]["cka"])
res["best_layer"] = best; res["last_layer"] = nL - 1
ct = ad.obs["cell_type"].astype(str).to_numpy(); cp = ad.obs["compartment"].astype(str).to_numpy()
res["probe_balanced_accuracy"] = {
    name: {"cell_type": probe(X, ct), "compartment": probe(X, cp)}
    for name, X in (("official", E_off), ("ours_last_layer", z[f"res_{nL - 1}"].astype(np.float32)),
                    ("ours_best_cka_layer", z[f"res_{best}"].astype(np.float32)), ("raw_pca50", P))}
json.dump(res, open(os.path.join(OUT, f"{M}.json"), "w"), indent=1)
print(M, how, "PCA ref", res["reference_pca50"], "last", res["layers"][nL - 1], "best", best, res["layers"][best],
      "probes", res["probe_balanced_accuracy"], flush=True)
