#!/usr/bin/env python
"""Host-response prediction with scGPT — skeleton for two framings:
  (1) CELL-level classification  (responder / infected / stimulated vs resting)
  (3) PATIENT-level outcome       (severity / vaccine response) via aggregated cells

Robust path first: use scGPT as a frozen encoder -> per-cell embedding, then a light
classifier with LEAVE-ONE-DONOR-OUT CV and strong baselines. A full fine-tune upgrade
is described at the bottom. Fill the TODO paths for your dataset (Kang2018 / Liao2020 /
Stephenson2021 — see the table).

    conda create -n scgpt python=3.10 && conda activate scgpt
    pip install scgpt scanpy scikit-learn   # + a scGPT whole-human checkpoint
    python scripts/host_response_scgpt.py
"""
from __future__ import annotations
import numpy as np
import scanpy as sc
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

# --- config -----------------------------------------------------------------
ADATA_PATH   = "TODO/host_response.h5ad"      # AnnData with raw counts
MODEL_DIR    = "TODO/scGPT_human"             # whole-human scGPT checkpoint dir
GENE_COL     = "feature_name"                 # var column with gene SYMBOLS (scGPT vocab)
CELL_LABEL   = "condition"                    # framing 1: e.g. ctrl/stim, infected/bystander
PATIENT_KEY  = "patient"                      # framing 3: donor/patient id
PATIENT_LABEL= "severity"                     # framing 3: e.g. mild/severe
DONOR_KEY    = "patient"                      # split unit — NEVER split by cell


# --- 0. load + minimal preprocessing (scGPT wants log-normalized, symbol genes) ----
def load():
    adata = sc.read_h5ad(ADATA_PATH)
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if GENE_COL not in adata.var:
        adata.var[GENE_COL] = adata.var_names           # assume var_names are symbols
    return adata


# --- 1. scGPT frozen embedding (zero-shot) ----------------------------------
def embed_scgpt(adata):
    import scgpt as scg
    emb = scg.tasks.embed_data(adata, MODEL_DIR, gene_col=GENE_COL,
                               batch_size=64, return_new_adata=False)
    return emb.obsm["X_scGPT"]                            # [n_cells, d]


# --- baselines for a fair comparison ----------------------------------------
def baseline_embeddings(adata):
    out = {}
    a = adata.copy(); sc.pp.highly_variable_genes(a, n_top_genes=2000, subset=True)
    sc.pp.scale(a, max_value=10); sc.pp.pca(a, n_comps=50)
    out["PCA-HVG"] = a.obsm["X_pca"]
    try:
        import scvi
        s = adata.copy(); s.X = s.layers["counts"]
        scvi.model.SCVI.setup_anndata(s, batch_key=DONOR_KEY)
        m = scvi.model.SCVI(s, n_latent=30); m.train(max_epochs=100)
        out["scVI"] = m.get_latent_representation()
    except Exception as e:
        print("  (scVI baseline skipped:", e, ")")
    return out


def leave_one_donor_out(Z, y, donors):
    """AUROC/AUPRC with each donor held out once (the honest split)."""
    y = np.asarray(y); donors = np.asarray(donors)
    oof = np.zeros(len(y), float)
    for d in np.unique(donors):
        te = donors == d
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(Z[~te], y[~te])
        oof[te] = clf.predict_proba(Z[te])[:, 1]
    return roc_auc_score(y, oof), average_precision_score(y, oof)


# --- framing 1: cell-level response classification --------------------------
def run_cell_level(adata, Zdict):
    y = (adata.obs[CELL_LABEL].astype(str) == adata.obs[CELL_LABEL].astype(str).unique()[-1]).values.astype(int)
    donors = adata.obs[DONOR_KEY].values
    print(f"\n== FRAMING 1: cell-level ({CELL_LABEL}) — {y.sum()}/{len(y)} positive ==")
    for name, Z in Zdict.items():
        au, ap = leave_one_donor_out(Z, y, donors)
        print(f"   {name:<10} AUROC {au:.3f}  AUPRC {ap:.3f}  (leave-one-donor-out)")


# --- framing 3: patient-level outcome ---------------------------------------
def run_patient_level(adata, Zdict):
    obs = adata.obs
    pats = obs[PATIENT_KEY].astype(str)
    plabel = obs.groupby(pats)[PATIENT_LABEL].first()
    y = (plabel.astype(str).isin(["severe", "critical", "responder"])).astype(int)  # TODO: your positive class
    print(f"\n== FRAMING 3: patient-level ({PATIENT_LABEL}) — {int(y.sum())}/{len(y)} patients positive ==")
    for name, Z in Zdict.items():
        # aggregate cells -> one vector per patient (mean pooling of embeddings)
        df = pd.DataFrame(Z, index=pats.values)
        P = df.groupby(level=0).mean().loc[plabel.index].values
        yy = y.values
        # leave-one-patient-out (few patients -> LOO)
        oof = np.zeros(len(yy), float)
        for i in range(len(yy)):
            tr = np.arange(len(yy)) != i
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(P[tr], yy[tr]); oof[i] = clf.predict_proba(P[i:i+1])[:, 1]
        au = roc_auc_score(yy, oof) if 0 < yy.sum() < len(yy) else float("nan")
        print(f"   {name:<10} AUROC {au:.3f}  (leave-one-patient-out, n={len(yy)})")


def main():
    adata = load()
    Z = {}
    try:
        Z["scGPT"] = embed_scgpt(adata)
    except Exception as e:
        print("scGPT embedding failed (check checkpoint/env):", e)
    Z.update(baseline_embeddings(adata))
    if CELL_LABEL in adata.obs:
        run_cell_level(adata, Z)
    if PATIENT_KEY in adata.obs and PATIENT_LABEL in adata.obs:
        run_patient_level(adata, Z)


# ============================================================================
# FULL FINE-TUNE UPGRADE (when the frozen probe isn't enough)
# ----------------------------------------------------------------------------
# Framing 1: scGPT annotation-style fine-tune — TransformerModel with a CLS
#   classification decoder (see scGPT repo Tutorial_Annotation.ipynb). Unfreeze
#   the encoder, cross-entropy on the response label off the <cls> token, small LR
#   (1e-4), early-stop on a held-out DONOR. Beats the frozen probe on subtle states.
# Framing 3: two options — (a) attention-pool cells per patient inside the model and
#   fine-tune end-to-end (MIL), or (b) fine-tune cell-level first, then aggregate the
#   fine-tuned embeddings as above. (a) is stronger but data-hungry.
# MECHINTERP BONUS (your wheelhouse): after fine-tuning, train a TopK SAE on the
#   response-fine-tuned residual, find the feature(s) that detect responder cells
#   (state-dependent, like RFX in APCs), then activation-patch/ablate that feature and
#   measure the shift in predicted response genes (L4). Turns the predictor into an
#   INTERPRETABLE host-response circuit.
# ============================================================================

if __name__ == "__main__":
    main()
