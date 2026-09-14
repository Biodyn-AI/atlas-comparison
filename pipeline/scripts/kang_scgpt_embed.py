import os, sys, numpy as np, scanpy as sc, pandas as pd
if not hasattr(os,"sched_getaffinity"):
    os.sched_getaffinity=lambda pid=0: set(range(os.cpu_count() or 1))
sys.path.insert(0,"external/single_cell_mechinterp/external/scGPT")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
CKPT="external/single_cell_mechinterp/external/scGPT_checkpoints/whole-human"

ad=sc.read_h5ad("data/kang/kang_batch2.h5ad")
ad.var["gene_name"]=ad.var["sym"].astype(str).values
# stratified subsample ~3000 by (stim,ind) for CPU speed, keep all donors
rng=np.random.default_rng(0); idx=[]
for (s,d),g in ad.obs.groupby(["stim","ind"]):
    take=min(len(g), 190); idx+=list(rng.choice(g.index.values,take,replace=False))
ad=ad[idx].copy(); print("subsample",ad.shape,"donors",ad.obs.ind.nunique())

from scgpt.tasks import embed_data
emb=embed_data(ad, CKPT, gene_col="gene_name", max_length=1200, batch_size=32,
               device="cpu", use_fast_transformer=False, return_new_adata=True)
Z=emb.obsm["X_scGPT"] if "X_scGPT" in emb.obsm else emb.X
np.savez_compressed("outputs/singlecell/kang_scgpt.npz", Z=Z,
                    stim=(ad.obs.stim.values=="stim").astype(int), donors=ad.obs.ind.values.astype(str))
def loo(Zm,y,dn):
    y=np.asarray(y);oof=np.zeros(len(y))
    for d in np.unique(dn):
        te=dn==d; c=LogisticRegression(max_iter=2000,class_weight="balanced").fit(Zm[~te],y[~te]); oof[te]=c.predict_proba(Zm[te])[:,1]
    return roc_auc_score(y,oof),average_precision_score(y,oof)
y=(ad.obs.stim.values=="stim").astype(int); dn=ad.obs.ind.values.astype(str)
au,ap=loo(Z,y,dn); print(f"== scGPT frozen (stim vs ctrl, LODO): AUROC {au:.3f} AUPRC {ap:.3f} ==")
print("(PCA-HVG baseline on full data was 0.996)")
