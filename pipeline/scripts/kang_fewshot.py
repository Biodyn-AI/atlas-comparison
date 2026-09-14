import numpy as np, scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
# reproduce the SAME 3040 subsample as the scGPT run (seed 0, same grouping)
ad=sc.read_h5ad("data/kang/kang_batch2.h5ad")
rng=np.random.default_rng(0); idx=[]
for (s,d),g in ad.obs.groupby(["stim","ind"],observed=True):
    idx+=list(rng.choice(g.index.values,min(len(g),190),replace=False))
ad=ad[idx].copy()
sg=np.load("outputs/singlecell/kang_scgpt.npz")
assert np.array_equal((ad.obs.stim.values=="stim").astype(int), sg["stim"]), "alignment mismatch!"
Zscg=sg["Z"]
# PCA on same cells
a=ad.copy(); sc.pp.filter_genes(a,min_cells=5); sc.pp.normalize_total(a,target_sum=1e4); sc.pp.log1p(a)
sc.pp.highly_variable_genes(a,n_top_genes=2000,subset=True); sc.pp.scale(a,max_value=10); sc.pp.pca(a,n_comps=50)
Zpca=a.obsm["X_pca"]
emb={"PCA-HVG":Zpca,"scGPT":Zscg}
ct=ad.obs["cell"].astype(str).values; don=ad.obs["ind"].values.astype(str)
donors=np.unique(don); train_d=set(donors[:4]); test_d=set(donors[4:])
tr_mask=np.array([d in train_d for d in don]); te_mask=~tr_mask
types=[c for c in np.unique(ct) if (ct[tr_mask]==c).sum()>=20 and (ct[te_mask]==c).sum()>=5]
print(f"few-shot cell-type annotation: {len(types)} types, train donors {sorted(train_d)}, test donors {sorted(test_d)}")
def macro_auroc(Ztr,ytr,Zte,yte,classes):
    aus=[]
    for c in classes:
        if (ytr==c).sum()<2 or (yte==c).sum()<1: continue
        clf=LogisticRegression(max_iter=1000,class_weight="balanced").fit(Ztr,(ytr==c).astype(int))
        aus.append(roc_auc_score((yte==c).astype(int),clf.predict_proba(Zte)[:,1]))
    return np.mean(aus)
print(f"\n{'k/type':>6} " + "  ".join(f"{n:>10}" for n in emb))
for k in (2,5,10,20):
    row={n:[] for n in emb}
    for rep in range(10):
        r=np.random.default_rng(rep); sel=[]
        for c in types:
            ci=np.where(tr_mask & (ct==c))[0]
            sel+=list(r.choice(ci,min(k,len(ci)),replace=False))
        sel=np.array(sel); yte=ct[te_mask]
        for n,Z in emb.items():
            row[n].append(macro_auroc(Z[sel],ct[sel],Z[te_mask],yte,types))
    print(f"{k:>6} " + "  ".join(f"{np.mean(row[n]):.3f}±{np.std(row[n]):.02f}" for n in emb))
# ceiling check: stim/ctrl few-shot (train 1 donor, test rest)
print("\nstim/ctrl few-shot (train on 1 donor, test on other 7), mean over donors:")
y=(ad.obs.stim.values=="stim").astype(int)
for n,Z in emb.items():
    aus=[]
    for d in donors:
        tr=don==d; 
        if 0<y[tr].sum()<tr.sum():
            clf=LogisticRegression(max_iter=1000,class_weight="balanced").fit(Z[tr],y[tr])
            aus.append(roc_auc_score(y[~tr],clf.predict_proba(Z[~tr])[:,1]))
    print(f"   {n:<9} AUROC {np.mean(aus):.3f}")
