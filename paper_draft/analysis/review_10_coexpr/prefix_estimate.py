"""Estimate how many corpus cells each model's SAE saw (cap 500,000 gene positions, batches of 8,
stop at the first batch that reaches the cap: pipeline/atlas_h100/common/pipeline.py:13-26).
Tokens per cell = min(#genes with count>0, per-model cap) (07 section 2.1 table). Two gene sets for
#genes: (a) all 61,497 genes (upper bound on tokens -> lower bound on cells); (b) the co-firing union
U expressed in >=10 cells (18,250 genes, close to model vocabularies). UCE samples 1,024 fixed.
Also counts named (non-ENSG) genes expressed in >=10 cells (unrestricted universe for review_heavy.py)."""
import json, os, pickle, sys
import numpy as np, anndata as ad
from scipy import sparse
H = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("ATLAS_BASE", os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "..")))
A = ad.read_h5ad(sys.argv[1]); X = sparse.csr_matrix(A.X)
names = np.array([str(s).upper() for s in A.var_names])
PRED, _ = pickle.load(open(f"{BASE}/outputs/atlas/comparative/.cofire_cache.pkl", "rb"))
GENES = sorted({g for m in PRED for e in PRED[m] for g in e})
nexpr = np.asarray((X > 0).sum(0)).ravel()
first = {}
for i, s in enumerate(names): first.setdefault(s, i)
U = [first[g] for g in GENES if g in first and nexpr[first[g]] >= 10]
nnz_all = np.diff(X.indptr)
nnz_U = np.asarray((X[:, U] > 0).sum(1)).ravel()
caps = {"tGPT": 64, "C2S": 512, "UCE": None, "scGPT": 1200, "Geneformer": 2048, "Tahoe": 2048, "MaxToki": 2048, "GeneCompass": 2048}
def cells_seen(tok, cap=500000, bs=8):
    tot = 0
    for s in range(0, len(tok), bs):
        tot += int(tok[s:s + bs].sum())
        if tot >= cap: return s + bs
    return len(tok)
out = {"rule": __doc__.strip().splitlines()[0], "per_model": {}}
for m, c in caps.items():
    if c is None:
        ka = kb = cells_seen(np.full(len(nnz_all), 1024))
    else:
        ka = cells_seen(np.minimum(nnz_all, c)); kb = cells_seen(np.minimum(nnz_U, c))
    out["per_model"][m] = {"cap": c, "cells_seen_tokens_all_genes": ka, "cells_seen_tokens_U_genes": kb}
named = np.array([not s.startswith("ENSG") for s in names])
first_mask = np.zeros(len(names), bool); first_mask[list(first.values())] = True
out["named_genes_expr_ge10"] = int((named & first_mask & (nexpr >= 10)).sum())
out["U_genes"] = len(U)
out["median_nnz_all"] = float(np.median(nnz_all)); out["median_nnz_U"] = float(np.median(nnz_U))
json.dump(out, open(os.path.join(H, "prefix_estimate.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
