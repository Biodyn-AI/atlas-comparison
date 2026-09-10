#!/usr/bin/env python
"""#2 — null-test the annotation-FREE survivors (same rigour as the retracted core).
(A) CKA: is the cross-model representational similarity meaningful, or what any two 2000-cell embeddings give?
    Real linear CKA vs a CELL-SHUFFLE null (permute cells in one embedding -> destroys correspondence).
(B) SVD-vs-SAE variance: is the SAE's edge over PCA real, or just overcompleteness? Compare, at matched sparsity
    k, variance-explained of: top-k SVD, the TRAINED SAE, and a RANDOM-init TopK dictionary (untrained, same shape).
    -> cka_svd_null.json
"""
from __future__ import annotations
import json, glob, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.sae import TopKSAE, SAECfg

R = "out_alllayers"
import itertools
ALL = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
PAIRS = list(itertools.combinations(ALL, 2))   # all 45 pairs, not a hand-picked four
SVD_MODELS = ALL                                # all ten, not a hand-picked three
rng = np.random.default_rng(0)


def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return (np.linalg.norm(Y.T @ X) ** 2) / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y) + 1e-12)


def mid_res(m):
    d = np.load(f"{R}/cka/{m}_emb.npz", allow_pickle=True)
    Ls = list(d["layers"]); L = Ls[len(Ls) // 2]
    return d[f"res_{L}"]


# (A) CKA null
ckaA = []
emb = {}
for m in set([x for p in PAIRS for x in p]):
    try:
        emb[m] = mid_res(m)
    except Exception as e:
        print("emb miss", m, e)
for a, b in PAIRS:
    if a not in emb or b not in emb:
        continue
    X, Y = emb[a], emb[b]
    n = min(len(X), len(Y)); X, Y = X[:n], Y[:n]
    real = float(cka(X, Y))
    shuf = float(np.mean([cka(X, Y[rng.permutation(n)]) for _ in range(20)]))
    ckaA.append({"pair": f"{a}-{b}", "real_cka": round(real, 3), "shuffled_cka": round(shuf, 4)})
    print(f"CKA {a}-{b}: real {real:.3f} vs cell-shuffled {shuf:.4f}", flush=True)

# (B) SVD vs SAE vs random-dict var-explained at matched sparsity
def var_expl(A, recon):
    return 1.0 - ((A - recon) ** 2).sum() / ((A - A.mean(0)) ** 2).sum()

svdB = []
for m in SVD_MODELS:
    cats = sorted(glob.glob(f"{R}/{m}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])
    cat = json.load(open(cats[len(cats) // 2])); L = cat["layer"]; k = cat.get("k", 32); d_sae = cat["d_sae"]
    act = f"{R}/{m}/layer_{L:02d}_activations.npy"
    if not os.path.exists(act):
        print("act miss", m); continue
    A = np.asarray(np.load(act, mmap_mode="r"))[:200000].astype(np.float32)
    Ac = A - A.mean(0)
    # SVD var@k
    U, S, Vt = np.linalg.svd(Ac, full_matrices=False)
    svd_k = float((S[:k] ** 2).sum() / (S ** 2).sum())
    # trained SAE var-explained
    sp = f"{R}/{m}/sae_L{L:02d}.pt"
    sae_ve = None
    if os.path.exists(sp):
        ck = torch.load(sp, map_location="cpu"); sae_ve = float(ck.get("stats", {}).get("var_explained", 0))
    # random-init TopK dict, same shape (untrained)
    d = A.shape[1]
    torch.manual_seed(0)
    sae = TopKSAE(d, d_sae, k)
    with torch.no_grad():
        xb = torch.as_tensor(A[:20000] - A.mean(0), dtype=torch.float32)
        recon = sae.decode(sae.encode(xb)).numpy()
    rand_ve = float(var_expl(A[:20000] - A.mean(0), recon))
    svdB.append({"model": m, "k": k, "svd_var_at_k": round(svd_k, 3),
                 "sae_var_explained": round(sae_ve, 3) if sae_ve is not None else None,
                 "random_dict_var_explained": round(rand_ve, 3)})
    print(f"SVD/SAE {m}: SVD@{k} {svd_k:.3f} | SAE {sae_ve} | random-dict {rand_ve:.3f}", flush=True)

json.dump({"cka_null": ckaA, "svd_null": svdB}, open("cka_svd_null.json", "w"), indent=1)
print("==> cka_svd_null.json", flush=True)
