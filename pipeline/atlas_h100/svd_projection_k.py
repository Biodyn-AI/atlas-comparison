#!/usr/bin/env python
"""Reviewer R2-Major-2 (capacity-matched SAE vs SVD): cumulative projection of each SAE decoder direction onto the
top-k PCA/SVD subspace, across a RANGE of k, per model. For a decoder direction v (unit norm) and the top-k principal
basis B_k (orthonormal, k x d), projected energy = ||B_k v||^2 in [0,1]; we report the mean over all d_sae features.
Baseline = a random unit direction, whose expected energy is exactly k/d (isotropic). If SAE directions accumulate
energy FASTER than k/d they preferentially align with high-variance axes; the curve rising to 1 as k->d shows SVD can
span them given enough axes, so the SAE's edge is sparse allocation at fixed k, not spanning unreachable directions.
Top-k basis from the covariance eigendecomposition (d x d) -> scales to large d without the tall U of full SVD.
    -> svd_projection_k.json
"""
from __future__ import annotations
import json, glob, os, sys
import numpy as np
import torch

R = "out_alllayers"
MODELS = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
KS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
NSUB = 100000  # rows for the covariance estimate


def mid_cat(m):
    cats = sorted(glob.glob(f"{R}/{m}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])
    return json.load(open(cats[len(cats) // 2]))


out = []
for m in MODELS:
    try:
        cat = mid_cat(m); L = cat["layer"]; d_sae = cat["d_sae"]
    except Exception as e:
        print("cat miss", m, e, flush=True); continue
    act = f"{R}/{m}/layer_{L:02d}_activations.npy"
    sp = f"{R}/{m}/sae_L{L:02d}.pt"
    if not (os.path.exists(act) and os.path.exists(sp)):
        print("data miss", m, os.path.exists(act), os.path.exists(sp), flush=True); continue
    A = np.asarray(np.load(act, mmap_mode="r"))[:NSUB].astype(np.float64)
    d = A.shape[1]
    Ac = A - A.mean(0)
    # top-k principal basis via covariance eigendecomposition (columns = directions, ascending eigenvalue)
    C = Ac.T @ Ac
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    V = V[:, order]                      # [d, d], columns = principal directions, desc variance
    total_var = float(w.sum())
    # SAE decoder directions, unit-norm columns  [d, d_sae]
    ck = torch.load(sp, map_location="cpu")
    Wdec = ck["state_dict"]["W_dec.weight"].to(torch.float64)          # [d, d_sae]
    Wn = torch.nn.functional.normalize(Wdec, dim=0).numpy()
    proj_full = V.T @ Wn                 # [d, d_sae]; energy of each feature along each principal axis
    energy = proj_full ** 2             # rows already ordered desc; cumulative over rows = top-k subspace energy
    cum = np.cumsum(energy, axis=0)      # [d, d_sae]
    row = {"model": m, "layer": int(L), "d_model": int(d), "d_sae": int(d_sae), "curve": {}}
    for k in KS:
        if k > d:
            continue
        sae_mean = float(cum[k - 1].mean())            # mean over features of energy in top-k subspace
        svd_var = float(w[order][:k].sum() / total_var)  # variance explained by top-k (for reference)
        row["curve"][str(k)] = {"sae_proj": round(sae_mean, 4), "random_baseline": round(k / d, 4),
                                "svd_var_at_k": round(svd_var, 4)}
    out.append(row)
    c = row["curve"]
    msg = " ".join(f"k{k}:{c[str(k)]['sae_proj']}(rnd{c[str(k)]['random_baseline']})" for k in KS if str(k) in c)
    print(f"{m:12} d={d:5} d_sae={d_sae:6} | {msg}", flush=True)

json.dump(out, open("svd_projection_k.json", "w"), indent=1)
print("==> svd_projection_k.json", flush=True)
