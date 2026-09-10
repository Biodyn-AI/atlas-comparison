#!/usr/bin/env python
"""Causal test of the predicted gene pairs on genome-wide Perturb-seq (Replogle et al. 2022).

This is the only test here that is neither a database nor the literature. If a model says genes A
and B belong to the same feature, then knocking A down should move B. We ask exactly that.

Data   : CRISPRi pseudobulk, K562 (11,258 perturbations x 8,248 genes) and RPE1 as replication.
         X is each perturbation's normalised expression change relative to controls.
Read-out: for an ordered pair (A perturbed, B measured), the percentile of |X[A,B]| WITHIN row A --
         among all measured genes, how strongly did B respond to perturbing A? Being a within-row
         rank, this is immune to how globally disruptive perturbation A is (essential genes move
         everything).
Null   : B is replaced by a random gene matched on its own responsiveness (mean |X| across all
         perturbations, matched in 20 quantile bins), so a gene cannot score by being reactive to
         everything. Row effects are handled by the ranking, column effects by the matching.
Also   : does agreement between models raise the causal hit rate, as it does for TRRUST?
    python scripts/hypothesis_perturb.py [NPERM] -> outputs/atlas/comparative/hypothesis_perturb.json
"""
from __future__ import annotations
import h5py, json, sys
import numpy as np
from collections import defaultdict

BASE = "/Users/annaantipova/Desktop/biomech"
C = f"{BASE}/outputs/atlas/comparative"; PD = f"{BASE}/outputs/atlas/perturb"
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200
TOPQ = 0.05                       # "B is among the 5% most-moved genes when A is knocked down"
NBIN = 20
rng = np.random.default_rng(0)

PAIRS = json.load(open(f"{C}/hypothesis_pairs_full.json"))["pairs"]


def load(fn):
    f = h5py.File(f"{PD}/{fn}", "r")
    gt = [x.decode() if isinstance(x, bytes) else str(x) for x in f["obs/gene_transcript"][:]]
    pert = [s.split("_")[1].upper() for s in gt]
    cats = [x.decode() if isinstance(x, bytes) else str(x) for x in f["var/__categories/gene_name"][:]]
    genes = [cats[i].upper() for i in f["var/gene_name"][:]]
    X = f["X"][:]
    X[~np.isfinite(X)] = 0.0
    # average the rows of a gene that was targeted by several constructs
    byg = defaultdict(list)
    for i, g in enumerate(pert): byg[g].append(i)
    names = sorted(byg)
    M = np.empty((len(names), X.shape[1]), np.float32)
    for i, g in enumerate(names): M[i] = X[byg[g]].mean(axis=0)
    return names, {g: i for i, g in enumerate(names)}, genes, {g: i for i, g in enumerate(genes)}, M


def run(fn, label):
    pnames, pix, gnames, gix, M = load(fn)
    A = np.abs(M)
    print(f"\n{label}: {A.shape[0]} perturbed genes x {A.shape[1]} measured genes", flush=True)

    # within-row percentile of every entry
    order = np.argsort(A, axis=1)
    pct = np.empty_like(A)
    ranks = np.arange(A.shape[1], dtype=np.float32) / (A.shape[1] - 1)
    for i in range(A.shape[0]): pct[i, order[i]] = ranks

    # responsiveness bins for the measured genes
    resp = A.mean(axis=0)
    edges = np.quantile(resp, np.linspace(0, 1, NBIN + 1)[1:-1])
    gbin = np.digitize(resp, edges)
    bins = defaultdict(list)
    for j, b in enumerate(gbin): bins[b].append(j)
    bins = {b: np.array(v) for b, v in bins.items()}

    # every testable ordered pair, keeping the number of models that predicted it
    rows, cols, nmod = [], [], []
    for c in PAIRS:
        a, b = c["pair"][0].upper(), c["pair"][1].upper()
        for u, v in ((a, b), (b, a)):
            if u in pix and v in gix:
                rows.append(pix[u]); cols.append(gix[v]); nmod.append(c["n_models"])
    rows = np.array(rows); cols = np.array(cols); nmod = np.array(nmod)
    print(f"  testable ordered pairs: {len(rows)}", flush=True)
    if not len(rows): return None

    obs_pct = pct[rows, cols]
    obs_top = float((obs_pct >= 1 - TOPQ).mean())

    null_mean, null_top = [], []
    for _ in range(NPERM):
        rc = np.empty_like(cols)
        for b, members in bins.items():
            sel = gbin[cols] == b
            if sel.any(): rc[sel] = rng.choice(members, sel.sum(), replace=True)
        v = pct[rows, rc]
        null_mean.append(float(v.mean())); null_top.append(float((v >= 1 - TOPQ).mean()))
    nm = np.array(null_mean); nt = np.array(null_top)

    out = {"n_ordered_pairs": int(len(rows)),
           "mean_percentile": round(float(obs_pct.mean()), 4),
           "null_mean_percentile": round(float(nm.mean()), 4),
           "p_mean": round((int((nm >= obs_pct.mean()).sum()) + 1) / (NPERM + 1), 5),
           "top5pct_rate": round(obs_top, 4), "null_top5pct_rate": round(float(nt.mean()), 4),
           "top5pct_fold": round(obs_top / max(float(nt.mean()), 1e-9), 2),
           "z_top5": round(float((obs_top - nt.mean()) / (nt.std() or 1)), 2),
           "p_top5": round((int((nt >= obs_top).sum()) + 1) / (NPERM + 1), 5)}
    print(f"  mean within-row percentile {out['mean_percentile']:.4f} vs null "
          f"{out['null_mean_percentile']:.4f}  (p={out['p_mean']})", flush=True)
    print(f"  B lands in the top {int(TOPQ*100)}% of moved genes: {100*obs_top:.2f}% of pairs vs "
          f"{100*nt.mean():.2f}% null  ->  {out['top5pct_fold']}x, z={out['z_top5']}, "
          f"p={out['p_top5']}", flush=True)

    by_k = {}
    for k in sorted(set(nmod.tolist())):
        sel = nmod >= k
        if sel.sum() < 30: continue
        r = float((obs_pct[sel] >= 1 - TOPQ).mean())
        # null for THIS subset: same rows, responsiveness-matched replacements for its own genes
        sub_r, sub_c = rows[sel], cols[sel]
        sn = []
        for _ in range(NPERM):
            rc = np.empty_like(sub_c)
            for b, members in bins.items():
                m2 = gbin[sub_c] == b
                if m2.any(): rc[m2] = rng.choice(members, m2.sum(), replace=True)
            sn.append(float((pct[sub_r, rc] >= 1 - TOPQ).mean()))
        sn = np.array(sn)
        by_k[int(k)] = {"n": int(sel.sum()), "top5pct_rate": round(r, 4),
                        "null_rate": round(float(sn.mean()), 4),
                        "fold": round(r / max(float(sn.mean()), 1e-9), 2),
                        "z": round(float((r - sn.mean()) / (sn.std() or 1)), 2),
                        "p_emp": round((int((sn >= r).sum()) + 1) / (NPERM + 1), 5)}
        d = by_k[int(k)]
        print(f"    >={k} models: n={int(sel.sum()):>5}  top-5% {100*r:>5.2f}% vs null "
              f"{100*sn.mean():>5.2f}%  ->  {d['fold']}x, z={d['z']}, p={d['p_emp']}", flush=True)
    out["by_n_models"] = by_k
    return out


res = {"data": "Replogle et al. 2022 genome-wide CRISPRi Perturb-seq, normalised pseudobulk",
       "null": "measured gene replaced by a responsiveness-matched random gene; ranks taken within row",
       "n_perm": NPERM, "top_quantile": TOPQ}
res["K562"] = run("K562_gwps_normalized_bulk_01.h5ad", "K562 (genome-wide)")
res["RPE1"] = run("rpe1_normalized_bulk_01.h5ad", "RPE1 (replication)")
json.dump(res, open(f"{C}/hypothesis_perturb.json", "w"), indent=1)
print("\n==> hypothesis_perturb.json", flush=True)
