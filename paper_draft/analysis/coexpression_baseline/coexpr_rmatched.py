#!/usr/bin/env python
"""Co-expression-level null: how many TRRUST hits would a graph get from its pairs' co-expression
level alone?

For all pairs of the gene set U (co-firing union, expressed in >= 10 cells), bin Pearson r (bins of
0.01). In each bin b, the TRRUST rate is q_b = (# TRRUST edges in U with r in b) / (# U pairs in b).
For any graph, the expected hits from co-expression level alone are sum over its pairs of q_{bin(r)},
and the null distribution is sum_b Binomial(n_b, q_b) (n_b = the graph's pairs in bin b), drawn
NSIM times with a fixed seed. p = (#sim >= obs + 1)/(NSIM + 1). This null keeps each pair's
co-expression level but NOT gene degree (the configuration null keeps degree but not co-expression);
the two are complementary. Applied to each model's co-firing graph (within U), to its part outside the
top 5 % co-expressed pairs, and, as a sanity check, to a co-expression top-N graph.

usage: python coexpr_rmatched.py <corpus.h5ad> [NSIM=10000] [kinds=pearson,withincomp]
"""
from __future__ import annotations
import json, os, pickle, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("ATLAS_BASE", os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "..")))  # repository root with the release archives unpacked
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
CORPUS = sys.argv[1]
NSIM = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
KINDS = (sys.argv[3] if len(sys.argv) > 3 else "pearson,withincomp").split(",")
MIN_CELLS, NB, BLOCK = 10, 200, 512
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

TRUTH = set()
for a, bs in json.load(open(f"{G}/trrust_edges.json")).items():
    for b in bs:
        u, v = sorted((str(a).upper(), str(b).upper()))
        if u != v: TRUTH.add((u, v))
PRED, _ = pickle.load(open(f"{C}/.cofire_cache.pkl", "rb"))
GENES = sorted({g for m in ORDER for e in PRED[m] for g in e})

import anndata as ad
from scipy import sparse
A = ad.read_h5ad(CORPUS)
X = sparse.csr_matrix(A.X, dtype=np.float32)
names = [str(s).upper() for s in A.var_names]; comp = A.obs["compartment"].astype(str).to_numpy()
name_idx = {}
for i, s in enumerate(names): name_idx.setdefault(s, i)
nexpr = np.asarray((X > 0).sum(0)).ravel(); tot = np.asarray(X.sum(1)).ravel(); ncell = X.shape[0]
UG = [g for g in GENES if g in name_idx and nexpr[name_idx[g]] >= MIN_CELLS]; uidx = {g: i for i, g in enumerate(UG)}; ng = len(UG)
Xc = X[:, [name_idx[g] for g in UG]].tocsc()
L = np.asarray(Xc.multiply((1e4 / np.maximum(tot, 1)).astype(np.float32)[:, None]).todense(), dtype=np.float32)
del Xc, X, A
np.log1p(L, out=L)
TRU = sorted({p for p in TRUTH if p[0] in uidx and p[1] in uidx})
tb = lambda r: np.clip(((r + 1) * NB / 2).astype(np.int32), 0, NB - 1)


def pair_r(Z, pairs):
    a = np.array([uidx[p[0]] for p in pairs]); b = np.array([uidx[p[1]] for p in pairs])
    out = np.empty(len(pairs), np.float32)
    for s in range(0, len(pairs), 4000):
        out[s:s + 4000] = (Z[:, a[s:s + 4000]] * Z[:, b[s:s + 4000]]).sum(0) / ncell
    return out


res = {"method": __doc__.strip().splitlines()[0], "n_sim": NSIM, "bin_width": 2 / NB, "n_U": ng,
       "n_trrust_in_U": len(TRU), "kinds": {}}
for kind in KINDS:
    Z = L.copy()
    if kind == "withincomp":
        for c in np.unique(comp):
            r_ = comp == c; Z[r_] -= Z[r_].mean(0, keepdims=True)
    Z -= Z.mean(0, keepdims=True); sd = Z.std(0, keepdims=True); sd[sd == 0] = 1; Z /= sd
    h = np.zeros(NB, np.int64)
    for s in range(0, ng, BLOCK):
        R = (Z[:, s:s + BLOCK].T @ Z) / ncell
        tri = np.arange(ng)[None, :] > np.arange(s, min(s + BLOCK, ng))[:, None]
        h += np.bincount(tb(R[tri]), minlength=NB)
    tr_r = pair_r(Z, TRU); ht = np.bincount(tb(tr_r), minlength=NB)
    q = np.where(h > 0, ht / np.maximum(h, 1), 0.0)
    c5 = np.cumsum(h[::-1]); b5 = NB - 1 - int(np.searchsorted(c5, 0.05 * h.sum()))
    K = {"trrust_rate_by_r_bin": {f"{-1 + 2 * b / NB:.2f}": [int(h[b]), int(ht[b]), float(q[b])] for b in range(NB) if h[b] > 0},
         "top5pct_bin_lower_r": round(-1 + 2 * b5 / NB, 3), "models": {}}
    rng = np.random.default_rng(7)

    def rnull(bins):
        nb = np.bincount(bins, minlength=NB); e = float((nb * q).sum())
        sims = np.zeros(NSIM)
        for b in np.nonzero(nb)[0]:
            if q[b] > 0: sims += rng.binomial(nb[b], q[b], NSIM)
        return e, sims

    for m in ORDER:
        E = [e for e in PRED[m] if e[0] in uidx and e[1] in uidx]
        if len(E) < 2: continue
        r = pair_r(Z, E); bins = tb(r)
        hit = np.array([p in TRUTH for p in E])
        d = {}
        for lab, sel in (("all_in_U", np.ones(len(E), bool)), ("not_top5pct", bins < b5), ("top5pct", bins >= b5)):
            if sel.sum() == 0: continue
            e, sims = rnull(bins[sel]); o = int(hit[sel].sum())
            d[lab] = {"n_pairs": int(sel.sum()), "hits": o, "expected_from_r": round(e, 3),
                      "null_sd": round(float(sims.std()), 3), "ratio": round(o / e, 3) if e > 0 else None,
                      "p_emp": round((int((sims >= o).sum()) + 1) / (NSIM + 1), 5)}
        K["models"][m] = d
        log(kind, m, {k: (v["hits"], v["expected_from_r"], v["ratio"], v["p_emp"]) for k, v in d.items()})
    res["kinds"][kind] = K
    del Z
json.dump(res, open(os.path.join(HERE, "coexpr_rmatched.json"), "w"), indent=1)
log("done")
