#!/usr/bin/env python
"""Degree-matched co-expression graphs (the strictest like-for-like baseline).

For each model m, take its co-firing graph restricted to genes expressed in >= 10 cells (same set U
as coexpr_trrust.py). Build a co-expression graph on the SAME genes with (nearly) the SAME degree
sequence and the SAME number of pairs: go down the list of that gene set's pairs in order of
decreasing Pearson r and accept a pair if both genes still have unused degree (greedy b-matching,
capacities = co-firing degrees). So each gene gets as many partners as it has in the co-firing graph,
but the partners are chosen by co-expression instead of co-firing. Scored against TRRUST with the
repo's configuration-model null (config_shuffle logic of hypothesis_trrust2.py:60-69), 200 rewirings.

usage: python coexpr_degmatched.py <corpus.h5ad> [NPERM=200] [models=Tahoe,scGPT,UCE,C2S,Geneformer,MaxToki]
"""
from __future__ import annotations
import json, os, pickle, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("ATLAS_BASE", os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "..")))  # repository root with the release archives unpacked
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
CORPUS = sys.argv[1]
NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 200
MODELS = (sys.argv[3] if len(sys.argv) > 3 else "Tahoe,scGPT,UCE,C2S,Geneformer,MaxToki").split(",")
MIN_CELLS, NBIN, BLOCK, CAND_MULT = 10, 20000, 512, 40
t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

TRUTH = set()
for a, bs in json.load(open(f"{G}/trrust_edges.json")).items():
    for b in bs:
        u, v = sorted((str(a).upper(), str(b).upper()))
        if u != v: TRUTH.add((u, v))
PRED, NFEAT = pickle.load(open(f"{C}/.cofire_cache.pkl", "rb"))
GENES = sorted({g for m in ORDER for e in PRED[m] for g in e}); GID = {g: i for i, g in enumerate(GENES)}
TKEY = np.array(sorted({(lambda i, j: i * (1 << 21) + j)(*sorted((GID[a], GID[b])))
                        for a, b in TRUTH if a in GID and b in GID}), np.int64)


def config_shuffle(edges_gid, rng):          # verbatim logic of hypothesis_trrust2.py:60-69
    a, b = edges_gid
    stubs = np.concatenate([a, b]); rng.shuffle(stubs)
    h = stubs[::2]; t = stubs[1::2]
    ok = h != t
    h, t = h[ok], t[ok]
    lo = np.minimum(h, t).astype(np.int64); hi = np.maximum(h, t).astype(np.int64)
    return np.unique(lo * (1 << 21) + hi)


def gid_arrays(pairs):
    return (np.fromiter((GID[e[0]] for e in pairs), np.int32, len(pairs)),
            np.fromiter((GID[e[1]] for e in pairs), np.int32, len(pairs)))


def keys_of(pairs):
    a, b = gid_arrays(pairs)
    return np.minimum(a, b).astype(np.int64) * (1 << 21) + np.maximum(a, b).astype(np.int64)


def score(pairs, seed):
    rng = np.random.default_rng(seed); n = len(pairs)
    obs = int(np.isin(keys_of(pairs), TKEY).sum()); ga = gid_arrays(pairs)
    nul = np.array([np.isin(config_shuffle(ga, rng), TKEY).sum() for _ in range(NPERM)], float)
    nm, sd = float(nul.mean()), float(nul.std())
    return {"n_pairs": n, "hits": obs, "precision": round(obs / n, 6), "null_mean": round(nm, 3),
            "null_sd": round(sd, 3), "fold": round(obs / nm, 3) if nm > 0 else None,
            "z": round((obs - nm) / sd, 3) if sd > 0 else None,
            "p_emp": round((int((nul >= obs).sum()) + 1) / (NPERM + 1), 5), "n_perm": NPERM, "seed": seed}


import anndata as ad
from scipy import sparse
from scipy.stats import fisher_exact
A = ad.read_h5ad(CORPUS)
X = sparse.csr_matrix(A.X, dtype=np.float32)
names = [str(s).upper() for s in A.var_names]
name_idx = {}
for i, s in enumerate(names): name_idx.setdefault(s, i)
ncell = X.shape[0]
nexpr = np.asarray((X > 0).sum(0)).ravel(); tot = np.asarray(X.sum(1)).ravel()
UG = [g for g in GENES if g in name_idx and nexpr[name_idx[g]] >= MIN_CELLS]
uidx = {g: i for i, g in enumerate(UG)}
Xc = X[:, [name_idx[g] for g in UG]].tocsc()
Z = np.asarray(Xc.multiply((1e4 / np.maximum(tot, 1)).astype(np.float32)[:, None]).todense(), dtype=np.float32)
del Xc, X, A
np.log1p(Z, out=Z); Z -= Z.mean(0, keepdims=True); sd = Z.std(0, keepdims=True); sd[sd == 0] = 1; Z /= sd
log(f"U = {len(UG)} genes, Z {Z.shape}")

res = {"method": "greedy b-matching on Pearson r (counts/10k, log1p), capacities = co-firing degrees within U",
       "n_perm": NPERM, "cand_mult": CAND_MULT, "models": {}}
seed = 5000
for m in MODELS:
    E = [e for e in PRED[m] if e[0] in uidx and e[1] in uidx]
    V = sorted({g for e in E for g in e}); vi = np.array([uidx[g] for g in V]); nv = len(V)
    deg = np.zeros(nv, np.int64); loc = {g: k for k, g in enumerate(V)}
    for a, b in E: deg[loc[a]] += 1; deg[loc[b]] += 1
    N = len(E); Zv = Z[:, vi]
    # histogram of r over V pairs -> threshold giving ~CAND_MULT*N candidates
    h = np.zeros(NBIN, np.int64)
    for s in range(0, nv, BLOCK):
        R = (Zv[:, s:s + BLOCK].T @ Zv) / ncell
        tri = np.arange(nv)[None, :] > np.arange(s, min(s + BLOCK, nv))[:, None]
        h += np.bincount(np.clip(((R[tri] + 1) * NBIN / 2).astype(np.int32), 0, NBIN - 1), minlength=NBIN)
    want = min(int(h.sum()), CAND_MULT * N)
    cb = NBIN - 1 - int(np.searchsorted(np.cumsum(h[::-1]), want))
    rthr = cb / (NBIN / 2) - 1
    I, J, Rv = [], [], []
    for s in range(0, nv, BLOCK):
        R = (Zv[:, s:s + BLOCK].T @ Zv) / ncell
        tri = np.arange(nv)[None, :] > np.arange(s, min(s + BLOCK, nv))[:, None]
        ii, jj = np.nonzero(tri & (R >= rthr)); I.append(ii + s); J.append(jj); Rv.append(R[ii, jj])
    I = np.concatenate(I); J = np.concatenate(J); Rv = np.concatenate(Rv)
    o = np.lexsort((J, I, -Rv)); I, J, Rv = I[o], J[o], Rv[o]
    cap = deg.copy(); acc = []; acc_r = []

    def greedy(I, J, Rv):
        for a, b, r in zip(I.tolist(), J.tolist(), Rv.tolist()):
            if cap[a] > 0 and cap[b] > 0:
                cap[a] -= 1; cap[b] -= 1; acc.append((a, b)); acc_r.append(r)
                if len(acc) == N: return

    greedy(I, J, Rv)
    ncand_total = len(I); rounds = 1; upper = rthr
    del I, J, Rv
    # Continue the SAME greedy below the first threshold, only among genes that still have free
    # degree. Exact: a pair with a saturated gene can never be accepted later in the greedy order.
    while len(acc) < N and rounds < 12:
        free = np.nonzero(cap > 0)[0]
        if len(free) < 2: break
        Zf = Zv[:, free]; nf = len(free)
        hf = np.zeros(NBIN, np.int64)
        for s in range(0, nf, BLOCK):
            R = (Zf[:, s:s + BLOCK].T @ Zf) / ncell
            m_ = (np.arange(nf)[None, :] > np.arange(s, min(s + BLOCK, nf))[:, None]) & (R < upper)
            hf += np.bincount(np.clip(((R[m_] + 1) * NBIN / 2).astype(np.int32), 0, NBIN - 1), minlength=NBIN)
        if hf.sum() == 0: break
        want = min(int(hf.sum()), 8_000_000)
        cb = NBIN - 1 - int(np.searchsorted(np.cumsum(hf[::-1]), want))
        lower = cb / (NBIN / 2) - 1 if want < hf.sum() else -2.0
        I, J, Rv = [], [], []
        for s in range(0, nf, BLOCK):
            R = (Zf[:, s:s + BLOCK].T @ Zf) / ncell
            m_ = (np.arange(nf)[None, :] > np.arange(s, min(s + BLOCK, nf))[:, None]) & (R < upper) & (R >= lower)
            ii, jj = np.nonzero(m_); I.append(free[ii + s]); J.append(free[jj]); Rv.append(R[ii, jj])
        I = np.concatenate(I); J = np.concatenate(J); Rv = np.concatenate(Rv)
        o = np.lexsort((J, I, -Rv)); greedy(I[o], J[o], Rv[o])
        ncand_total += len(I); rounds += 1; upper = lower
        del I, J, Rv, Zf
        if lower <= -1.5: break
    pairs = [tuple(sorted((V[a], V[b]))) for a, b in acc]
    seed += 1; sx = score(pairs, seed)
    seed += 1; sc = score(E, seed)
    hc, hx, n2 = sc["hits"], sx["hits"], len(pairs)
    kx = set(keys_of(pairs)[np.isin(keys_of(pairs), TKEY)].tolist()); kc = set(keys_of(E)[np.isin(keys_of(E), TKEY)].tolist())
    d = {"n_genes": nv, "N_cofire_in_U": N, "candidates_first_round": int(ncand_total), "candidate_r_threshold_first_round": round(rthr, 4),
         "greedy_rounds": rounds,
         "N_coexpr_accepted": n2, "fill_share": round(n2 / N, 4),
         "degree_L1_unfilled_share": round(float(cap.sum()) / float(deg.sum()), 4),
         "min_r_accepted": round(float(min(acc_r)), 4) if acc_r else None,
         "median_r_accepted": round(float(np.median(acc_r)), 4) if acc_r else None,
         "cofire_in_U": sc, "coexpr_degmatched": sx,
         "shared_pairs": len(set(pairs) & set(tuple(sorted(e)) for e in E)),
         "hits_shared": len(kx & kc),
         "fisher_cofire_vs_coexpr": {"odds_ratio": round(float(fisher_exact([[hc, N - hc], [hx, n2 - hx]])[0]), 3) if hx > 0 else None,
                                     "p_two_sided": round(float(fisher_exact([[hc, N - hc], [hx, n2 - hx]])[1]), 6)}}
    res["models"][m] = d
    log(f"{m}: N {N}, genes {nv}, coexpr accepted {n2} ({d['fill_share']}); cofire {hc} hits fold {sc['fold']} p {sc['p_emp']} | "
        f"coexpr-degmatched {hx} hits fold {sx['fold']} p {sx['p_emp']} | shared hits {d['hits_shared']} | fisher p {d['fisher_cofire_vs_coexpr']['p_two_sided']}")
    del Zv
    json.dump(res, open(os.path.join(HERE, "coexpr_degmatched.json"), "w"), indent=1)
res["wall_s"] = round(time.time() - t0, 1)
json.dump(res, open(os.path.join(HERE, "coexpr_degmatched.json"), "w"), indent=1)
log("done")
