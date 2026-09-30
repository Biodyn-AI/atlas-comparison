#!/usr/bin/env python
"""Hostile review of notes/10 -- checks that need the rebuilt corpus.

Same ruler as analysis/coexpression_baseline/coexpr_trrust.py and coexpr_degmatched.py (which copy
pipeline/scripts/hypothesis_trrust2.py:29-33 truth, :60-69 config_shuffle, :100-106 read-out; pair keys
of hypothesis_trrust3.py:57-59). Their functions are re-implemented here line for line (their scripts
run top-level code, so they cannot be imported).

P1  Independent recount at 6,000 cells (Pearson): global top-8,631 and top-41,952 hits, node-matched
    Tahoe-x1 hits, by a direct threshold-and-sort (no histogram). Degree-matched graphs for the five
    passing models (their greedy b-matching), keeping the hit lists: which genes carry the hits that
    only one graph finds, and a leave-one-hub-out paired test.
P2  Cell-prefix test. Each model's SAE and catalogue were built on a prefix of the shuffled corpus
    (500,000-position cap; prefix_estimate.json: Tahoe-x1 and Geneformer-V2 264 cells, scGPT 424,
    UCE 496, C2S-Scale 984). Co-expression is recomputed on exactly those first K cells (genes U, and a
    gene must have a count > 0 in >= MINC cells of the prefix, else its column is zeroed), then:
    global top-N (N = the model's published graph size), node-matched (same genes, same N) and
    degree-matched (same genes, same degrees), each scored with the configuration null.
P3  Unrestricted gene universe: all named (non-ENSG) genes with count > 0 in >= 10 of 6,000 cells
    (not only the co-firing union, which is a model-derived gene filter). Global top-N Pearson at the
    passing models' sizes, truth = every TRRUST edge inside that universe.
usage: python review_heavy.py <corpus.h5ad> [NPERM=200] [parts=P1,P2,P3]
"""
from __future__ import annotations
import json, os, pickle, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("ATLAS_BASE", os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "..")))  # repository root with the release archives unpacked
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
PASS = ["Tahoe", "scGPT", "UCE", "C2S", "Geneformer"]
CORPUS = sys.argv[1]
NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 200
PARTS = (sys.argv[3] if len(sys.argv) > 3 else "P1,P2,P3").split(",")
MIN_CELLS, NBIN, BLOCK, CAND_MULT = 10, 20000, 512, 40
SRC = os.path.join(os.path.dirname(HERE), "coexpression_baseline")
PREF = json.load(open(os.path.join(HERE, "prefix_estimate.json")))["per_model"]
OUT = os.path.join(HERE, "review_heavy.json")
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

TRJ = json.load(open(f"{G}/trrust_edges.json"))
TRUTH = set()
for a, bs in TRJ.items():
    for b in bs:
        u, v = sorted((str(a).upper(), str(b).upper()))
        if u != v: TRUTH.add((u, v))
TF = {str(k).upper() for k in TRJ}
PRED, _ = pickle.load(open(f"{C}/.cofire_cache.pkl", "rb"))
SIZES = json.load(open(f"{C}/hypothesis_sizematched.json"))["graph_sizes"]
PUB = json.load(open(f"{C}/hypothesis_trrust2.json"))["per_model"]


class Ruler:
    """gene ids, truth keys, config null -- as coexpr_trrust.py / hypothesis_trrust2.py"""
    def __init__(self, genes):
        self.GENES = sorted(genes); self.GID = {g: i for i, g in enumerate(self.GENES)}
        self.TKEY = np.array(sorted({(lambda i, j: i * (1 << 21) + j)(*sorted((self.GID[a], self.GID[b])))
                                     for a, b in TRUTH if a in self.GID and b in self.GID}), np.int64)

    def arrays(self, pairs):
        return (np.fromiter((self.GID[e[0]] for e in pairs), np.int32, len(pairs)),
                np.fromiter((self.GID[e[1]] for e in pairs), np.int32, len(pairs)))

    def keys(self, pairs):
        a, b = self.arrays(pairs)
        return np.minimum(a, b).astype(np.int64) * (1 << 21) + np.maximum(a, b).astype(np.int64)

    def score(self, pairs, seed, nperm=NPERM):
        pairs = list(pairs); n = len(pairs)
        if n == 0: return {"n_pairs": 0}
        rng = np.random.default_rng(seed); a, b = self.arrays(pairs)
        obs = int(np.isin(self.keys(pairs), self.TKEY).sum()); nul = np.empty(nperm)
        for i in range(nperm):
            st = np.concatenate([a, b]); rng.shuffle(st)
            h, t = st[::2], st[1::2]; ok = h != t; h, t = h[ok], t[ok]
            nul[i] = np.isin(np.unique(np.minimum(h, t).astype(np.int64) * (1 << 21) + np.maximum(h, t).astype(np.int64)), self.TKEY).sum()
        nm, sd = float(nul.mean()), float(nul.std())
        return {"n_pairs": n, "hits": obs, "null_mean": round(nm, 3), "null_sd": round(sd, 3),
                "fold": round(obs / nm, 3) if nm > 0 else None,
                "p_emp": round((int((nul >= obs).sum()) + 1) / (nperm + 1), 5), "n_perm": nperm, "seed": seed}

    def hit_pairs(self, pairs):
        k = self.keys(list(pairs)); m = np.isin(k, self.TKEY)
        return {tuple(sorted(p)) for p, x in zip(pairs, m) if x}


R0 = Ruler({g for m in ORDER for e in PRED[m] for g in e})          # the repo's universe (co-firing union)

import anndata as ad
from scipy import sparse
from scipy.stats import binomtest
A = ad.read_h5ad(CORPUS)
X = sparse.csr_matrix(A.X, dtype=np.float32)
names = [str(s).upper() for s in A.var_names]; del A
name_idx = {}
for i, s in enumerate(names): name_idx.setdefault(s, i)
ncell = X.shape[0]; tot = np.asarray(X.sum(1)).ravel()
nexpr = np.asarray((X > 0).sum(0)).ravel()
UG = [g for g in R0.GENES if g in name_idx and nexpr[name_idx[g]] >= MIN_CELLS]      # = coexpr_trrust.py U
log(f"corpus {X.shape}; U {len(UG)}")


def make_Z(genes, rows=None, minc=0):
    """counts per 10k over all genes, log1p, keep `genes`; rows = first K cells; columns with count>0
    in fewer than minc of those rows are zeroed; then standardise over rows (float32)"""
    Xr = X if rows is None else X[:rows]
    tr = tot if rows is None else tot[:rows]
    Xc = Xr[:, [name_idx[g] for g in genes]].tocsc()
    Z = np.asarray(Xc.multiply((1e4 / np.maximum(tr, 1)).astype(np.float32)[:, None]).todense(), dtype=np.float32)
    ne = np.asarray((Xc > 0).sum(0)).ravel(); del Xc
    np.log1p(Z, out=Z); Z -= Z.mean(0, keepdims=True); sd = Z.std(0, keepdims=True); sd[sd == 0] = 1; Z /= sd
    if minc > 0: Z[:, ne < minc] = 0.0
    return Z, int((ne >= max(minc, 1)).sum())


def to_bin(r): return np.clip(((r + 1.0) * (NBIN / 2)).astype(np.int32), 0, NBIN - 1)


def top_N_pairs(Z, genes, N, sub=None):
    """exact top-N pairs by r among all pairs of `genes` (or of the index subset `sub`), ties by index;
    histogram threshold then collect (same approach as coexpr_trrust.py pass_hist/pass_collect)"""
    idx = np.arange(len(genes)) if sub is None else np.asarray(sub)
    Zs = Z if sub is None else Z[:, idx]; n = len(idx); nr = Z.shape[0]; h = np.zeros(NBIN, np.int64)
    for s in range(0, n, BLOCK):
        R = (Zs[:, s:s + BLOCK].T @ Zs) / nr; tri = np.arange(n)[None, :] > np.arange(s, min(s + BLOCK, n))[:, None]
        h += np.bincount(to_bin(R[tri]), minlength=NBIN)
    c = np.cumsum(h[::-1]); thr = NBIN - 1 - min(int(np.searchsorted(c, N)), NBIN - 1)
    I, J, V = [], [], []
    for s in range(0, n, BLOCK):
        R = (Zs[:, s:s + BLOCK].T @ Zs) / nr; tri = np.arange(n)[None, :] > np.arange(s, min(s + BLOCK, n))[:, None]
        ii, jj = np.nonzero(tri & (to_bin(R) >= thr)); I.append(ii + s); J.append(jj); V.append(R[ii, jj])
    I = np.concatenate(I); J = np.concatenate(J); V = np.concatenate(V)
    o = np.lexsort((J, I, -V))[:N]
    return [tuple(sorted((genes[idx[a]], genes[idx[b]]))) for a, b in zip(I[o], J[o])], float(V[o[-1]])


def degmatched(Z, genes, E):
    """greedy b-matching on r, capacities = co-firing degrees (coexpr_degmatched.py logic)"""
    gi = {g: i for i, g in enumerate(genes)}
    V = sorted({g for e in E for g in e}); vi = np.array([gi[g] for g in V]); nv = len(V)
    loc = {g: k for k, g in enumerate(V)}; deg = np.zeros(nv, np.int64)
    for a, b in E: deg[loc[a]] += 1; deg[loc[b]] += 1
    N = len(E); Zv = Z[:, vi]; nr = Z.shape[0]; h = np.zeros(NBIN, np.int64)
    for s in range(0, nv, BLOCK):
        R = (Zv[:, s:s + BLOCK].T @ Zv) / nr; tri = np.arange(nv)[None, :] > np.arange(s, min(s + BLOCK, nv))[:, None]
        h += np.bincount(to_bin(R[tri]), minlength=NBIN)
    want = min(int(h.sum()), CAND_MULT * N); cb = NBIN - 1 - int(np.searchsorted(np.cumsum(h[::-1]), want))
    rthr = cb / (NBIN / 2) - 1
    cap = deg.copy(); acc = []

    def greedy(I, J):
        for a, b in zip(I.tolist(), J.tolist()):
            if cap[a] > 0 and cap[b] > 0:
                cap[a] -= 1; cap[b] -= 1; acc.append((a, b))
                if len(acc) == N: return

    I, J, Rv = [], [], []
    for s in range(0, nv, BLOCK):
        R = (Zv[:, s:s + BLOCK].T @ Zv) / nr; tri = np.arange(nv)[None, :] > np.arange(s, min(s + BLOCK, nv))[:, None]
        ii, jj = np.nonzero(tri & (R >= rthr)); I.append(ii + s); J.append(jj); Rv.append(R[ii, jj])
    I = np.concatenate(I); J = np.concatenate(J); Rv = np.concatenate(Rv); o = np.lexsort((J, I, -Rv))
    greedy(I[o], J[o]); upper = rthr; rounds = 1; del I, J, Rv
    while len(acc) < N and rounds < 8:
        free = np.nonzero(cap > 0)[0]
        if len(free) < 2: break
        Zf = Zv[:, free]; nf = len(free); hf = np.zeros(NBIN, np.int64)
        for s in range(0, nf, BLOCK):
            R = (Zf[:, s:s + BLOCK].T @ Zf) / nr
            m_ = (np.arange(nf)[None, :] > np.arange(s, min(s + BLOCK, nf))[:, None]) & (R < upper)
            hf += np.bincount(to_bin(R[m_]), minlength=NBIN)
        if hf.sum() == 0: break
        want = min(int(hf.sum()), 8_000_000); cb = NBIN - 1 - int(np.searchsorted(np.cumsum(hf[::-1]), want))
        lower = cb / (NBIN / 2) - 1 if want < hf.sum() else -2.0
        I, J, Rv = [], [], []
        for s in range(0, nf, BLOCK):
            R = (Zf[:, s:s + BLOCK].T @ Zf) / nr
            m_ = (np.arange(nf)[None, :] > np.arange(s, min(s + BLOCK, nf))[:, None]) & (R < upper) & (R >= lower)
            ii, jj = np.nonzero(m_); I.append(free[ii + s]); J.append(free[jj]); Rv.append(R[ii, jj])
        I = np.concatenate(I); J = np.concatenate(J); Rv = np.concatenate(Rv); o = np.lexsort((J, I, -Rv))
        greedy(I[o], J[o]); rounds += 1; upper = lower; del I, J, Rv, Zf
        if lower <= -1.5: break
    return [tuple(sorted((V[a], V[b]))) for a, b in acc], N, rounds


def paired(Ec, Ex, R):
    """discordant-hit exact binomial test (pairs found by only one graph)"""
    Sc, Sx = {tuple(sorted(e)) for e in Ec}, {tuple(sorted(e)) for e in Ex}
    hc, hx = R.hit_pairs(list(Sc)), R.hit_pairs(list(Sx))
    a, b = hc - hx, hx - hc; na, nb = len(Sc - Sx), len(Sx - Sc)
    p = binomtest(len(a), len(a) + len(b), na / (na + nb)).pvalue if (a or b) else None
    return {"cofire_only_hits": len(a), "coexpr_only_hits": len(b), "shared_hits": len(hc & hx),
            "cofire_only_pairs": na, "coexpr_only_pairs": nb, "binom_p": round(p, 5) if p is not None else None}, a, b


res = json.load(open(OUT)) if os.path.exists(OUT) else {}
res.update({"n_perm": NPERM, "corpus": CORPUS, "U": len(UG)})
seed = 7000

# ============================================================== P1
if "P1" in PARTS:
    Z, _ = make_Z(UG)
    P1 = {}
    for N in (8631, 41952):
        pairs, rmin = top_N_pairs(Z, UG, N)
        P1[f"global_top{N}"] = {"hits": len(R0.hit_pairs(pairs)), "r_cutoff": round(rmin, 4)}
        log("P1 global", N, P1[f"global_top{N}"])
    uidx = {g: i for i, g in enumerate(UG)}
    COF = {m: [e for e in PRED[m] if e[0] in uidx and e[1] in uidx] for m in ORDER}
    nodes = sorted({g for e in COF["Tahoe"] for g in e})
    pairs, rmin = top_N_pairs(Z, UG, len(COF["Tahoe"]), sub=[uidx[g] for g in nodes])
    P1["nodematched_Tahoe"] = {"hits": len(R0.hit_pairs(pairs)), "N": len(pairs), "r_cutoff": round(rmin, 4)}
    log("P1 node-matched Tahoe", P1["nodematched_Tahoe"])
    P1["degmatched"] = {}
    for m in PASS:
        dm, N, rounds = degmatched(Z, UG, COF[m])
        pt, a, b = paired(COF[m], dm, R0)
        hub = {}
        for s_ in (a, b):
            for u, v in s_:
                for g in (u, v): hub[g] = hub.get(g, 0) + 1
        top_hub = max(hub, key=hub.get) if hub else None
        a2 = {e for e in a if top_hub not in e}; b2 = {e for e in b if top_hub not in e}
        p2 = binomtest(len(a2), len(a2) + len(b2), pt["cofire_only_pairs"] / (pt["cofire_only_pairs"] + pt["coexpr_only_pairs"])).pvalue if (a2 or b2) else None
        P1["degmatched"][m] = {"N_cofire": N, "N_coexpr": len(dm), "rounds": rounds,
                               "coexpr_hits": len(R0.hit_pairs(dm)), "cofire_hits": len(R0.hit_pairs(COF[m])),
                               "paired": pt, "cofire_only_hit_pairs": sorted(map(list, a)), "coexpr_only_hit_pairs": sorted(map(list, b)),
                               "genes_in_discordant_hits_top5": sorted(hub.items(), key=lambda x: -x[1])[:5],
                               "leave_top_hub_out": {"hub": top_hub, "cofire_only": len(a2), "coexpr_only": len(b2),
                                                     "binom_p": round(p2, 5) if p2 is not None else None}}
        log("P1 degmatched", m, {k: v for k, v in P1["degmatched"][m].items() if "hit_pairs" not in k})
    res["P1_recount_6000"] = P1
    del Z
    json.dump(res, open(OUT, "w"), indent=1)

# ============================================================== P2
if "P2" in PARTS:
    P2 = {"minc_rule": "gene needs count > 0 in >= 3 cells of the prefix (else column zeroed); genes = U",
          "cells_seen": {m: PREF[m]["cells_seen_tokens_U_genes"] for m in PASS}, "models": {}}
    uidx = {g: i for i, g in enumerate(UG)}
    for m in PASS:
        K = PREF[m]["cells_seen_tokens_U_genes"]
        Z, n_ok = make_Z(UG, rows=K, minc=3)
        COF = [e for e in PRED[m] if e[0] in uidx and e[1] in uidx]
        nodes = sorted({g for e in COF for g in e})
        d = {"K_cells": K, "U_genes_usable_in_prefix": n_ok, "cofire_in_U_hits": len(R0.hit_pairs(COF))}
        # global top-N at the published graph size
        pairs, rmin = top_N_pairs(Z, UG, SIZES[m]); seed += 1
        d["global_topN"] = {**R0.score(pairs, seed), "r_cutoff": round(rmin, 4), "cofire_published_hits": PUB[m]["hits"]}
        # node-matched
        pairs, rmin = top_N_pairs(Z, UG, len(COF), sub=[uidx[g] for g in nodes]); seed += 1
        d["node_matched"] = {**R0.score(pairs, seed), "r_cutoff": round(rmin, 4)}
        d["node_matched_paired"] = paired(COF, pairs, R0)[0]
        # degree-matched
        dm, N, rounds = degmatched(Z, UG, COF); seed += 1
        d["deg_matched"] = {**R0.score(dm, seed), "fill": round(len(dm) / N, 4), "rounds": rounds}
        d["deg_matched_paired"] = paired(COF, dm, R0)[0]
        # also the within-prefix min-count sensitivity for the global top-N: no extra filter, and >= 10 cells
        for mc in (1, 10):
            Z2, n2 = make_Z(UG, rows=K, minc=mc)
            pairs, rmin = top_N_pairs(Z2, UG, SIZES[m]); seed += 1
            d[f"global_topN_minc{mc}"] = {"hits": len(R0.hit_pairs(pairs)), "r_cutoff": round(rmin, 4), "genes_usable": n2}
            del Z2
        P2["models"][m] = d
        log("P2", m, K, {k: (v.get("hits"), v.get("fold"), v.get("p_emp")) if isinstance(v, dict) else v for k, v in d.items()})
        del Z
        res["P2_prefix"] = P2
        json.dump(res, open(OUT, "w"), indent=1)

# ============================================================== P2b (sensitivity of the prefix test)
if "P2b" in PARTS:
    uidx = {g: i for i, g in enumerate(UG)}
    P2b = {"what": "degree-matched co-expression on the prefix: (a) gene filter >= 10 cells in the prefix instead "
                   "of >= 3; (b) twice the estimated prefix (in case the token estimate is too low), filter >= 3",
           "runs": {}}
    runs = [(m, PREF[m]["cells_seen_tokens_U_genes"], 10) for m in PASS] + \
           [(m, 2 * PREF[m]["cells_seen_tokens_U_genes"], 3) for m in ("Tahoe", "Geneformer", "scGPT")]
    for m, K, mc in runs:
        Z, n_ok = make_Z(UG, rows=K, minc=mc)
        COF = [e for e in PRED[m] if e[0] in uidx and e[1] in uidx]
        dm, N, rounds = degmatched(Z, UG, COF); seed += 1
        d = {"K_cells": K, "minc": mc, "genes_usable": n_ok, "cofire_in_U_hits": len(R0.hit_pairs(COF)),
             "deg_matched": {**R0.score(dm, seed), "fill": round(len(dm) / N, 4)},
             "deg_matched_paired": paired(COF, dm, R0)[0]}
        P2b["runs"][f"{m}|K{K}|minc{mc}"] = d
        log("P2b", m, K, mc, d["deg_matched"]["hits"], d["deg_matched_paired"])
        del Z
        res["P2b_prefix_sensitivity"] = P2b
        json.dump(res, open(OUT, "w"), indent=1)

# ============================================================== P2c (hub check of the prefix result)
if "P2c" in PARTS:
    uidx = {g: i for i, g in enumerate(UG)}
    P2c = {"what": "degree-matched on the estimated prefix (floor >= 3 cells): discordant hits with the "
                   "most frequent gene, then the two most frequent genes, removed", "models": {}}
    for m in ("Tahoe", "scGPT"):
        K = PREF[m]["cells_seen_tokens_U_genes"]
        Z, _ = make_Z(UG, rows=K, minc=3)
        COF = [e for e in PRED[m] if e[0] in uidx and e[1] in uidx]
        dm, N, _r = degmatched(Z, UG, COF)
        pt, a, b = paired(COF, dm, R0)
        cnt = {}
        for s_ in (a, b):
            for u, v in s_:
                for g in (u, v): cnt[g] = cnt.get(g, 0) + 1
        top = [g for g, _ in sorted(cnt.items(), key=lambda x: -x[1])[:2]]
        p0 = pt["cofire_only_pairs"] / (pt["cofire_only_pairs"] + pt["coexpr_only_pairs"])
        drops = {}
        for k in (1, 2):
            a2 = {e for e in a if not set(e) & set(top[:k])}; b2 = {e for e in b if not set(e) & set(top[:k])}
            drops[f"drop_top{k}"] = {"genes": top[:k], "cofire_only": len(a2), "coexpr_only": len(b2),
                                     "binom_p": round(binomtest(len(a2), len(a2) + len(b2), p0).pvalue, 5)}
        P2c["models"][m] = {"K": K, "paired": pt, "top_genes": sorted(cnt.items(), key=lambda x: -x[1])[:6], **drops,
                            "cofire_only_hit_pairs": sorted(map(list, a)), "coexpr_only_hit_pairs": sorted(map(list, b))}
        log("P2c", m, pt, drops)
        del Z
    res["P2c_prefix_hub_check"] = P2c
    json.dump(res, open(OUT, "w"), indent=1)

# ============================================================== P3
if "P3" in PARTS:
    named = sorted({s for s in name_idx if not s.startswith("ENSG") and nexpr[name_idx[s]] >= MIN_CELLS})
    R3 = Ruler(named)
    Z, _ = make_Z(named)
    inU = set(UG)
    P3 = {"n_genes": len(named), "n_genes_outside_U": len(set(named) - inU), "trrust_edges_in_universe": int(len(R3.TKEY)), "global": {}}
    for m in PASS:
        N = SIZES[m]
        pairs, rmin = top_N_pairs(Z, named, N); seed += 1
        sc = R3.score(pairs, seed)
        sc["r_cutoff"] = round(rmin, 4)
        sc["pairs_with_gene_outside_U"] = int(sum((a not in inU) or (b not in inU) for a, b in pairs))
        sc["U_restricted_hits_notes"] = json.load(open(f"{SRC}/coexpr_trrust.json"))["kinds"]["pearson"]["global"][str(N)]["hits"]
        P3["global"][str(N)] = sc
        log("P3", N, sc)
    del Z
    res["P3_unrestricted_universe"] = P3

res["wall_s"] = round(time.time() - t0, 1)
json.dump(res, open(OUT, "w"), indent=1)
log("done")
