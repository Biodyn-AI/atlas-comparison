#!/usr/bin/env python
"""Co-expression baseline for the held-out TRRUST test (gap G1 in notes/07).

Question: do the top co-expressed gene pairs, computed from the same kind of cells, recover
held-out TRRUST edges as well as the SAE co-firing graphs do, under the same configuration-model null?

Same ruler as the repo:
  * co-firing graphs = the repo's own W>=5 graphs, read from the stage-3 cache
    outputs/atlas/comparative/.cofire_cache.pkl written by pipeline/scripts/hypothesis_trrust3.py
    (identical to hypothesis_trrust2.py:cofire_pairs; sizes match hypothesis_sizematched.json:graph_sizes).
  * truth = TRRUST undirected, self-pairs dropped (hypothesis_trrust2.py:29-33).
  * gene ids / pair keys = union of the ten co-firing node sets, key i*2^21+j (hypothesis_trrust3.py:57-59).
  * null = config_shuffle, copied verbatim from hypothesis_trrust2.py:60-69 (stub matching,
    self-loops and duplicate pairs dropped), here with a fixed per-graph seed.
  * p = (#null >= obs + 1) / (NPERM + 1), fold = obs / null mean (hypothesis_trrust2.py:100-106).

Co-expression: counts per 10k + log1p over all genes; restricted to the union gene set U expressed
(count>0) in >= MIN_CELLS cells; gene-gene correlation over the 6,000 cells, in row blocks, float32.
Kinds: pearson (main), spearman (rank per gene, average ties), withincomp (log-normalised values
centred within each compartment, i.e. pooled within-compartment Pearson).

usage: python coexpr_trrust.py <corpus.h5ad> [NPERM=200] [kinds=pearson,spearman,withincomp]
writes coexpr_trrust.json next to this script.
"""
from __future__ import annotations
import json, os, pickle, sys, time, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# repository root with the release archives unpacked (pipeline/scripts/fetch_runs.sh)
BASE = os.environ.get("ATLAS_BASE", os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "..")))
ORIG_C = os.path.join(BASE, "outputs/atlas/comparative")
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
CACHE = f"{C}/.cofire_cache.pkl"
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
PASSED = ["Tahoe", "scGPT", "UCE", "C2S", "Geneformer"]      # hypothesis_trrust3.py:18
CORPUS = sys.argv[1]
NPERM = int(sys.argv[2]) if len(sys.argv) > 2 else 200
KINDS = (sys.argv[3] if len(sys.argv) > 3 else "pearson,spearman,withincomp").split(",")
MIN_CELLS = 10
BUDGET = 8631
NBIN = 20000                                  # histogram bins of width 1e-4 on [-1, 1]
BLOCK = 512
OUT = os.path.join(HERE, "coexpr_trrust.json")
t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- truth, graphs, keys (repo logic)
TRUTH = set()
for a, bs in json.load(open(f"{G}/trrust_edges.json")).items():
    for b in bs:
        u, v = sorted((str(a).upper(), str(b).upper()))
        if u != v: TRUTH.add((u, v))
PRED, NFEAT = pickle.load(open(CACHE, "rb"))
GENES = sorted({g for m in ORDER for e in PRED[m] for g in e}); GID = {g: i for i, g in enumerate(GENES)}
TKEY = np.array(sorted({(lambda i, j: i * (1 << 21) + j)(*sorted((GID[a], GID[b])))
                        for a, b in TRUTH if a in GID and b in GID}), np.int64)
PUB2 = json.load(open(f"{ORIG_C}/hypothesis_trrust2.json"))["per_model"]
PUBSM = json.load(open(f"{ORIG_C}/hypothesis_sizematched.json"))
for m in ORDER:
    assert len(PRED[m]) == PUBSM["graph_sizes"][m], m
log(f"co-firing union: {len(GENES)} genes; TRRUST edges inside union: {len(TKEY)}")


def config_shuffle(edges_gid, rng):
    """verbatim logic of hypothesis_trrust2.py:60-69 (input already as gene-id arrays)"""
    a, b = edges_gid
    stubs = np.concatenate([a, b]); rng.shuffle(stubs)
    h = stubs[::2]; t = stubs[1::2]
    ok = h != t
    h, t = h[ok], t[ok]
    lo = np.minimum(h, t).astype(np.int64); hi = np.maximum(h, t).astype(np.int64)
    return np.unique(lo * (1 << 21) + hi)


def gid_arrays(pairs):
    a = np.fromiter((GID[e[0]] for e in pairs), np.int32, len(pairs))
    b = np.fromiter((GID[e[1]] for e in pairs), np.int32, len(pairs))
    return a, b


def keys_of(pairs):
    a, b = gid_arrays(pairs)
    lo = np.minimum(a, b).astype(np.int64); hi = np.maximum(a, b).astype(np.int64)
    return lo * (1 << 21) + hi


_seed = [1000]


def score(pairs, nperm=NPERM, tkey=None):
    """hits vs configuration-model null; same read-out as hypothesis_trrust2.py:100-106.
    tkey: truth keys to count against (default all TRRUST edges in the union; a subset gives a
    'masked' null in which observed and rewired graphs are scored on the same restricted truth)"""
    TK = TKEY if tkey is None else tkey
    pairs = list(pairs)
    n = len(pairs)
    if n == 0:
        return {"n_pairs": 0}
    _seed[0] += 1
    rng = np.random.default_rng(_seed[0])
    k = keys_of(pairs)
    obs = int(np.isin(k, TK).sum())
    ga = gid_arrays(pairs)
    nul = np.empty(nperm); kept = np.empty(nperm)
    for i in range(nperm):
        s = config_shuffle(ga, rng)
        nul[i] = np.isin(s, TK).sum(); kept[i] = len(s)
    nm, sd = float(nul.mean()), float(nul.std())
    genes = {g for e in pairs for g in e}
    return {"n_pairs": n, "n_genes": len(genes), "hits": obs, "precision": round(obs / n, 6),
            "null_mean": round(nm, 3), "null_sd": round(sd, 3),
            "fold": round(obs / nm, 3) if nm > 0 else None,
            "z": round((obs - nm) / sd, 3) if sd > 0 else None,
            "p_emp": round((int((nul >= obs).sum()) + 1) / (nperm + 1), 5),
            "n_perm": nperm, "seed": _seed[0],
            "null_kept_pair_share": round(float(kept.mean()) / n, 4)}


# ---------------------------------------------------------------- corpus
import anndata as ad
from scipy import sparse
from scipy.stats import rankdata, fisher_exact

A = ad.read_h5ad(CORPUS)
X = sparse.csr_matrix(A.X, dtype=np.float32)
names = np.array([str(s).upper() for s in A.var_names])
comp = A.obs["compartment"].astype(str).to_numpy()
ncell = X.shape[0]
log(f"corpus {X.shape}, compartments {dict(zip(*np.unique(comp, return_counts=True)))}")
tot = np.asarray(X.sum(1)).ravel()
ncells_expr = np.asarray((X > 0).sum(0)).ravel()
name_idx = {}
for i, s in enumerate(names):
    name_idx.setdefault(s, i)
U_all = set(GENES)
in_corpus = [g for g in GENES if g in name_idx]
expr = [g for g in in_corpus if ncells_expr[name_idx[g]] >= MIN_CELLS]
UG = expr                                     # sorted (GENES is sorted)
col = np.array([name_idx[g] for g in UG])
ng = len(UG)
uidx = {g: i for i, g in enumerate(UG)}
log(f"union genes {len(GENES)}; in corpus {len(in_corpus)}; expressed in >={MIN_CELLS} cells: {ng}")

# normalise: counts per 10k over ALL genes, log1p; then keep U genes (dense float32)
Xc = X[:, col].tocsc()
scale = (1e4 / np.maximum(tot, 1)).astype(np.float32)
Xn = np.asarray(Xc.multiply(scale[:, None]).todense(), dtype=np.float32)
np.log1p(Xn, out=Xn)
del Xc
log(f"dense log-normalised matrix {Xn.shape}, {Xn.nbytes/1e9:.2f} GB")

TR_U = {(a, b) for a, b in TRUTH if a in uidx and b in uidx}
n_pairs_U = ng * (ng - 1) // 2


def standardise(M):
    M -= M.mean(0, keepdims=True)
    sd = M.std(0, keepdims=True); sd[sd == 0] = 1
    M /= sd
    return M


def make_Z(kind):
    Z = Xn.copy()
    if kind == "pearson":
        pass
    elif kind == "spearman":
        for s in range(0, ng, 1000):
            Z[:, s:s + 1000] = rankdata(Z[:, s:s + 1000], axis=0, method="average").astype(np.float32)
    elif kind == "withincomp":
        for c in np.unique(comp):
            r = comp == c
            Z[r] -= Z[r].mean(0, keepdims=True)
    return standardise(Z)


# ---------------------------------------------------------------- model node sets (restricted to U)
COF_U, NODES = {}, {}
member = np.zeros((len(ORDER), ng), bool)
for mi, m in enumerate(ORDER):
    COF_U[m] = {e: w for e, w in PRED[m].items() if e[0] in uidx and e[1] in uidx}
    NODES[m] = sorted({g for e in COF_U[m] for g in e})
    member[mi, [uidx[g] for g in NODES[m]]] = True


def to_bin(r):
    return np.clip(((r + 1.0) * (NBIN / 2)).astype(np.int32), 0, NBIN - 1)


def pass_hist(Z):
    """histograms of r over all U pairs (i<j) and over each model's node-set pairs"""
    hg = np.zeros(NBIN, np.int64); hm = np.zeros((len(ORDER), NBIN), np.int64)
    for s in range(0, ng, BLOCK):
        e = min(s + BLOCK, ng)
        R = (Z[:, s:e].T @ Z) / ncell
        tri = np.arange(ng)[None, :] > np.arange(s, e)[:, None]
        B = to_bin(R)
        hg += np.bincount(B[tri], minlength=NBIN)
        for mi in range(len(ORDER)):
            rm = member[mi, s:e]
            if not rm.any(): continue
            mk = tri[rm][:, member[mi]]
            hm[mi] += np.bincount(B[rm][:, member[mi]][mk], minlength=NBIN)
    return hg, hm


def thr_for_top(h, N):
    """lowest bin index b such that #pairs with bin >= b is >= N"""
    c = np.cumsum(h[::-1])
    k = int(np.searchsorted(c, N))
    return NBIN - 1 - min(k, NBIN - 1)


def pass_collect(Z, gthr, mthr):
    """collect (i, j, r) with r-bin >= threshold, globally and per model node set"""
    gl = []; ml = {mi: [] for mi in mthr}
    for s in range(0, ng, BLOCK):
        e = min(s + BLOCK, ng)
        R = (Z[:, s:e].T @ Z) / ncell
        tri = np.arange(ng)[None, :] > np.arange(s, e)[:, None]
        B = to_bin(R)
        ii, jj = np.nonzero(tri & (B >= gthr))
        gl.append((ii + s, jj, R[ii, jj]))
        for mi, t in mthr.items():
            ii, jj = np.nonzero(tri & (B >= t) & member[mi, s:e][:, None] & member[mi][None, :])
            ml[mi].append((ii + s, jj, R[ii, jj]))
    cat = lambda L: tuple(np.concatenate([x[k] for x in L]) for k in range(3))
    return cat(gl), {mi: cat(v) for mi, v in ml.items()}


def top_pairs(trip, N):
    i, j, r = trip
    o = np.lexsort((j, i, -r))[:N]              # r descending, ties by gene index
    return [(UG[a], UG[b]) for a, b in zip(i[o], j[o])], float(r[o[-1]]) if len(o) else None


def pair_r(Z, pairs):
    a = np.array([uidx[p[0]] for p in pairs]); b = np.array([uidx[p[1]] for p in pairs])
    out = np.empty(len(pairs), np.float32)
    for s in range(0, len(pairs), 4000):
        out[s:s + 4000] = (Z[:, a[s:s + 4000]] * Z[:, b[s:s + 4000]]).sum(0) / ncell
    return out


def quantile_thr(h, frac):
    """bin threshold for the top `frac` of pairs; returns (lower r edge, exact achieved fraction)"""
    tot_ = h.sum(); b = thr_for_top(h, int(np.ceil(frac * tot_)))
    return b, float(h[b:].sum()) / tot_


def edge_r(b):
    return b / (NBIN / 2) - 1.0


# ---------------------------------------------------------------- main
res = {"question": "do top co-expressed pairs recover held-out TRRUST as well as SAE co-firing?",
       "corpus": CORPUS, "n_cells": int(ncell), "min_cells": MIN_CELLS, "n_perm": NPERM,
       "null": "configuration model (hypothesis_trrust2.py:config_shuffle, verbatim), fixed seed per graph",
       "truth": "TRRUST undirected, self-pairs dropped",
       "gene_universe": {"cofire_union": len(GENES), "in_corpus": len(in_corpus), "expressed_ge_min_cells": ng,
                         "trrust_edges_in_cofire_union": int(len(TKEY)), "trrust_edges_in_U": len(TR_U),
                         "n_pairs_U": int(n_pairs_U),
                         "random_pair_precision_U": round(len(TR_U) / n_pairs_U, 8)},
       "normalisation": "counts per 10k over all genes, log1p; U restricted after normalising",
       "published": {m: {"full": PUB2[m], "sizematched": PUBSM["models"][m]} for m in ORDER},
       "graph_sizes_published": PUBSM["graph_sizes"],
       "cofire": {}, "kinds": {}}

# co-firing graphs re-scored on the same code (full, restricted to U, top-8631 by weight)
for m in ORDER:
    g = PRED[m]
    top8631 = sorted(g, key=lambda e: -g[e])[:BUDGET] if len(g) >= BUDGET else None
    res["cofire"][m] = {"full": score(g), "in_U": score(COF_U[m]),
                        "n_nodes_in_U": len(NODES[m]),
                        "top8631_by_weight": score(top8631) if top8631 else None}
    log(f"cofire {m}: full {res['cofire'][m]['full'].get('hits')} hits fold {res['cofire'][m]['full'].get('fold')}; "
        f"in U {res['cofire'][m]['in_U'].get('n_pairs')} pairs {res['cofire'][m]['in_U'].get('hits')} hits "
        f"fold {res['cofire'][m]['in_U'].get('fold')}")

for kind in KINDS:
    log(f"=== {kind}")
    Z = make_Z(kind)
    hg, hm = pass_hist(Z)
    log("histograms done")
    Ns_global = sorted({PUBSM["graph_sizes"][m] for m in ORDER} | {BUDGET})
    need_m = {mi: max(len(COF_U[m]), BUDGET, PUBSM["graph_sizes"][m]) for mi, m in enumerate(ORDER)
              if len(COF_U[m]) > 0}
    gthr = thr_for_top(hg, max(Ns_global))
    mthr = {mi: thr_for_top(hm[mi], N) for mi, N in need_m.items()}
    gtrip, mtrip = pass_collect(Z, gthr, mthr)
    log(f"collected {len(gtrip[0])} global candidates")
    t1b, f1 = quantile_thr(hg, 0.01); t5b, f5 = quantile_thr(hg, 0.05)
    K = {"thresholds": {"top1pct_r": round(edge_r(t1b), 4), "top1pct_achieved": round(f1, 5),
                        "top5pct_r": round(edge_r(t5b), 4), "top5pct_achieved": round(f5, 5)},
         "global": {}, "node_matched": {}, "overlap": {}}
    # (i) global top-N at each model's published graph size, (ii) equal budget
    for N in Ns_global:
        pairs, rmin = top_pairs(gtrip, N)
        sc = score(pairs); sc["r_cutoff"] = round(rmin, 4) if rmin is not None else None
        K["global"][str(N)] = sc
        log(f"  global top-{N}: {sc['hits']} hits, null {sc['null_mean']}, fold {sc['fold']}, p {sc['p_emp']}")
    # (iii) node-matched: same genes, same N as the model's co-firing graph within U
    for mi, m in enumerate(ORDER):
        if mi not in mtrip: continue
        d = {}
        for lab, N in (("N_cofire_in_U", len(COF_U[m])), ("N_published", PUBSM["graph_sizes"][m]),
                       ("N_8631", BUDGET)):
            npairs_node = len(NODES[m]) * (len(NODES[m]) - 1) // 2
            if N > npairs_node or N == 0:
                d[lab] = {"n_pairs": N, "testable": False}; continue
            pairs, rmin = top_pairs(mtrip[mi], N)
            sc = score(pairs); sc["r_cutoff"] = round(rmin, 4)
            if lab == "N_cofire_in_U":
                shared = len(set(pairs) & set(COF_U[m]))
                sc["shared_pairs_with_cofire"] = shared
                sc["shared_share"] = round(shared / N, 4)
                hc = res["cofire"][m]["in_U"]["hits"]; hx = sc["hits"]
                sc["fisher_cofire_vs_coexpr"] = {"cofire_hits": hc, "coexpr_hits": hx, "N": N,
                    "odds_ratio": (round(float(fisher_exact([[hc, N - hc], [hx, N - hx]])[0]), 3) if hx > 0 else None),
                    "p_two_sided": round(float(fisher_exact([[hc, N - hc], [hx, N - hx]])[1]), 6)}
                hk_x = set(keys_of(pairs)[np.isin(keys_of(pairs), TKEY)].tolist())
                hk_c = set(keys_of(list(COF_U[m]))[np.isin(keys_of(list(COF_U[m])), TKEY)].tolist())
                sc["hits_shared_with_cofire"] = len(hk_x & hk_c)
                inv = {v: k for k, v in GID.items()}
                sc["hit_pairs"] = sorted([inv[int(k) >> 21], inv[int(k) & ((1 << 21) - 1)]] for k in hk_x)
            d[lab] = sc
        K["node_matched"][m] = d
        s0 = d["N_cofire_in_U"]
        log(f"  node-matched {m}: N {s0.get('n_pairs')} coexpr {s0.get('hits')} hits fold {s0.get('fold')} p {s0.get('p_emp')} "
            f"| cofire {res['cofire'][m]['in_U'].get('hits')} fold {res['cofire'][m]['in_U'].get('fold')}")
    # TRRUST edges (inside U) that are NOT highly co-expressed: truth for the masked null
    inv = {v: k for k, v in GID.items()}
    tr_pairs = [(inv[int(k) >> 21], inv[int(k) & ((1 << 21) - 1)]) for k in TKEY]
    tr_in_U = [(k, p) for k, p in zip(TKEY, tr_pairs) if p[0] in uidx and p[1] in uidx]
    trr = pair_r(Z, [p for _, p in tr_in_U]); trb = to_bin(trr)
    tk_arr = np.array([k for k, _ in tr_in_U], np.int64)
    tr_low5 = tk_arr[trb < t5b]; tr_low1 = tk_arr[trb < t1b]
    K["trrust_edges_in_U"] = int(len(tk_arr))
    K["trrust_share_in_global_top5pct"] = round(float((trb >= t5b).mean()), 4)
    K["trrust_share_in_global_top1pct"] = round(float((trb >= t1b).mean()), 4)
    K["trrust_median_r"] = round(float(np.median(trr)), 4)
    # pair-level overlap and co-firing minus co-expression
    for mi, m in enumerate(ORDER):
        E = list(COF_U[m])
        if not E: continue
        r = pair_r(Z, E)
        rb = to_bin(r)
        n_node = int(hm[mi].sum())
        m1b, mf1 = quantile_thr(hm[mi], 0.01); m5b, mf5 = quantile_thr(hm[mi], 0.05)
        ist = np.isin(keys_of(E), TKEY)
        low5 = TKEY[np.isin(TKEY, tr_low5)]; low1 = TKEY[np.isin(TKEY, tr_low1)]
        o = {"n_pairs": len(E), "median_r": round(float(np.median(r)), 4),
             "share_in_global_top1pct": round(float((rb >= t1b).mean()), 4),
             "share_in_global_top5pct": round(float((rb >= t5b).mean()), 4),
             "node_set_pairs": n_node,
             "node_top1pct_r": round(edge_r(m1b), 4), "node_top5pct_r": round(edge_r(m5b), 4),
             "share_in_node_top1pct": round(float((rb >= m1b).mean()), 4),
             "share_in_node_top5pct": round(float((rb >= m5b).mean()), 4)}
        for lab, sel, tk in (("cofire_not_global_top5", rb < t5b, None), ("cofire_in_global_top5", rb >= t5b, None),
                             ("cofire_not_global_top1", rb < t1b, None), ("cofire_in_global_top1", rb >= t1b, None),
                             ("cofire_not_node_top5", rb < m5b, None),
                             ("cofire_not_global_top5_maskednull", rb < t5b, low5),
                             ("cofire_not_global_top1_maskednull", rb < t1b, low1)):
            sub = [e for e, s in zip(E, sel) if s]
            o[lab] = score(sub, tkey=tk) if sub else {"n_pairs": 0}
        # TRRUST hits of co-firing graph: their co-expression r
        o["hit_pairs_r"] = sorted([[e[0], e[1], round(float(x), 4)] for e, x, t in zip(E, r, ist) if t],
                                  key=lambda z: -z[2])
        K["overlap"][m] = o
        log(f"  overlap {m}: {o['share_in_global_top5pct']:.3f} in top5%, {o['share_in_global_top1pct']:.3f} in top1%; "
            f"not-top5 hits {o['cofire_not_global_top5'].get('hits')} fold {o['cofire_not_global_top5'].get('fold')} "
            f"p {o['cofire_not_global_top5'].get('p_emp')}")
    # median r of all U pairs (from histogram)
    c = np.cumsum(hg); med = edge_r(int(np.searchsorted(c, c[-1] / 2)))
    K["median_r_all_U_pairs"] = round(med, 4)
    K["hist_global_coarse"] = {f"{edge_r(b):.2f}": int(hg[b:b + 500].sum()) for b in range(0, NBIN, 500)}
    res["kinds"][kind] = K
    del Z
    json.dump(res, open(OUT, "w"), indent=1)
    log(f"wrote {OUT} ({kind})")

res["wall_s"] = round(time.time() - t0, 1)
json.dump(res, open(OUT, "w"), indent=1)
log("done")
