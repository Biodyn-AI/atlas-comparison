#!/usr/bin/env python
"""Part (e) of notes/11_input_encoding.md: how much of each model's held-out TRRUST result comes from
pairs that are lexical / family pairs, or from tie-tail slots of the catalogues.

Graph and scoring are exactly hypothesis_trrust2.py (A) / hypothesis_trrust3.py:
  graph  = gene pairs co-occurring in >= W0=5 feature top-10 lists (upper-case, ENSG dropped, >= 2 genes),
           all layers of the model;  the published graphs are read from the release copy's
           outputs/atlas/comparative/.cofire_cache.pkl (written by hypothesis_trrust3.py).
  truth  = TRRUST edges (genesets/trrust_edges.json), undirected, self-pairs dropped.
  null   = configuration model on the graph's own edge list (stubs shuffled, self-loops and duplicate
           pairs dropped), NPERM rewirings; fold = hits / null mean; p = (#null >= obs + 1)/(NPERM + 1).
Variants (a pair "removed from both" is dropped from the graph AND from the truth set, so the null
cannot score it either):
  V0 published       published graph, full truth
  V1 no_p3           remove pairs whose symbols share the first 3 characters
  V2 no_hgnc         remove pairs sharing an HGNC gene group (hgnc_complete_set.txt, gene_group_id)
  V3 no_alpha20      remove pairs within 20 positions in the global alphabetical gene order
                     (sorted union of all ten models' filtered genes)
  V4 no_lexfam       remove the union of V1-V3
  V5 clean           graph rebuilt from top-10 lists with tie-tail slots blanked (ie_common), full truth
  V6 clean_no_lexfam V5 graph with the V4 removal
  and the equal-budget (top-8,631 pairs by weight, hypothesis_sizematched.py) versions of V0 and V5.
Fresh default_rng(0) per model x variant.
    ATLAS_BASE=<copy> python trrust_lexical.py [NPERM]   -> trrust_lexical.json
"""
from __future__ import annotations

import json, sys, time, pickle, itertools
from collections import Counter, defaultdict
import numpy as np
from ie_common import *

NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200
BUDGET = 8631
t00 = time.time()
H, _, HVER = hgnc()

TRUTH = set()
for a, bs in json.load(open(f"{G}/trrust_edges.json")).items():
    for b in bs:
        u, v = sorted((str(a).upper(), str(b).upper()))
        if u != v: TRUTH.add((u, v))
PRED_PUB, NFEAT_PUB = pickle.load(open(f"{C}/.cofire_cache.pkl", "rb"))


# global gene ids over every upper-case symbol any model uses, plus TRRUST genes
_UN = {}
for m in ORDER:
    _d = load_model(m); _UN[m] = upper_map(_d["names"])
ALLG = sorted({u for m in ORDER for u in _UN[m][1]} | {g for e in TRUTH for g in e})
GID = {g: i for i, g in enumerate(ALLG)}


def graph_from(m, clean):
    """rebuild the W>=5 co-firing graph of model m from the catalogues (optionally without tie tails)."""
    d = load_model(m); names = d["names"]; rank = universe_ranks(names); raw2u, un = _UN[m]
    tail, _, _ = tie_tail_with_firing(d["T20"], rank, d["nfire"])
    keep = (np.arange(20)[None, :] < (20 - tail)[:, None]) if clean else None
    X, _ = filtered_top10(d["T20"], raw2u, un, keep)
    g_of_u = np.array([GID[u] for u in un], np.int64)
    A, B = pairs_of(X); a, b = g_of_u[A], g_of_u[B]
    k = np.minimum(a, b) * (1 << 21) + np.maximum(a, b)     # GID is alphabetical, so min = name-sorted first
    del A, B, a, b
    uk, c = np.unique(k, return_counts=True); sel = c >= W0
    out = {(ALLG[int(x >> 21)], ALLG[int(x & ((1 << 21) - 1))]): int(n) for x, n in zip(uk[sel], c[sel])}
    return out, int(len(X))


print("rebuilding graphs (raw = check against the published cache; clean = tie tails removed)", flush=True)
RAW, CLEAN, NF = {}, {}, {}
for m in ORDER:
    RAW[m], NF[m] = graph_from(m, False); CLEAN[m], _ = graph_from(m, True)
    same = RAW[m] == PRED_PUB[m]
    print(f"  {m:<13} published {len(PRED_PUB[m]):>7} pairs (nfeat {NFEAT_PUB[m]}) | rebuilt {len(RAW[m]):>7} "
          f"(nfeat {NF[m]}) identical={same} | clean {len(CLEAN[m]):>7}", flush=True)

# global alphabetical rank: sorted union of genes that appear in any model's graph (as the repo's GENES)
GENES_ALPHA = sorted({g for m in ORDER for e in PRED_PUB[m] for g in e} | {g for m in ORDER for e in CLEAN[m] for g in e})
ARANK = {g: i for i, g in enumerate(GENES_ALPHA)}


def grp(g):
    h = H.get(g); return set(h[1]) if h else set()


def lex_flags(a, b):
    p3 = a[:3] == b[:3]
    hg = bool(grp(a) & grp(b))
    ra, rb = ARANK.get(a), ARANK.get(b)
    al = ra is not None and rb is not None and abs(ra - rb) <= 20
    return {"p3": p3, "p2": a[:2] == b[:2], "hgnc": hg, "alpha20": al, "any": p3 or hg or al}


def key(e):
    i, j = sorted((GID[e[0]], GID[e[1]])); return i * (1 << 21) + j


def score(edges, truth, rng):
    if len(edges) < 2: return {"n_pairs": len(edges), "hits": 0, "null_mean": None, "fold": None, "p_emp": None}
    a = np.fromiter((GID[e[0]] for e in edges), np.int64, len(edges))
    b = np.fromiter((GID[e[1]] for e in edges), np.int64, len(edges))
    TK = np.array(sorted(key(e) for e in truth if e[0] in GID and e[1] in GID), np.int64)
    lo = np.minimum(a, b); hi = np.maximum(a, b)
    obs = int(np.isin(lo * (1 << 21) + hi, TK).sum())
    nul = np.empty(NPERM)
    for r in range(NPERM):
        s = np.concatenate([a, b]); rng.shuffle(s); h, t = s[::2], s[1::2]; ok = h != t; h, t = h[ok], t[ok]
        k = np.unique(np.minimum(h, t) * (1 << 21) + np.maximum(h, t))
        nul[r] = np.isin(k, TK).sum()
    mu, sd = float(nul.mean()), float(nul.std())
    return {"n_pairs": len(edges), "hits": obs, "null_mean": round(mu, 3), "null_sd": round(sd, 3),
            "fold": round(obs / mu, 3) if mu > 0 else None, "z": round((obs - mu) / sd, 2) if sd > 0 else None,
            "p_emp": round((int((nul >= obs).sum()) + 1) / (NPERM + 1), 5), "n_perm": NPERM}


FL_T = {e: lex_flags(*e) for e in TRUTH}
TRUTH_NO = {k: {e for e in TRUTH if not FL_T[e][k]} for k in ("p3", "hgnc", "alpha20", "any")}
RES = {"design": __doc__.strip().split("\n")[0], "n_perm": NPERM, "budget": BUDGET, "hgnc": HVER,
       "truth_pairs": len(TRUTH), "truth_pairs_flagged": {k: len(TRUTH) - len(v) for k, v in TRUTH_NO.items()},
       "models": {}}
VAR = {"no_p3": "p3", "no_hgnc": "hgnc", "no_alpha20": "alpha20", "no_lexfam": "any"}
for m in ORDER:
    t0 = time.time()
    pub, cln = PRED_PUB[m], CLEAN[m]
    R = {"n_features_raw": NFEAT_PUB[m], "published_rebuilt_identical": RAW[m] == pub}
    E = list(pub); fl = {e: lex_flags(*e) for e in E}
    R["graph_share_flagged"] = {k: round(sum(fl[e][k] for e in E) / max(len(E), 1), 4) for k in ("p3", "p2", "hgnc", "alpha20", "any")}
    hits = [e for e in E if e in TRUTH]
    R["hits"] = len(hits)
    R["hits_flagged"] = {k: sum(fl[e][k] for e in hits) for k in ("p3", "p2", "hgnc", "alpha20", "any")}
    # tie-tail dependence of each hit: weight in the published graph vs the clean graph
    lost = [e for e in hits if e not in cln]
    R["hits_lost_when_tie_tails_removed"] = len(lost)
    R["hits_lost_list"] = ["-".join(e) for e in lost]
    R["hits_flagged_list"] = {k: ["-".join(e) for e in hits if fl[e][k]] for k in ("p3", "hgnc", "alpha20")}
    V = {}
    V["published"] = score(E, TRUTH, np.random.default_rng(0))
    for vn, k in VAR.items():
        V[vn] = score([e for e in E if not fl[e][k]], TRUTH_NO[k], np.random.default_rng(0))
    Ec = list(cln); flc = {e: lex_flags(*e) for e in Ec}
    V["clean"] = score(Ec, TRUTH, np.random.default_rng(0))
    V["clean_no_lexfam"] = score([e for e in Ec if not flc[e]["any"]], TRUTH_NO["any"], np.random.default_rng(0))
    if len(E) >= BUDGET:
        V["published_top8631"] = score(sorted(pub, key=lambda e: -pub[e])[:BUDGET], TRUTH, np.random.default_rng(0))
    if len(Ec) >= BUDGET:
        V["clean_top8631"] = score(sorted(cln, key=lambda e: -cln[e])[:BUDGET], TRUTH, np.random.default_rng(0))
    R["clean_graph_share_flagged"] = {k: round(sum(flc[e][k] for e in Ec) / max(len(Ec), 1), 4) for k in ("p3", "hgnc", "alpha20", "any")}
    R["variants"] = V
    RES["models"][m] = R
    print(f"[{m}] " + " | ".join(f"{vn}: {v['n_pairs']}p {v['hits']}h {v['fold']}x p={v['p_emp']}" for vn, v in V.items())
          + f"  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(RES, open(f"{OUTDIR}/trrust_lexical.json", "w"), indent=1)

# ---- the released hypothesis list (4,349 pairs; hypothesis_pairs_full.json)
rel = json.load(open(f"{C}/hypothesis_pairs_full.json"))["pairs"]
rp = [tuple(sorted((p["pair"][0].upper(), p["pair"][1].upper()))) for p in rel]
flr = [lex_flags(*e) for e in rp]
n_clean_models = [sum(1 for m in PASSED if e in CLEAN[m]) for e in rp]
RES["released_pairs"] = {
    "n": len(rp),
    "flagged": {k: sum(f[k] for f in flr) for k in ("p3", "p2", "hgnc", "alpha20", "any")},
    "share_flagged": {k: round(sum(f[k] for f in flr) / len(rp), 4) for k in ("p3", "p2", "hgnc", "alpha20", "any")},
    "still_in_ge2_passing_models_after_tie_tail_removal": sum(1 for n in n_clean_models if n >= 2),
    "n_models_published_vs_clean_mean": [round(float(np.mean([p["n_models"] for p in rel])), 3),
                                         round(float(np.mean(n_clean_models)), 3)],
    "examples_flagged_any": ["-".join(e) for e, f in zip(rp, flr) if f["any"]][:30],
    "examples_lost_after_clean": ["-".join(e) for e, n in zip(rp, n_clean_models) if n < 2][:30]}
json.dump(RES, open(f"{OUTDIR}/trrust_lexical.json", "w"), indent=1)
print("released:", RES["released_pairs"]["share_flagged"], "kept after clean:",
      RES["released_pairs"]["still_in_ge2_passing_models_after_tie_tail_removal"], f"total {time.time()-t00:.0f}s")
