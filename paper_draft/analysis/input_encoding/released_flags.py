"""Name and family flags of the released pair list (Table S-encoding d), as in trrust_lexical.py:154-170,
on the current hypothesis_pairs_full.json. Rebuilds the same gene order (GENES_ALPHA) as trrust_lexical.py.
Output: released_flags.json (this folder)"""
from __future__ import annotations
import json, pickle, time
import numpy as np
from ie_common import *

t0 = time.time()
H, _, HVER = hgnc()
TRUTH = set()
for a, bs in json.load(open(f"{G}/trrust_edges.json")).items():
    for b in bs:
        u, v = sorted((str(a).upper(), str(b).upper()))
        if u != v: TRUTH.add((u, v))
PRED_PUB, _ = pickle.load(open(f"{C}/.cofire_cache.pkl", "rb"))
_UN = {}
for m in ORDER:
    _d = load_model(m); _UN[m] = upper_map(_d["names"])
ALLG = sorted({u for m in ORDER for u in _UN[m][1]} | {g for e in TRUTH for g in e})
GID = {g: i for i, g in enumerate(ALLG)}


def clean_graph_genes(m):
    d = load_model(m); names = d["names"]; rank = universe_ranks(names); raw2u, un = _UN[m]
    tail, _, _ = tie_tail_with_firing(d["T20"], rank, d["nfire"])
    keep = (np.arange(20)[None, :] < (20 - tail)[:, None])
    X, _ = filtered_top10(d["T20"], raw2u, un, keep)
    g_of_u = np.array([GID[u] for u in un], np.int64)
    A, B = pairs_of(X); a, b = g_of_u[A], g_of_u[B]
    k = np.minimum(a, b) * (1 << 21) + np.maximum(a, b)
    uk, c = np.unique(k, return_counts=True); sel = c >= W0
    ids = np.unique(np.concatenate([uk[sel] >> 21, uk[sel] & ((1 << 21) - 1)]))
    return {ALLG[int(i)] for i in ids}


CG = set()
for m in ORDER:
    CG |= clean_graph_genes(m); print(m, f"{time.time() - t0:.0f}s", flush=True)
GENES_ALPHA = sorted({g for m in ORDER for e in PRED_PUB[m] for g in e} | CG)
ARANK = {g: i for i, g in enumerate(GENES_ALPHA)}


def grp(g):
    h = H.get(g); return set(h[1]) if h else set()


def lex_flags(a, b):
    p3 = a[:3] == b[:3]; hg = bool(grp(a) & grp(b))
    ra, rb = ARANK.get(a), ARANK.get(b)
    al = ra is not None and rb is not None and abs(ra - rb) <= 20
    return {"p3": p3, "p2": a[:2] == b[:2], "hgnc": hg, "alpha20": al, "any": p3 or hg or al}


rel = json.load(open(f"{C}/hypothesis_pairs_full.json"))["pairs"]
rp = [tuple(sorted((p["pair"][0].upper(), p["pair"][1].upper()))) for p in rel]
flr = [lex_flags(*e) for e in rp]
out = {"n": len(rp), "hgnc": HVER,
       "flagged": {k: sum(f[k] for f in flr) for k in ("p3", "p2", "hgnc", "alpha20", "any")},
       "share_flagged": {k: round(sum(f[k] for f in flr) / len(rp), 4) for k in ("p3", "p2", "hgnc", "alpha20", "any")},
       "examples_flagged_any": ["-".join(e) for e, f in zip(rp, flr) if f["any"]][:30]}
json.dump(out, open(f"{OUTDIR}/released_flags.json", "w"), indent=1)
print(out["flagged"], out["share_flagged"], f"{time.time() - t0:.0f}s")
