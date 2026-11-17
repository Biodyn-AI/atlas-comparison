#!/usr/bin/env python
"""Is the shared-prefix excess explained by gene families?  (notes/11_input_encoding.md, part a/c)

Same features, filter, tie-tail cleaning and slot-permutation null as lexical.py. Metrics over
within-feature pairs:
  p3_nofam     share of pairs with the same first 3 characters that do NOT share an HGNC gene group
  p3_diffgrp   among pairs whose two genes both have >= 1 HGNC group but share none: share with the
               same first 3 characters ("named alike, annotated as different families")
  p3_samegrp   among pairs sharing an HGNC group: share with the same first 3 characters (context)
Scopes: mid layer (R_MID perms) and all layers pooled (R_POOL perms); raw and clean.
    ATLAS_BASE=<copy> python prefix_beyond_family.py [R_POOL R_MID] -> prefix_beyond_family.json
"""
import json, sys, time
import numpy as np
from ie_common import *

R_POOL = int(sys.argv[1]) if len(sys.argv) > 1 else 20
R_MID = int(sys.argv[2]) if len(sys.argv) > 2 else 200
H, _, HVER = hgnc()


def attrs(d):
    raw2u, un = upper_map(d["names"])
    c3 = {}; p3 = np.array([c3.setdefault(u[:3], len(c3)) for u in un], np.int64)
    K = 4; Gk = np.full((len(un), K), -1, np.int64)
    for i, u in enumerate(un):
        h = H.get(u)
        if h and h[1]:
            g = sorted(h[1])[:K]; Gk[i, :len(g)] = g
    return raw2u, un, p3, Gk


def metrics(A, B, p3, Gk):
    s3 = p3[A] == p3[B]
    Ga, Gb = Gk[A], Gk[B]; same = np.zeros(len(A), bool)
    for i in range(Ga.shape[1]):
        v = Ga[:, i] >= 0
        for j in range(Gb.shape[1]): same |= v & (Ga[:, i] == Gb[:, j])
    both = (Ga[:, 0] >= 0) & (Gb[:, 0] >= 0); diff = both & ~same
    return {"p3_nofam": float(np.mean(s3 & ~same)),
            "p3_diffgrp": float(s3[diff].mean()) if diff.any() else None,
            "p3_samegrp": float(s3[same].mean()) if same.any() else None}


def scope(X, p3, Gk, R, rng):
    A, B = pairs_of(X); obs = metrics(A, B, p3, Gk); nul = {k: [] for k in obs}
    for _ in range(R):
        a, b = pairs_of(slot_permute(X, rng)); mm = metrics(a, b, p3, Gk)
        for k in nul:
            if mm[k] is not None: nul[k].append(mm[k])
    return {"n_features": int(len(X)), "n_pairs": int(len(A)),
            "metrics": {k: empirical(v, nul[k]) if v is not None and nul[k] else v for k, v in obs.items()}}


out = {"R_pool": R_POOL, "R_mid": R_MID, "seed": 0, "hgnc": HVER, "models": {}}
t00 = time.time()
for m in ORDER:
    t0 = time.time(); rng = np.random.default_rng(0)
    d = load_model(m); raw2u, un, p3, Gk = attrs(d)
    rank = universe_ranks(d["names"])
    tail, _, _ = tie_tail_with_firing(d["T20"], rank, d["nfire"])
    keep = np.arange(20)[None, :] < (20 - tail)[:, None]
    Xr, kr = filtered_top10(d["T20"], raw2u, un); Xc, kc = filtered_top10(d["T20"], raw2u, un, keep)
    mid = mid_layer(m)
    out["models"][m] = {"mid_layer": mid,
                        "mid": {"raw": scope(Xr[d["layer"][kr] == mid], p3, Gk, R_MID, rng),
                                "clean": scope(Xc[d["layer"][kc] == mid], p3, Gk, R_MID, rng)},
                        "pooled": {"clean": scope(Xc, p3, Gk, R_POOL, rng)}}
    r = out["models"][m]
    print(m, "mid clean p3_nofam", r["mid"]["clean"]["metrics"]["p3_nofam"]["fold"], "p3_diffgrp",
          (r["mid"]["clean"]["metrics"]["p3_diffgrp"] or {}).get("fold"), "| pooled clean p3_nofam",
          r["pooled"]["clean"]["metrics"]["p3_nofam"]["fold"], f"({time.time()-t0:.0f}s)", flush=True)
    json.dump(out, open(f"{OUTDIR}/prefix_beyond_family.json", "w"), indent=1)
print(f"done {time.time()-t00:.0f}s")
