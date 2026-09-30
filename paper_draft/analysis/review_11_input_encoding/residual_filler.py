#!/usr/bin/env python
"""Review check for notes/11_input_encoding.md: does filler survive in the "clean" lists?

Filler gene set F_m = genes whose appearances in detected tie-tail slots (ie_common.tie_tail_with_firing,
all layers, top-20) are >= 50 % of all their top-20 appearances AND >= 20 in number. For the clean
top-10 lists (same filter as lexical.py), per layer and pooled, report:
  * share of slots holding an F gene (raw vs clean)
  * share of same-3-letter-prefix pairs that are F-F pairs (raw vs clean)
  * the clean p3 share recomputed with every F gene dropped from the clean lists ("clean minus F")
    against the same slot-permutation null (R perms, seed 0) - a stricter clean
Also: share of the published top-8,631 C2S / UCE / tGPT pairs that touch an F gene.
    ATLAS_BASE=<copy> python residual_filler.py [R] -> residual_filler.json
"""
import json, os, pickle, sys
from collections import Counter
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "input_encoding"))
from ie_common import *   # noqa

R = int(sys.argv[1]) if len(sys.argv) > 1 else 20
PRED_PUB, _ = pickle.load(open(f"{C}/.cofire_cache.pkl", "rb"))
out = {"R": R, "seed": 0, "filler_rule": "tail share >= 0.5 of the gene's top-20 appearances and >= 20 tail appearances",
       "models": {}}


def p3_scope(X, p3, rng, R):
    A, B = pairs_of(X)
    if len(A) == 0: return None
    obs = float(np.mean(p3[A] == p3[B])); nul = []
    for _ in range(R):
        a, b = pairs_of(slot_permute(X, rng)); nul.append(float(np.mean(p3[a] == p3[b])))
    return {"n_features": int(len(X)), "n_pairs": int(len(A)), **empirical(obs, nul)}


for m in ["C2S", "UCE", "tGPT", "Tahoe"]:
    rng = np.random.default_rng(0)
    d = load_model(m); names = d["names"]; rank = universe_ranks(names); raw2u, un = upper_map(names)
    T20 = d["T20"]
    tail, _, _ = tie_tail_with_firing(T20, rank, d["nfire"])
    intail = np.arange(20)[None, :] >= (20 - tail)[:, None]
    v = T20 >= 0
    tot = np.bincount(T20[v], minlength=len(names)); tl = np.bincount(T20[v & intail], minlength=len(names))
    Fraw = (tl >= 20) & (tl >= 0.5 * np.maximum(tot, 1))
    Fu = np.zeros(len(un), bool); Fu[raw2u[Fraw]] = True
    c3 = {}; p3 = np.array([c3.setdefault(u[:3], len(c3)) for u in un], np.int64)
    keep = ~intail
    Xr, kr = filtered_top10(T20, raw2u, un); Xc, kc = filtered_top10(T20, raw2u, un, keep)
    # stricter clean: also blank every F gene anywhere in the list
    keepF = keep & ~(v & Fraw[np.maximum(T20, 0)])
    Xs, ks = filtered_top10(T20, raw2u, un, keepF)
    Lr, Lc, Ls = d["layer"][kr], d["layer"][kc], d["layer"][ks]

    def desc(X):
        s = X >= 0; slots = int(s.sum()); fs = int((s & Fu[np.maximum(X, 0)]).sum())
        A, B = pairs_of(X); same = p3[A] == p3[B]; ff = Fu[A] & Fu[B]
        return {"slots": slots, "share_slots_F": round(fs / max(slots, 1), 4),
                "same_p3_pairs": int(same.sum()), "share_same_p3_pairs_FF": round(float((same & ff).sum() / max(same.sum(), 1)), 4)}

    M = {"n_F_genes": int(Fraw.sum()), "F_examples": [names[i] for i in np.argsort(-tl)[:15]],
         "F_rank_positions": sorted({round(float(rank[i]) / len(names), 3) for i in np.where(Fraw)[0]})[:20],
         "pooled": {"raw": desc(Xr), "clean": desc(Xc)}, "per_layer": {}}
    for L in sorted(np.unique(d["layer"]).tolist()):
        M["per_layer"][str(L)] = {"raw": desc(Xr[Lr == L]), "clean": desc(Xc[Lc == L])}
    # stricter clean p3 fold at the layers where clean > raw and at the mid layer
    mid = mid_layer(m)
    test_layers = {"C2S": [0, 1, 4, mid], "UCE": [0, 2, 16, 30], "tGPT": [0, 4, 6, 7], "Tahoe": [0, 1, mid]}[m]
    M["strict_clean_p3"] = {}
    for L in test_layers:
        M["strict_clean_p3"][str(L)] = {"clean": p3_scope(Xc[Lc == L], p3, rng, R), "clean_minus_F": p3_scope(Xs[Ls == L], p3, rng, R)}
    if len(PRED_PUB[m]) >= 8631:
        top = sorted(PRED_PUB[m], key=lambda e: -PRED_PUB[m][e])[:8631]
        Fset = {un[i] for i in np.where(Fu)[0]}
        M["published_top8631_pairs_touching_F"] = round(sum(1 for a, b in top if a in Fset or b in Fset) / 8631, 4)
        M["published_top8631_pairs_both_F"] = round(sum(1 for a, b in top if a in Fset and b in Fset) / 8631, 4)
    out["models"][m] = M
    print(m, "F", M["n_F_genes"], M["F_examples"][:6], "pooled", M["pooled"], flush=True)
    for L, x in M["strict_clean_p3"].items():
        print("   L", L, "clean", x["clean"] and (x["clean"]["fold"], x["clean"]["n_pairs"]),
              "clean-F", x["clean_minus_F"] and (x["clean_minus_F"]["fold"], x["clean_minus_F"]["n_pairs"]), flush=True)
    json.dump(out, open(f"{HERE}/residual_filler.json", "w"), indent=1)
    del d, T20, Xr, Xc, Xs
