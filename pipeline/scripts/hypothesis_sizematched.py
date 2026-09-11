#!/usr/bin/env python
"""Is the per-model ranking of Fig 5A a ranking of model quality, or of SAE dictionary size?

The number of predicted pairs a model contributes is set by its dictionary (4 x d_model) and its
layer count, and spans four orders of magnitude -- AIDO.Cell offers 2 pairs, C2S-Scale 147,728. Hit
counts correlate with graph size at r = +0.90. The fold is nominally protected, since each model is
compared to a null built from its own graph, but a reader of Fig 5A sees a league table and cannot
tell capability from capacity.

So we give every model the SAME prediction budget and re-score:

  top-N   the N highest-weight pairs -- what a practitioner with a fixed budget would actually take,
          and the comparison that asks "given equal effort, whose predictions are better?"
  rand-N  N pairs drawn at random from the model's graph, repeated, which isolates size alone

N defaults to the smallest graph among the models that pass at full size, so no model is scored on a
sample larger than it has. Models whose graph is smaller than N are reported as untestable at that
budget rather than as failures -- the distinction the manuscript previously elided for AIDO.Cell.
Null and read-out are unchanged: configuration model, exact degree preserved, TRRUST held out.
    python scripts/hypothesis_sizematched.py [N] [NDRAW] [NPERM]
        -> outputs/atlas/comparative/hypothesis_sizematched.json
"""
from __future__ import annotations
import json, pickle, sys, os
import numpy as np

BASE = "/Users/annaantipova/Desktop/biomech"
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
CACHE = f"{C}/.cofire_cache.pkl"
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 0          # 0 = smallest passing graph
NDRAW = int(sys.argv[2]) if len(sys.argv) > 2 else 20     # random draws per model
NPERM = int(sys.argv[3]) if len(sys.argv) > 3 else 100
rng = np.random.default_rng(0)

TRUTH = set()
for a, bs in json.load(open(f"{G}/trrust_edges.json")).items():
    for b in bs:
        u, v = sorted((str(a).upper(), str(b).upper()))
        if u != v: TRUTH.add((u, v))

if not os.path.exists(CACHE):
    sys.exit("co-firing cache missing; run scripts/hypothesis_trrust3.py first")
PRED, NFEAT = pickle.load(open(CACHE, "rb"))
GENES = sorted({g for m in ORDER for e in PRED.get(m, {}) for g in e})
GID = {g: i for i, g in enumerate(GENES)}
TK = np.array(sorted({(lambda i, j: i * (1 << 21) + j)(*sorted((GID[a], GID[b])))
                      for a, b in TRUTH if a in GID and b in GID}), np.int64)


def keys(edges):
    o = np.empty(len(edges), np.int64)
    for n_, (a, b) in enumerate(edges):
        i, j = sorted((GID[a], GID[b])); o[n_] = i * (1 << 21) + j
    return o


def shuffle_keys(edges):
    a = np.fromiter((GID[e[0]] for e in edges), np.int32, len(edges))
    b = np.fromiter((GID[e[1]] for e in edges), np.int32, len(edges))
    s = np.concatenate([a, b]); rng.shuffle(s)
    h, t = s[::2], s[1::2]; ok = h != t; h, t = h[ok], t[ok]
    return np.unique(np.minimum(h, t).astype(np.int64) * (1 << 21) + np.maximum(h, t).astype(np.int64))


def score(edges):
    """observed TRRUST hits and the degree-preserving null for this exact edge set"""
    obs = int(np.isin(keys(edges), TK).sum())
    nul = np.array([int(np.isin(shuffle_keys(edges), TK).sum()) for _ in range(NPERM)], float)
    nm = float(nul.mean())
    return obs, nm, (obs / nm if nm > 0 else None), (int((nul >= obs).sum()) + 1) / (NPERM + 1)


FULL = json.load(open(f"{C}/hypothesis_trrust2.json"))["per_model"]
sizes = {m: len(PRED.get(m, {})) for m in ORDER}
passing = [m for m in ORDER if FULL.get(m, {}).get("p_emp", 1) <= 0.05]
if not N:
    N = min(sizes[m] for m in passing)
print(f"budget N = {N} pairs (smallest graph among the models passing at full size: "
      f"{min(passing, key=lambda m: sizes[m])})\n")

res = {"budget": N, "n_draws": NDRAW, "n_perm": NPERM,
       "null": "configuration model on the budgeted edge set; TRRUST held out",
       "graph_sizes": sizes, "models": {}}
print(f"{'model':<14}{'graph':>9}{'top-N fold':>12}{'p':>8}{'rand-N fold':>14}{'':>3}")
for m in ORDER:
    g = PRED.get(m, {})
    if len(g) < N:
        res["models"][m] = {"graph": len(g), "testable": False,
                            "reason": f"graph smaller than the {N}-pair budget"}
        print(f"{m:<14}{len(g):>9}{'— untestable at this budget':>40}")
        continue
    edges = sorted(g, key=lambda e: -g[e])[:N]                      # top-N by weight
    o, nm, fold, p = score(edges)
    all_e = list(g)
    rf = []
    for _ in range(NDRAW):
        pick = [all_e[i] for i in rng.choice(len(all_e), N, replace=False)]
        oo, nn, ff, _ = score(pick)
        if ff is not None: rf.append(ff)
    rmean = float(np.mean(rf)) if rf else None
    rsd = float(np.std(rf)) if rf else None
    res["models"][m] = {"graph": len(g), "testable": True,
                        "topN": {"hits": o, "null_mean": round(nm, 2),
                                 "fold": round(fold, 2) if fold else None, "p_emp": round(p, 4)},
                        "randN": {"fold_mean": round(rmean, 2) if rmean else None,
                                  "fold_sd": round(rsd, 2) if rsd else None, "n_draws": len(rf)},
                        "full_fold": FULL.get(m, {}).get("fold")}
    print(f"{m:<14}{len(g):>9}{(str(round(fold,2))+'x' if fold else '—'):>12}{p:>8.4f}"
          f"{(f'{rmean:.2f}±{rsd:.2f}x' if rmean else '—'):>14}", flush=True)

ok = {m: d for m, d in res["models"].items() if d.get("testable")}
full_rank = sorted(ok, key=lambda m: -(FULL[m]["fold"] or 0))
top_rank = sorted(ok, key=lambda m: -(ok[m]["topN"]["fold"] or 0))
res["ranking_full_size"] = full_rank
res["ranking_size_matched"] = top_rank
res["ranking_preserved"] = full_rank == top_rank
print(f"\nranking at full size : {' > '.join(full_rank)}")
print(f"ranking at equal N   : {' > '.join(top_rank)}")
print(f"ranking preserved    : {res['ranking_preserved']}")
if len(ok) > 2:
    a = [FULL[m]["fold"] or 0 for m in ok]; b = [ok[m]["topN"]["fold"] or 0 for m in ok]
    res["spearman_full_vs_matched"] = round(float(np.corrcoef(
        np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]), 3)
    print(f"rank correlation     : {res['spearman_full_vs_matched']}")
json.dump(res, open(f"{C}/hypothesis_sizematched.json", "w"), indent=1)
print("\n==> hypothesis_sizematched.json", flush=True)
