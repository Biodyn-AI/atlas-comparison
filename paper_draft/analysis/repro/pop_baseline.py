#!/usr/bin/env python
"""Popularity-only predictors for the held-out TRRUST test (the paper's 'gene popularity' baseline,
Methods and Supplementary Table S-equalbudget). Rebuilt from the paper's description, because the
original script lived in working notes that were not released:

  "Two predictors that know only how often each gene appears in top-gene lists (gene popularity)
   were scored in the same way. One takes the pairs with the largest product of the two genes'
   feature counts; the other takes all pairs among the most featured genes. Each is cut to the
   model's graph size."

Feature count of a gene = number of features (all layers, top-10 genes, ENSG ids and features with
fewer than two genes dropped) whose top-10 contains it: the same lists the co-firing graph is built
from (scripts/hypothesis_trrust2.py:cofire_pairs).
  HUB    : the n pairs with the largest product of feature counts (ties by gene name).
  CLIQUE : all pairs among the k most featured genes, k the smallest with k(k-1)/2 >= n, cut to n
           pairs in the same product order.
n = the model's W>=5 co-firing graph size, and separately the equal-budget size of
hypothesis_sizematched.py. Scored like hypothesis_trrust2.py: TRRUST hits against a configuration
model that keeps every gene's degree (200 rewirings), fold = hits / null mean, one-sided p.

    python pop_baseline.py [NPERM=200] -> pop_baseline.json next to this script
"""
from __future__ import annotations

import glob, itertools, json, os, sys
from collections import defaultdict
import numpy as np

BASE = os.environ.get("ATLAS_BASE")
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"; TS = f"{BASE}/outputs/atlas/ts3_out"
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
TOPG, W0 = 10, 5
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200
BUDGET = json.load(open(f"{C}/hypothesis_sizematched.json"))["budget"]

TRUTH = set()
for a, bs in json.load(open(f"{G}/trrust_edges.json")).items():
    for b in bs:
        u, v = sorted((str(a).upper(), str(b).upper()))
        if u != v:
            TRUTH.add((u, v))


def lists(m):
    d = ALLCAT if glob.glob(f"{ALLCAT}/{m}/feature_catalog_L*.json") else TS
    for p in sorted(glob.glob(f"{d}/{m}/feature_catalog_L*.json")):
        for f in json.load(open(p))["features"].values():
            s = sorted({str(x).upper() for x in f.get("top_genes", [])[:TOPG] if not str(x).upper().startswith("ENSG")})
            if len(s) >= 2:
                yield s


def predictors(m):
    cnt = defaultdict(int); pair = defaultdict(int)
    for s in lists(m):
        for g in s:
            cnt[g] += 1
        for a, b in itertools.combinations(s, 2):
            pair[(a, b)] += 1
    n = sum(1 for v in pair.values() if v >= W0)
    genes = sorted(cnt, key=lambda g: (-cnt[g], g))
    return cnt, genes, n


def top_product_pairs(cnt, genes, n, pool):
    """the n pairs with the largest count product among the first `pool` genes"""
    g = genes[:pool]
    c = np.array([cnt[x] for x in g], float)
    i, j = np.triu_indices(len(g), 1)
    prod = c[i] * c[j]
    order = np.lexsort((j, i, -prod))[:n]
    return [(min(g[i[k]], g[j[k]]), max(g[i[k]], g[j[k]])) for k in order]


def hub(cnt, genes, n):
    pool = len(genes)
    k = 2
    while k * (k - 1) // 2 < 4 * n and k < pool:      # candidates: enough genes that the top-n products are inside
        k += max(1, k // 4)
    return top_product_pairs(cnt, genes, n, min(pool, k))


def clique(cnt, genes, n):
    k = 2
    while k * (k - 1) // 2 < n and k < len(genes):
        k += 1
    return top_product_pairs(cnt, genes, n, k)


def score(edges, rng):
    if not edges:
        return None
    gs = sorted({x for e in edges for x in e}); gid = {x: i for i, x in enumerate(gs)}
    tk = np.array(sorted(min(gid[a], gid[b]) * (1 << 21) + max(gid[a], gid[b])
                         for a, b in TRUTH if a in gid and b in gid), np.int64)
    key = np.array([min(gid[a], gid[b]) * (1 << 21) + max(gid[a], gid[b]) for a, b in edges], np.int64)
    obs = int(np.isin(key, tk).sum())
    a = np.array([gid[e[0]] for e in edges], np.int32); b = np.array([gid[e[1]] for e in edges], np.int32)
    null = []
    for _ in range(NPERM):
        st = np.concatenate([a, b]); rng.shuffle(st)
        h, t = st[::2], st[1::2]; ok = h != t
        k2 = np.unique(np.minimum(h[ok], t[ok]).astype(np.int64) * (1 << 21) + np.maximum(h[ok], t[ok]))
        null.append(int(np.isin(k2, tk).sum()))
    null = np.array(null, float)
    return {"n_pairs": len(edges), "hits": obs, "null_mean": round(float(null.mean()), 3),
            "fold": round(obs / null.mean(), 2) if null.mean() > 0 else None,
            "p_emp": round((int((null >= obs).sum()) + 1) / (NPERM + 1), 5)}


out = {"design": __doc__.strip().splitlines()[0], "n_perm": NPERM, "budget": BUDGET, "models": {}}
for mi, m in enumerate(ORDER):
    cnt, genes, n = predictors(m)
    rec = {"graph_size": n, "n_featured_genes": len(genes)}
    for name, fn in (("HUB", hub), ("CLIQUE", clique)):
        rec[name] = score(fn(cnt, genes, n), np.random.default_rng(1000 + mi)) if n else None
        rec[f"{name}_budget"] = score(fn(cnt, genes, BUDGET), np.random.default_rng(2000 + mi)) if n >= BUDGET else None
    out["models"][m] = rec
    print(m, n, {k: (v or {}).get("hits") for k, v in rec.items() if isinstance(v, dict) or v is None}, flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pop_baseline.json"), "w"), indent=1)
print("==> pop_baseline.json")
