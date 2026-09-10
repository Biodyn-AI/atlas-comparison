#!/usr/bin/env python
"""Stage 3: the deliverable. Restricts corroboration to the models that PASSED stage 2(A), fixes the
degenerate reporting when a null never produces a hit, flags paralog families, and writes the final
hypothesis list.  python scripts/hypothesis_trrust3.py [NPERM]
"""
from __future__ import annotations
import json, glob, sys, itertools, os, pickle, re
import numpy as np
from collections import defaultdict

BASE = "/Users/annaantipova/Desktop/biomech"
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"; TS = f"{BASE}/outputs/atlas/ts3_out"
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
PASSED = ["Tahoe", "scGPT", "UCE", "C2S", "Geneformer"]          # significant in stage 2(A)
TOPG, W0 = 10, 5
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200
CACHE = f"{C}/.cofire_cache.pkl"
rng = np.random.default_rng(0)

TR = json.load(open(f"{G}/trrust_edges.json")); TRUTH = set()
for a, bs in TR.items():
    for b in bs:
        u, v = sorted((str(a).upper(), str(b).upper()))
        if u != v: TRUTH.add((u, v))
SEEN = set()
for src in ("GO_BP", "KEGG", "Reactome"):
    for t, genes in json.load(open(f"{G}/{src}_gene_sets.json")).items():
        gl = sorted({str(x).upper() for x in genes})
        if len(gl) <= 200: SEEN.update(itertools.combinations(gl, 2))
STRING = set()
for a, bs in json.load(open(f"{G}/string_edges.json")).items():
    for b in bs:
        u, v = sorted((str(a).upper(), str(b).upper()))
        if u != v: STRING.add((u, v))


def build():
    P, N = {}, {}
    for m in ORDER:
        cnt = defaultdict(int); nfeat = 0
        d = ALLCAT if glob.glob(f"{ALLCAT}/{m}/feature_catalog_L*.json") else TS
        for p in sorted(glob.glob(f"{d}/{m}/feature_catalog_L*.json")):
            for f in json.load(open(p))["features"].values():
                s = sorted({str(x).upper() for x in f.get("top_genes", [])[:TOPG]
                            if not str(x).upper().startswith("ENSG")})
                if len(s) < 2: continue
                nfeat += 1
                for a, b in itertools.combinations(s, 2): cnt[(a, b)] += 1
        P[m] = {k: c for k, c in cnt.items() if c >= W0}; N[m] = nfeat
        print(f"  {m}: {nfeat} features -> {len(P[m])} pairs", flush=True)
    return P, N


if os.path.exists(CACHE):
    PRED, NFEAT = pickle.load(open(CACHE, "rb")); print("loaded cached co-firing graphs", flush=True)
else:
    print("building co-firing graphs", flush=True)
    PRED, NFEAT = build(); pickle.dump((PRED, NFEAT), open(CACHE, "wb"))

GENES = sorted({g for m in ORDER for e in PRED[m] for g in e}); GID = {g: i for i, g in enumerate(GENES)}
TKEY = np.array(sorted({(lambda i, j: i * (1 << 21) + j)(*sorted((GID[a], GID[b])))
                        for a, b in TRUTH if a in GID and b in GID}), np.int64)


def config_shuffle(edges):
    a = np.fromiter((GID[e[0]] for e in edges), np.int32, len(edges))
    b = np.fromiter((GID[e[1]] for e in edges), np.int32, len(edges))
    s = np.concatenate([a, b]); rng.shuffle(s)
    h, t = s[::2], s[1::2]; ok = h != t; h, t = h[ok], t[ok]
    lo = np.minimum(h, t).astype(np.int64); hi = np.maximum(h, t).astype(np.int64)
    return np.unique(lo * (1 << 21) + hi)


def keys_of(pairs):
    o = np.empty(len(pairs), np.int64)
    for n, (a, b) in enumerate(pairs):
        i, j = sorted((GID[a], GID[b])); o[n] = i * (1 << 21) + j
    return o


print(f"\nCross-model corroboration among the {len(PASSED)} models that passed: {', '.join(PASSED)}", flush=True)
allp = defaultdict(int)
for m in PASSED:
    for e in PRED[m]: allp[e] += 1
keys = keys_of(list(allp)); nmod = np.fromiter(allp.values(), np.int32, len(allp))
istruth = np.isin(keys, TKEY)

null_hits = defaultdict(list); null_n = defaultdict(list)
for it in range(NPERM):
    cnt = defaultdict(int)
    for m in PASSED:
        for kk in config_shuffle(list(PRED[m])): cnt[int(kk)] += 1
    kk = np.fromiter(cnt.keys(), np.int64, len(cnt)); vv = np.fromiter(cnt.values(), np.int32, len(cnt))
    ist = np.isin(kk, TKEY)
    for k in range(1, len(PASSED) + 1):
        sel = vv >= k; null_hits[k].append(int(ist[sel].sum())); null_n[k].append(int(sel.sum()))
    if (it + 1) % 50 == 0: print(f"  perm {it+1}/{NPERM}", flush=True)

curve = {}
print(f"\n{'k':>3} {'pairs':>8} {'hits':>5} {'precision':>10} {'null prec':>11} {'fold':>8} {'p':>8}")
for k in range(1, len(PASSED) + 1):
    sel = nmod >= k; n = int(sel.sum())
    if n == 0: break
    h = int(istruth[sel].sum()); prec = h / n
    nh = np.array(null_hits[k], float); nn = np.array(null_n[k], float)
    npr = nh / np.maximum(nn, 1)
    zero = int((nh == 0).sum())
    fold = (prec / npr.mean()) if npr.mean() > 0 else None
    p = (int((npr >= prec).sum()) + 1) / (NPERM + 1)
    curve[k] = {"n_pairs": n, "hits": h, "precision": round(prec, 6),
                "null_precision_mean": round(float(npr.mean()), 8),
                "null_hits_mean": round(float(nh.mean()), 2),
                "null_pairs_mean": round(float(nn.mean()), 1),
                "null_perms_with_zero_hits": zero, "n_perm": NPERM,
                "fold": round(fold, 2) if fold else None, "p_emp": round(p, 5)}
    fs = f"{fold:>8.1f}" if fold else f"{'inf*':>8}"
    print(f"{k:>3} {n:>8} {h:>5} {prec:>10.5f} {npr.mean():>11.7f} {fs} {p:>8.4f}"
          + (f"   (* null gave 0 hits in {zero}/{NPERM} perms)" if not fold else ""))

FAM = re.compile(r"^([A-Z]{2,}[0-9]*?)[0-9]*[A-Z]?$")
def family(g):
    m = FAM.match(g); return m.group(1) if m else g

K = 2                                     # corroborated by >=2 validated models
cands = []
for e, n in allp.items():
    if n < K or e in TRUTH or e in STRING or e in SEEN: continue
    fa, fb = family(e[0]), family(e[1])
    cands.append({"pair": list(e), "n_models": n,
                  "models": sorted(m for m in PASSED if e in PRED[m]),
                  "weight": int(sum(PRED[m][e] for m in PASSED if e in PRED[m])),
                  "same_family": bool(fa == fb and len(fa) >= 3)})
cands.sort(key=lambda c: (c["same_family"], -c["n_models"], -c["weight"]))
novel = [c for c in cands if not c["same_family"]]
print(f"\nHypothesis list at k>={K}: {len(cands)} pairs in NO database "
      f"({len(novel)} after dropping same-family paralogs)")
print(f"Expected hit rate on held-out TRRUST at this confidence: {curve[K]['precision']:.5f} "
      f"vs {curve[K]['null_precision_mean']:.7f} for degree-matched chance "
      f"({curve[K]['fold']}x)" if curve[K]["fold"] else "")
for c in novel[:25]:
    print(f"   {c['pair'][0]:>12} — {c['pair'][1]:<12} {c['n_models']} models "
          f"({', '.join(c['models'])}), weight {c['weight']}")

json.dump({"design": "gene pairs co-firing in >=%d SAE features predict held-out TRRUST TF->target edges" % W0,
           "truth": "TRRUST curated regulatory edges; excluded from the calibrated annotator (no PPI/regulatory DBs)",
           "null": "configuration model, every gene's co-firing degree preserved exactly",
           "models_validated": PASSED, "top_genes_per_feature": TOPG, "min_features_per_pair": W0,
           "n_perm": NPERM, "cross_model_curve": curve, "k_used": K,
           "n_hypotheses": len(novel), "hypotheses": novel[:80],
           "n_same_family_dropped": len(cands) - len(novel)},
          open(f"{C}/hypothesis_final.json", "w"), indent=1)
print("\n==> hypothesis_final.json", flush=True)
