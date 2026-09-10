#!/usr/bin/env python
"""Stage 2 of the held-out hypothesis test (see hypothesis_trrust.py for the design).

(A) Validate the analytic configuration-model expectation with real permutations: rewire each
    model's co-firing graph keeping every gene's degree exact, recount TRRUST hits, N times.
(B) Does COMPARING models make the generator better? Score each pair by how many models predict
    it, and measure precision against held-out TRRUST as that number rises -- with the same
    degree-preserving null applied to every model independently.
(C) Emit the hypothesis list: highest-confidence, multi-model pairs absent from TRRUST, STRING and
    every curated pathway, with the precision estimated from (B).
    python scripts/hypothesis_trrust2.py [NPERM]  -> outputs/atlas/comparative/hypothesis_trrust2.json
"""
from __future__ import annotations
import json, glob, sys, itertools
import numpy as np
from collections import defaultdict

BASE = "/Users/annaantipova/Desktop/biomech"
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"; TS = f"{BASE}/outputs/atlas/ts3_out"
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
TOPG, W0 = 10, 5                # top-10 genes per feature; a pair is "predicted" at >=5 features
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 500
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


def cofire_pairs(m):
    """set of (geneA,geneB) predicted by model m at >=W0 features, plus weights"""
    cnt = defaultdict(int); nfeat = 0
    d = ALLCAT if glob.glob(f"{ALLCAT}/{m}/feature_catalog_L*.json") else TS
    for p in sorted(glob.glob(f"{d}/{m}/feature_catalog_L*.json")):
        for f in json.load(open(p))["features"].values():
            s = sorted({str(x).upper() for x in f.get("top_genes", [])[:TOPG]
                        if not str(x).upper().startswith("ENSG")})
            if len(s) < 2: continue
            nfeat += 1
            for a, b in itertools.combinations(s, 2): cnt[(a, b)] += 1
    return {k: c for k, c in cnt.items() if c >= W0}, nfeat


def config_shuffle(edges, gid):
    """configuration model: exact degree preserved, edges rewired; self-loops/duplicates dropped"""
    a = np.fromiter((gid[e[0]] for e in edges), np.int32, len(edges))
    b = np.fromiter((gid[e[1]] for e in edges), np.int32, len(edges))
    stubs = np.concatenate([a, b]); rng.shuffle(stubs)
    h = stubs[::2]; t = stubs[1::2]
    ok = h != t
    h, t = h[ok], t[ok]
    lo = np.minimum(h, t).astype(np.int64); hi = np.maximum(h, t).astype(np.int64)
    return np.unique(lo * (1 << 21) + hi)


print("building per-model co-firing graphs (all layers, top-10 genes, pairs in >=%d features)" % W0, flush=True)
PRED, NFEAT = {}, {}
for m in ORDER:
    PRED[m], NFEAT[m] = cofire_pairs(m)
    print(f"  {m}: {NFEAT[m]} features -> {len(PRED[m])} predicted pairs", flush=True)

GENES = sorted({g for m in ORDER for e in PRED[m] for g in e})
GID = {g: i for i, g in enumerate(GENES)}
truth_key = set()
for a, b in TRUTH:
    if a in GID and b in GID:
        i, j = sorted((GID[a], GID[b])); truth_key.add(i * (1 << 21) + j)
TKEY = np.array(sorted(truth_key), np.int64)
print(f"\nuniverse: {len(GENES)} genes, {len(TKEY)} TRRUST edges reachable\n", flush=True)


def keys_of(pairs):
    out = np.empty(len(pairs), np.int64)
    for n, (a, b) in enumerate(pairs):
        i, j = sorted((GID[a], GID[b])); out[n] = i * (1 << 21) + j
    return out

# ---- (A) permutation validation, per model ---------------------------------
print("(A) permutation validation of the degree-matched null", flush=True)
A = {}
for m in ORDER:
    E = list(PRED[m])
    if not E: continue
    obs = int(np.isin(keys_of(E), TKEY).sum())
    nul = np.array([int(np.isin(config_shuffle(E, GID), TKEY).sum()) for _ in range(NPERM)], float)
    z = (obs - nul.mean()) / (nul.std() or 1)
    p = (int((nul >= obs).sum()) + 1) / (NPERM + 1)
    A[m] = {"n_pairs": len(E), "hits": obs, "null_mean": round(float(nul.mean()), 2),
            "null_sd": round(float(nul.std()), 2), "fold": round(obs / max(nul.mean(), 1e-9), 2),
            "z": round(float(z), 2), "p_emp": round(p, 5)}
    print(f"  {m:<13} {obs:>4} hits vs null {nul.mean():>7.2f}±{nul.std():<5.2f}  "
          f"fold {A[m]['fold']:>6}  z {A[m]['z']:>7}  p {A[m]['p_emp']}", flush=True)

# ---- (B) does cross-model agreement raise precision? -----------------------
print("\n(B) precision vs number of models predicting the pair", flush=True)
allp = defaultdict(int)
for m in ORDER:
    for e in PRED[m]: allp[e] += 1
keys = keys_of(list(allp)); nmod = np.fromiter(allp.values(), np.int32, len(allp))
istruth = np.isin(keys, TKEY)
obs_curve = {}
for k in range(1, len(ORDER) + 1):
    sel = nmod >= k
    if sel.sum() == 0: break
    obs_curve[k] = {"n_pairs": int(sel.sum()), "hits": int(istruth[sel].sum()),
                    "precision": round(float(istruth[sel].sum()) / int(sel.sum()), 5)}

null_curve = defaultdict(list)
for it in range(NPERM // 5):
    cnt = defaultdict(int)
    for m in ORDER:
        E = list(PRED[m])
        if not E: continue
        for kk in config_shuffle(E, GID): cnt[int(kk)] += 1
    kk = np.fromiter(cnt.keys(), np.int64, len(cnt)); vv = np.fromiter(cnt.values(), np.int32, len(cnt))
    it_ = np.isin(kk, TKEY)
    for k in obs_curve:
        sel = vv >= k
        null_curve[k].append(float(it_[sel].sum()) / max(int(sel.sum()), 1))
    if (it + 1) % 20 == 0: print(f"    perm {it+1}/{NPERM//5}", flush=True)

B = {}
for k, o in obs_curve.items():
    nc = np.array(null_curve[k], float)
    B[k] = {**o, "null_precision": round(float(nc.mean()), 6), "null_sd": round(float(nc.std()), 6),
            "fold": round(o["precision"] / max(float(nc.mean()), 1e-12), 2),
            "z": round(float((o["precision"] - nc.mean()) / (nc.std() or 1)), 2),
            "p_emp": round((int((nc >= o["precision"]).sum()) + 1) / (len(nc) + 1), 5)}
    print(f"  >={k} models: {o['n_pairs']:>7} pairs, {o['hits']:>4} hits, precision "
          f"{o['precision']:.5f} vs null {nc.mean():.5f}  ->  {B[k]['fold']}x  p={B[k]['p_emp']}", flush=True)

# ---- (C) the hypothesis list ------------------------------------------------
best_k = max((k for k in B if B[k]["n_pairs"] >= 20), key=lambda k: B[k]["fold"])
cands = []
for e, n in allp.items():
    if n < best_k: continue
    if e in TRUTH or e in STRING or e in SEEN: continue
    cands.append({"pair": list(e), "n_models": n,
                  "models": sorted(m for m in ORDER if e in PRED[m]),
                  "weight": int(sum(PRED[m][e] for m in ORDER if e in PRED[m]))})
cands.sort(key=lambda c: (-c["n_models"], -c["weight"]))
print(f"\n(C) hypothesis list at >={best_k} models ({B[best_k]['fold']}x enrichment, "
      f"precision {B[best_k]['precision']:.4f} vs null {B[best_k]['null_precision']:.5f}): "
      f"{len(cands)} pairs in NO database", flush=True)
for c in cands[:20]:
    print(f"   {c['pair'][0]:>12} — {c['pair'][1]:<12} {c['n_models']} models, weight {c['weight']}", flush=True)

json.dump({"design": "co-firing in SAE features predicts held-out TRRUST TF->target edges",
           "truth": "TRRUST, excluded from the calibrated annotator (no PPI, no regulatory DBs)",
           "null": "configuration model, each gene's co-firing degree preserved exactly",
           "top_genes_per_feature": TOPG, "min_features_per_pair": W0, "n_perm": NPERM,
           "per_model": A, "cross_model_curve": B, "best_k": best_k,
           "n_hypotheses": len(cands), "hypotheses": cands[:60]},
          open(f"{C}/hypothesis_trrust2.json", "w"), indent=1)
print("\n==> hypothesis_trrust2.json", flush=True)
