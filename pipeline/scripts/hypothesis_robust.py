#!/usr/bin/env python
"""Robustness of the held-out TRRUST result (hypothesis_trrust*.py).

R1  drop same-family paralog pairs (ACTB-ACTG2, S100A11-S100A4, ...) -- is the enrichment carried
    by gene families rather than by regulation?
R2  keep only features that actually fire (activation_frequency above the per-model median) -- a
    feature whose top genes are barely expressed in this corpus fires rarely, so this removes the
    "near-zero-expression genes cluster together" artefact.
R3  raise the per-pair evidence bar from >=5 to >=10 features.
Same configuration-model null throughout.
    python scripts/hypothesis_robust.py [NPERM] -> outputs/atlas/comparative/hypothesis_robust.json
"""
from __future__ import annotations
import json, glob, sys, itertools, re
import numpy as np
from collections import defaultdict

BASE = "/Users/annaantipova/Desktop/biomech"
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"; TS = f"{BASE}/outputs/atlas/ts3_out"
PASSED = ["Tahoe", "scGPT", "UCE", "C2S", "Geneformer"]
TOPG = 10
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200
rng = np.random.default_rng(0)

TR = json.load(open(f"{G}/trrust_edges.json")); TRUTH = set()
for a, bs in TR.items():
    for b in bs:
        u, v = sorted((str(a).upper(), str(b).upper()))
        if u != v: TRUTH.add((u, v))

FAM = re.compile(r"^([A-Z]{2,}[0-9]*?)[0-9]*[A-Z]?$")
def fam(g):
    m = FAM.match(g); return m.group(1) if m else g
def same_family(a, b):
    fa, fb = fam(a), fam(b); return fa == fb and len(fa) >= 3


def graph(m, W, freq_filter):
    d = ALLCAT if glob.glob(f"{ALLCAT}/{m}/feature_catalog_L*.json") else TS
    files = sorted(glob.glob(f"{d}/{m}/feature_catalog_L*.json"))
    feats = []
    for p in files:
        for f in json.load(open(p))["features"].values():
            feats.append((f.get("activation_frequency", 0.0), f.get("top_genes", [])))
    if freq_filter:
        med = float(np.median([a for a, _ in feats]))
        feats = [(a, g) for a, g in feats if a > med]
    cnt = defaultdict(int)
    for _, genes in feats:
        s = sorted({str(x).upper() for x in genes[:TOPG] if not str(x).upper().startswith("ENSG")})
        if len(s) < 2: continue
        for a, b in itertools.combinations(s, 2): cnt[(a, b)] += 1
    return {k: c for k, c in cnt.items() if c >= W}, len(feats)


def evaluate(PRED, label, drop_family):
    genes = sorted({g for m in PASSED for e in PRED[m] for g in e})
    gid = {g: i for i, g in enumerate(genes)}
    tkey = np.array(sorted({(lambda i, j: i * (1 << 21) + j)(*sorted((gid[a], gid[b])))
                            for a, b in TRUTH if a in gid and b in gid}), np.int64)

    def shuf(edges):
        a = np.fromiter((gid[e[0]] for e in edges), np.int32, len(edges))
        b = np.fromiter((gid[e[1]] for e in edges), np.int32, len(edges))
        s = np.concatenate([a, b]); rng.shuffle(s)
        h, t = s[::2], s[1::2]; ok = h != t; h, t = h[ok], t[ok]
        lo = np.minimum(h, t).astype(np.int64); hi = np.maximum(h, t).astype(np.int64)
        return np.unique(lo * (1 << 21) + hi)

    allp = defaultdict(int)
    for m in PASSED:
        for e in PRED[m]:
            if drop_family and same_family(*e): continue
            allp[e] += 1
    if not allp: return None
    keys = np.empty(len(allp), np.int64)
    for n, (a, b) in enumerate(allp):
        i, j = sorted((gid[a], gid[b])); keys[n] = i * (1 << 21) + j
    nmod = np.fromiter(allp.values(), np.int32, len(allp)); ist = np.isin(keys, tkey)

    nulls = {k: [] for k in (1, 2)}
    for _ in range(NPERM):
        cnt = defaultdict(int)
        for m in PASSED:
            E = [e for e in PRED[m] if not (drop_family and same_family(*e))]
            if E:
                for kk in shuf(E): cnt[int(kk)] += 1
        kk = np.fromiter(cnt.keys(), np.int64, len(cnt)); vv = np.fromiter(cnt.values(), np.int32, len(cnt))
        it_ = np.isin(kk, tkey)
        for k in (1, 2):
            sel = vv >= k; nulls[k].append(float(it_[sel].sum()) / max(int(sel.sum()), 1))
    out = {}
    for k in (1, 2):
        sel = nmod >= k; n = int(sel.sum()); h = int(ist[sel].sum())
        nc = np.array(nulls[k], float); prec = h / max(n, 1)
        out[k] = {"n_pairs": n, "hits": h, "precision": round(prec, 6),
                  "null_precision": round(float(nc.mean()), 8),
                  "fold": round(prec / nc.mean(), 2) if nc.mean() > 0 else None,
                  "p_emp": round((int((nc >= prec).sum()) + 1) / (NPERM + 1), 5)}
        f = out[k]["fold"]
        print(f"  {label:<34} k>={k}: {n:>7} pairs, {h:>4} hits, "
              f"{prec:.5f} vs {nc.mean():.7f}  ->  {f if f else 'inf'}x  p={out[k]['p_emp']}", flush=True)
    return out


RES = {}
for label, W, ff, df in (("R0 baseline (W>=5)", 5, False, False),
                         ("R1 no same-family pairs", 5, False, True),
                         ("R2 firing features only", 5, True, False),
                         ("R3 stricter evidence (W>=10)", 10, False, False),
                         ("R2+R1 firing, no families", 5, True, True)):
    PRED = {}
    for m in PASSED:
        PRED[m], nf = graph(m, W, ff)
    RES[label] = evaluate(PRED, label, df)

json.dump({"null": "configuration model, exact degree preserved", "n_perm": NPERM,
           "models": PASSED, "variants": RES}, open(f"{C}/hypothesis_robust.json", "w"), indent=1)
print("\n==> hypothesis_robust.json", flush=True)
