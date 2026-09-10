#!/usr/bin/env python
"""Can SAE features be used as a hypothesis generator? A falsifiable test on held-out interactions.

The calibrated annotator uses ONLY curated pathway sets (GO_BP / Reactome / KEGG, no PPI), so the
TRRUST database of literature-curated transcription-factor -> target regulatory edges is never seen
by any step of this atlas. It is therefore genuine held-out truth.

Prediction : two genes that repeatedly co-fire in the same SAE features are functionally linked.
             Per model, build the gene-gene co-firing graph over ALL layers (top-10 genes per
             feature, all pairs); a pair's weight is the number of features containing both.
Test       : do the predicted pairs recover TRRUST edges above chance?
Null       : the configuration model -- each gene keeps its EXACT co-firing degree, edges are
             rewired at random. A gene that co-fires with many partners (because it is abundant,
             or well studied) still does so in the null, so neither abundance nor study bias can
             produce the effect. Analytic expectation for the precision curve, permutations for p.
Output     : per-model enrichment (which models generate the best hypotheses) + for the winner,
             the top-weight pairs that are in NO database at all -- the actual hypothesis list,
             with a precision estimated from the held-out enrichment.
    python scripts/hypothesis_trrust.py [NPERM]  -> outputs/atlas/comparative/hypothesis_trrust.json
"""
from __future__ import annotations
import json, glob, sys, itertools
import numpy as np
from collections import defaultdict

BASE = "/Users/annaantipova/Desktop/biomech"
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"; TS = f"{BASE}/outputs/atlas/ts3_out"
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
TOPG = 10                       # genes per feature entering the co-firing graph
WGRID = (1, 2, 3, 5, 10)        # confidence levels: pair must co-fire in >= W features
WHEAD = 3                       # headline confidence level for the permutation test
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200
rng = np.random.default_rng(0)

# ---- held-out truth: TRRUST regulatory edges (never used by the annotator) --
TR = json.load(open(f"{G}/trrust_edges.json"))
TRUTH = set()
for a, bs in TR.items():
    for b in bs:
        u, v = sorted((str(a).upper(), str(b).upper()))
        if u != v: TRUTH.add((u, v))
print(f"held-out truth: {len(TRUTH)} TRRUST edges over {len({g for e in TRUTH for g in e})} genes\n", flush=True)

# ---- databases the annotator DID see, to exclude from the hypothesis list ---
SEEN = set()
for src in ("GO_BP", "KEGG", "Reactome"):
    for t, genes in json.load(open(f"{G}/{src}_gene_sets.json")).items():
        gl = sorted({str(x).upper() for x in genes})
        if len(gl) <= 200:
            SEEN.update(itertools.combinations(gl, 2))
STR = json.load(open(f"{G}/string_edges.json"))
STRING = set()
for a, bs in STR.items():
    for b in bs:
        u, v = sorted((str(a).upper(), str(b).upper()))
        if u != v: STRING.add((u, v))
print(f"excluded from the hypothesis list: {len(SEEN)} curated-pathway co-memberships, "
      f"{len(STRING)} STRING edges, {len(TRUTH)} TRRUST edges\n", flush=True)


def catalogs(m):
    d = ALLCAT if glob.glob(f"{ALLCAT}/{m}/feature_catalog_L*.json") else TS
    return sorted(glob.glob(f"{d}/{m}/feature_catalog_L*.json"))


def cofire(m):
    """distinct co-firing pairs with weights, as gene-index arrays"""
    voc, keys = {}, []
    def ix(g):
        i = voc.get(g)
        if i is None: i = voc[g] = len(voc)
        return i
    nfeat = 0
    for p in catalogs(m):
        for f in json.load(open(p))["features"].values():
            s = sorted({str(x).upper() for x in f.get("top_genes", [])[:TOPG]
                        if not str(x).upper().startswith("ENSG")})
            if len(s) < 2: continue
            nfeat += 1
            idx = sorted(ix(g) for g in s)
            for a, b in itertools.combinations(idx, 2): keys.append(a * (1 << 21) + b)
    inv = np.empty(len(voc), object)
    for g, i in voc.items(): inv[i] = g
    k, w = np.unique(np.asarray(keys, np.int64), return_counts=True)
    return inv, k >> 21, k & ((1 << 21) - 1), w, nfeat


def truth_mask(inv, u, v):
    gu = inv[u]; gv = inv[v]
    return np.fromiter(((a, b) in TRUTH if a < b else (b, a) in TRUTH for a, b in zip(gu, gv)),
                       bool, len(u))


def analytic_expected(u, v, deg, m_edges, is_truth_pairs):
    """configuration-model expectation of how many TRUTH edges land in a graph with these degrees"""
    # E[edge (a,b) present] ~ 1 - exp(-d_a d_b / (2m));  summed over TRUTH edges within the universe
    return float(np.sum(1.0 - np.exp(-is_truth_pairs / (2.0 * m_edges))))


results = {}
for m in ORDER:
    inv, u, v, w, nfeat = cofire(m)
    NG = len(inv); gix = {g: i for i, g in enumerate(inv.tolist())}
    tmask = truth_mask(inv, u, v)
    # TRRUST edges whose BOTH genes exist in this model's graph -- the reachable universe
    reach = [(a, b) for a, b in TRUTH if a in gix and b in gix]
    print(f"{m}: {nfeat} features, {NG} genes, {len(u)} distinct co-firing pairs, "
          f"{len(reach)} TRRUST edges reachable", flush=True)
    per_w = {}
    for W in WGRID:
        sel = w >= W
        if sel.sum() == 0: continue
        us, vs = u[sel], v[sel]
        hits = int(tmask[sel].sum())
        deg = np.bincount(np.concatenate([us, vs]), minlength=NG).astype(np.float64)
        mE = len(us)
        dp = np.array([deg[gix[a]] * deg[gix[b]] for a, b in reach], float)
        exp = analytic_expected(us, vs, deg, mE, dp)
        per_w[W] = {"n_pairs": int(mE), "hits": hits, "expected_degree_matched": round(exp, 2),
                    "fold": round(hits / exp, 2) if exp > 0 else None,
                    "precision": round(hits / mE, 6) if mE else 0.0}
        print(f"    W>={W:<3} pairs {mE:>8}  TRRUST hits {hits:>4}  "
              f"degree-matched expected {exp:>7.2f}  fold {per_w[W]['fold']}", flush=True)
    results[m] = {"n_features": nfeat, "n_genes": NG, "n_reachable_truth": len(reach), "by_W": per_w}

json.dump({"truth": "TRRUST curated TF->target edges, excluded from the calibrated annotator",
           "n_truth_edges": len(TRUTH), "top_genes_per_feature": TOPG,
           "null": "configuration model (each gene keeps its exact co-firing degree)",
           "per_model": results}, open(f"{C}/hypothesis_trrust.json", "w"), indent=1)
print("\n==> hypothesis_trrust.json", flush=True)
