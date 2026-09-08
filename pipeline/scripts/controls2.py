#!/usr/bin/env python
"""Round-2 reviewer controls, merged into controls.json.
  R1 ontology-redundancy: the core is inflated by nested/overlapping gene sets -> collapse core concepts by
     gene-set Jaccard (greedy) and report the deduplicated core (effective independent concepts).
  R5 vocabulary fairness: recompute the core/universe on concepts REACHABLE (>=2 set genes in a model's surfaced
     vocabulary) by ALL models, and correct each model's blind-spot count for concepts it simply cannot represent.
  R6 superposition null: '% novel to SVD' (cos<0.7) vs the null for RANDOM directions in the same dimension —
     random high-dim vectors are ~all 'novel' too, so the metric may be near-tautological.
  R7 correlation CIs: bootstrap 95% CI on the scaling correlation (params vs all-layer concepts) — n=10 is small.
    python scripts/controls2.py
"""
from __future__ import annotations
import json, glob, os, re
import numpy as np
from collections import defaultdict

BASE = "/Users/annaantipova/Desktop/biomech"
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"; TS = f"{BASE}/outputs/atlas/ts3_out"
MIN, MAX = 5, 500
rng = np.random.default_rng(0)

mat = json.load(open(f"{C}/matrix_alllayer.json"))
ctl = json.load(open(f"{C}/controls.json"))
atlas = json.load(open(f"{C}/atlas_full_notf.json"))
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
models = [m for m in ORDER if m in mat]; N = len(models)
cs = {m: set(mat[m]["term_count"]) for m in models}
cm = defaultdict(int)
for m in models:
    for t in cs[m]:
        cm[t] += 1
core_terms = [t for t, k in cm.items() if k == N]

# ---- full gene sets per term ----
GENES = {}
for nm in ("GO_BP", "KEGG", "Reactome"):
    for t, g in json.load(open(f"{G}/{nm}_gene_sets.json")).items():
        if MIN <= len(g) <= MAX:
            GENES[f"{nm}:{t}"] = set(x.upper() for x in g)
for nm, fn in (("TRRUST", "trrust_edges.json"), ("STRING", "string_edges.json")):
    for t, g in json.load(open(f"{G}/{fn}")).items():
        if MIN <= len(g) <= MAX:
            GENES[f"{nm}:{t}"] = set(x.upper() for x in g)

# ===== R1: dedup core by gene-set Jaccard (greedy representative clustering) =====
def dedup(terms, thr):
    terms = [t for t in terms if t in GENES]
    reps = []  # list of (rep_term, gene_set)
    g2rep = defaultdict(list)
    n_clusters = 0
    for t in sorted(terms, key=lambda x: -len(GENES[x])):  # big sets first as representatives
        gt = GENES[t]; assigned = False
        cand = set()
        for g in gt:
            cand.update(g2rep[g])
        for ri in cand:
            gr = reps[ri][1]
            if len(gt & gr) / len(gt | gr) >= thr:
                assigned = True; break
        if not assigned:
            reps.append((t, gt)); ri = len(reps) - 1
            for g in gt:
                g2rep[g].append(ri)
    return len(reps)

r1 = {"core_raw": len(core_terms),
      "core_dedup_j0.5": dedup(core_terms, 0.5),
      "core_dedup_j0.7": dedup(core_terms, 0.7)}
print(f"R1 dedup core: raw {r1['core_raw']} -> J0.7 {r1['core_dedup_j0.7']} -> J0.5 {r1['core_dedup_j0.5']}", flush=True)

# ===== R5: vocabulary-fair universe / blind spots =====
def vocab_genes(m):
    d = f"{ALLCAT}/{m}"
    if not glob.glob(f"{d}/feature_catalog_L*.json"):
        d = f"{TS}/{m}"
    gv = set()
    for cp in glob.glob(f"{d}/feature_catalog_L*.json"):
        for f in json.load(open(cp))["features"].values():
            gv.update(str(x).upper() for x in f["top_genes"])
    return gv
VG = {m: vocab_genes(m) for m in models}
reach = {m: set(t for t in GENES if len(GENES[t] & VG[m]) >= 2) for m in models}
fair_universe = set.intersection(*reach.values())            # concepts every model COULD represent
fair_core = [t for t in core_terms if t in fair_universe]
# corrected blind spots: concepts encoded by others but not m, minus those m cannot represent
blind_raw = {}; blind_real = {}
allc = set().union(*cs.values())
for m in models:
    missed = set(t for t in allc if t not in cs[m] and cm[t] >= 1)
    blind_raw[m] = len(missed)
    blind_real[m] = len(set(t for t in missed if t in reach[m]))   # only reachable-but-missed = real blind spot
r5 = {"reachable_universe": {m: len(reach[m]) for m in models},
      "fair_universe_all": len(fair_universe), "fair_core": len(fair_core),
      "blind_raw": blind_raw, "blind_real_reachable": blind_real,
      "pct_blind_that_is_vocab": {m: round(100 * (blind_raw[m] - blind_real[m]) / max(blind_raw[m], 1), 1) for m in models}}
print(f"R5 fair universe (reachable by all) {len(fair_universe)}; fair core {len(fair_core)} (raw core {len(core_terms)})", flush=True)

# ===== R6: superposition null (random directions in the same dimension) =====
def dmodel(m):
    d = f"{ALLCAT}/{m}"
    if not glob.glob(f"{d}/feature_catalog_L*.json"):
        d = f"{TS}/{m}"
    return json.load(open(sorted(glob.glob(f"{d}/feature_catalog_L*.json"))[0]))["d_model"]
def null_novel(d, k, n=5000, thr=0.7):
    X = rng.standard_normal((n, d))
    # random k-subspace = first k coords (rotationally symmetric); cos to subspace = ||first k|| / ||X||
    cos = np.sqrt((X[:, :k] ** 2).sum(1) / (X ** 2).sum(1))
    return round(float((cos < thr).mean()) * 100, 1)
svd = atlas.get("svd", {})
r6 = {"observed_pct_novel": {m: svd.get(m, {}).get("novel") for m in models if m in svd}, "null": {}}
for m in models:
    d = dmodel(m)
    r6["null"][m] = {"d_model": d, "null_novel_k50": null_novel(d, 50), "null_novel_k256": null_novel(d, min(256, d - 1))}
ex = models[0]
print(f"R6 superposition null: e.g. {ex} d={r6['null'][ex]['d_model']} observed novel {r6['observed_pct_novel'].get(ex)}% vs random-direction null {r6['null'][ex]['null_novel_k256']}% (k=256)", flush=True)

# ===== R7: bootstrap CI on the scaling correlation (params vs all-layer concepts) =====
def toM(p):
    if isinstance(p, dict): p = p.get("params_m") or p.get("M") or p.get("params") or p.get("n")
    if isinstance(p, (int, float)): return float(p)
    if isinstance(p, str):
        mm = re.search(r"([\d.]+)\s*([MB]?)", p)
        if mm: return float(mm.group(1)) * (1000 if mm.group(2) == "B" else 1)
    return None
params = atlas.get("params", {})
pv = np.array([np.log10(toM(params.get(m))) for m in models])
nc = np.array([len(cs[m]) for m in models])
r_obs = float(np.corrcoef(pv, nc)[0, 1])
boots = []
idx = np.arange(N)
for _ in range(5000):
    b = rng.choice(idx, N, replace=True)
    if len(set(b)) < 3: continue
    with np.errstate(all="ignore"):
        rr = np.corrcoef(pv[b], nc[b])[0, 1]
    if not np.isnan(rr): boots.append(rr)
lo, hi = np.percentile(boots, [2.5, 97.5])
r7 = {"scaling_r_params_concepts": round(r_obs, 3), "ci95": [round(float(lo), 3), round(float(hi), 3)], "n": N}
print(f"R7 scaling r = {r_obs:.3f}, bootstrap 95% CI [{lo:.2f}, {hi:.2f}] (n=10)", flush=True)

ctl["dedup"] = r1; ctl["vocab_fair"] = r5; ctl["superposition_null"] = r6; ctl["scaling_ci"] = r7
json.dump(ctl, open(f"{C}/controls.json", "w"), indent=1)
print("==> controls.json updated (R1/R5/R6/R7)", flush=True)
