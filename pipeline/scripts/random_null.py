#!/usr/bin/env python
"""THE decisive control the p-calibration (#6) demanded: since random genes annotate at ~the same per-feature
rate, is the UNIVERSAL CORE (intersection across models) also just what random genes give? Annotate all 10 models
(mid layer) with RANDOM background genes, build the random-gene core, and compare the REAL core to that null.
Also: mean pairwise Jaccard null, deduplicated-core null, and a held-out-style null (overlap of two independent
random-gene cores) to calibrate the 'reproducibility' claim.
    python scripts/random_null.py   -> outputs/atlas/comparative/random_null.json
"""
from __future__ import annotations
import json, glob, os
import numpy as np
from scipy.stats import hypergeom
from collections import defaultdict, Counter

BASE = "/Users/annaantipova/Desktop/biomech"
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"; TS = f"{BASE}/outputs/atlas/ts3_out"
MIN, MAX, ALPHA, TOP = 5, 500, 0.05, 5
NPERM = 100
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
rng = np.random.default_rng(0)

gs = {}
for nm in ("GO_BP", "KEGG", "Reactome"):
    for t, g in json.load(open(f"{G}/{nm}_gene_sets.json")).items():
        s = set(x.upper() for x in g)
        if MIN <= len(s) <= MAX:
            gs[f"{nm}:{t}"] = s
for nm, fn in (("TRRUST", "trrust_edges.json"), ("STRING", "string_edges.json")):
    for t, g in json.load(open(f"{G}/{fn}")).items():
        s = set(x.upper() for x in g)
        if MIN <= len(s) <= MAX:
            gs[f"{nm}:{t}"] = s
bg = set(x.upper() for x in json.load(open(f"{G}/background.json")))
bg |= set().union(*[v for k, v in gs.items() if k.startswith("STRING:")])
bg |= set(k.split(":", 1)[1].upper() for k in gs if k.startswith("STRING:"))
bg_list = np.array(sorted(bg)); bgn = len(bg)
idx = defaultdict(list); size = {}
GENES = {}
for t, genes in gs.items():
    size[t] = len(genes); GENES[t] = genes
    for g in genes:
        idx[g].append(t)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.arange(1, n + 1)
    q = np.minimum.accumulate((p[o] * n / r)[::-1])[::-1]; out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


def annotate(feat_lists):
    A = []; TG = []; LG = []; TERM = []
    for genes in feat_lists:
        g = set(str(x).upper() for x in genes[:TOP]) & bg
        if len(g) < 3:
            continue
        cand = Counter()
        for gene in g:
            for k in idx.get(gene, ()):
                cand[k] += 1
        for k, a in cand.items():
            if a < 2:
                continue
            A.append(a); TG.append(size[k]); LG.append(len(g)); TERM.append(k)
    if not A:
        return set()
    q = bh(hypergeom.sf(np.array(A) - 1, bgn, np.array(TG), np.array(LG)))
    return set(TERM[i] for i in np.where(q <= ALPHA)[0])


def orig_dir(m):
    d = f"{ALLCAT}/{m}"
    return d if glob.glob(f"{d}/feature_catalog_L*.json") else f"{TS}/{m}"


def dedup(terms, thr=0.5):
    terms = [t for t in terms if t in GENES]
    reps = []; g2rep = defaultdict(list)
    for t in sorted(terms, key=lambda x: -len(GENES[x])):
        gt = GENES[t]; assigned = False
        for ri in set(r for g in gt for r in g2rep[g]):
            if len(gt & reps[ri]) / len(gt | reps[ri]) >= thr:
                assigned = True; break
        if not assigned:
            reps.append(gt)
            for g in gt:
                g2rep[g].append(len(reps) - 1)
    return len(reps)


def mean_jac(sets):
    ms = list(sets); v = []
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            a, b = sets[ms[i]], sets[ms[j]]
            v.append(len(a & b) / max(len(a | b), 1))
    return float(np.mean(v))


# real (mid layer, all models)
nfeat = {}; real = {}
for m in ORDER:
    cats = sorted(glob.glob(f"{orig_dir(m)}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])
    cat = json.load(open(cats[len(cats) // 2]))
    feats = [f["top_genes"] for f in cat["features"].values()]
    nfeat[m] = len(feats)
    real[m] = annotate(feats)
real_core = set.intersection(*real.values())
real_jac = mean_jac(real)
print(f"REAL: core {len(real_core)} | mean Jaccard {real_jac:.3f} | per-model concepts {[len(real[m]) for m in ORDER]}", flush=True)

# random-gene null
rand_cores = []; rand_jacs = []; rand_dedups = []; two_draw_overlap = []
for it in range(NPERM):
    rsets = {m: annotate([rng.choice(bg_list, TOP, replace=False).tolist() for _ in range(nfeat[m])]) for m in ORDER}
    rc = set.intersection(*rsets.values())
    rand_cores.append(len(rc)); rand_jacs.append(mean_jac(rsets))
    if it < 20:
        rand_dedups.append(dedup(rc))
    # held-out-style: a SECOND independent random draw, overlap of the two random cores
    if it < 30:
        rsets2 = {m: annotate([rng.choice(bg_list, TOP, replace=False).tolist() for _ in range(nfeat[m])]) for m in ORDER}
        rc2 = set.intersection(*rsets2.values())
        two_draw_overlap.append(len(rc & rc2) / max(len(rc | rc2), 1))
    if (it + 1) % 20 == 0:
        print(f"  perm {it+1}/{NPERM}: rand core mean so far {np.mean(rand_cores):.0f}", flush=True)

rc = np.array(rand_cores); z = (len(real_core) - rc.mean()) / (rc.std() or 1)
real_dedup = dedup(real_core)
out = {
    "n_perm": NPERM, "layer": "mid", "models": ORDER,
    "real_core": len(real_core), "real_mean_jaccard": round(real_jac, 3), "real_dedup_core_j0.5": real_dedup,
    "rand_core_mean": round(float(rc.mean()), 1), "rand_core_sd": round(float(rc.std()), 1),
    "rand_core_max": int(rc.max()), "core_z": round(float(z), 1),
    "core_fold_over_null": round(len(real_core) / max(rc.mean(), 1), 2),
    "rand_jaccard_mean": round(float(np.mean(rand_jacs)), 3),
    "rand_dedup_core_mean": round(float(np.mean(rand_dedups)), 1) if rand_dedups else None,
    "two_random_draw_core_jaccard": round(float(np.mean(two_draw_overlap)), 3) if two_draw_overlap else None,
    "heldout_observed_core_jaccard": 0.518,
}
json.dump(out, open(f"{C}/random_null.json", "w"), indent=1)
print("\n=== RANDOM-GENE NULL (mid layer, 10 models) ===")
print(f"REAL core {out['real_core']}  vs  RANDOM-gene core {out['rand_core_mean']}±{out['rand_core_sd']} (max {out['rand_core_max']})")
print(f"  -> z={out['core_z']}, {out['core_fold_over_null']}x over null")
print(f"REAL mean-Jaccard {out['real_mean_jaccard']} vs random {out['rand_jaccard_mean']}")
print(f"REAL dedup core {out['real_dedup_core_j0.5']} vs random dedup {out['rand_dedup_core_mean']}")
print(f"held-out core Jaccard observed {out['heldout_observed_core_jaccard']} vs two-random-draw baseline {out['two_random_draw_core_jaccard']}")
print("==> random_null.json", flush=True)
