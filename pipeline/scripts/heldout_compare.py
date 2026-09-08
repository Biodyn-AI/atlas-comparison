#!/usr/bin/env python
"""Held-out replication (reviewer control R4): re-extracted all models on an INDEPENDENT 6000-cell sample
(seed-1 Tabula Sapiens, same 3 tissues) at the 5 depth-matched layers. Annotate held-out catalogs with the
canonical annotator, build the held-out universal core, and compare to the ORIGINAL core computed on the SAME
models at the SAME 5 layers (apples-to-apples). Reports: core sizes, Jaccard, % of the original core reproduced,
and per-model concept-set Jaccard (does each model detect the same biology on new cells?).
    python scripts/heldout_compare.py   -> outputs/atlas/comparative/heldout.json
scGPT is excluded — its held-out extraction crashed on a torchtext ABI mismatch in the shared venv.
"""
from __future__ import annotations
import json, glob, os
import numpy as np
from scipy.stats import hypergeom
from collections import defaultdict, Counter

BASE = "/Users/annaantipova/Desktop/biomech"
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
HO = f"{BASE}/outputs/atlas/heldout_cat/out_heldout"
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"; TS = f"{BASE}/outputs/atlas/ts3_out"
MIN, MAX, ALPHA, TOP = 5, 500, 0.05, 5
MODELS = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]  # scGPT excluded (torchtext crash)

# ---- canonical gene sets / background (== alllayer_matrix) ----
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
bgn = len(bg)
idx = defaultdict(list); size = {}
for t, genes in gs.items():
    size[t] = len(genes)
    for g in genes:
        idx[g].append(t)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.arange(1, n + 1)
    q = np.minimum.accumulate((p[o] * n / r)[::-1])[::-1]; out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


def annotate(cat_path):
    cat = json.load(open(cat_path))
    A = []; TG = []; LG = []; TERM = []
    for fid, f in cat["features"].items():
        g = set(str(x).upper() for x in f["top_genes"][:TOP]) & bg
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


ho_sets = {}; or_sets = {}; per_model = {}
for m in MODELS:
    ho_cats = sorted(glob.glob(f"{HO}/{m}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])
    if not ho_cats:
        print(f"  !! {m}: no held-out catalogs"); continue
    layers = [json.load(open(p))["layer"] for p in ho_cats]
    hs = set()
    for cp in ho_cats:
        hs |= annotate(cp)
    ho_sets[m] = hs
    # original at the SAME layer numbers
    od = orig_dir(m); os_ = set(); nfound = 0
    for L in layers:
        p = f"{od}/feature_catalog_L{L:02d}.json"
        if os.path.exists(p):
            os_ |= annotate(p); nfound += 1
    or_sets[m] = os_
    jac = round(len(hs & os_) / max(len(hs | os_), 1), 3)
    per_model[m] = {"held_out_concepts": len(hs), "orig_concepts": len(os_),
                    "jaccard": jac, "layers": layers, "orig_layers_found": nfound}
    print(f"  {m}: HO {len(hs)} / orig {len(os_)} concepts · Jaccard {jac} (layers {layers})", flush=True)

ho_core = set.intersection(*ho_sets.values()) if len(ho_sets) == len(MODELS) else set()
or_core = set.intersection(*or_sets.values()) if len(or_sets) == len(MODELS) else set()
core_overlap = round(len(ho_core & or_core) / max(len(ho_core | or_core), 1), 3)
pct_repro = round(100 * len(ho_core & or_core) / max(len(or_core), 1), 1)
mean_jac = round(float(np.mean([per_model[m]["jaccard"] for m in per_model])), 3)

out = {"models": MODELS, "n_models": len(MODELS), "excluded": ["scGPT (torchtext ABI crash)"],
       "layers": "5 depth-matched (0/25/50/75/100%)",
       "held_out_core": len(ho_core), "orig_core_same_models_layers": len(or_core),
       "core_jaccard": core_overlap, "pct_orig_core_reproduced": pct_repro,
       "mean_per_model_jaccard": mean_jac, "per_model": per_model}
json.dump(out, open(f"{C}/heldout.json", "w"), indent=1)
print(f"\n=== HELD-OUT REPLICATION ({len(MODELS)} models, 5 layers) ===")
print(f"held-out core {len(ho_core)} | original core (same models/layers) {len(or_core)}")
print(f"core Jaccard {core_overlap} | % of original core reproduced on new cells: {pct_repro}%")
print(f"mean per-model concept-set Jaccard (orig vs held-out): {mean_jac}")
print("==> heldout.json")
