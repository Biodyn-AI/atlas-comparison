#!/usr/bin/env python
"""All-layer concept matrix — the same top-5 Fisher+BH annotator (5 DBs, matching matrix_ts3_string) run on
EVERY layer of every model, then UNIONED across layers. This is a model's full concept repertoire across depth,
not just the mid layer. Drives coverage / universality spectrum / universal core / blind spots / similarity when
present (atlas_assemble prefers matrix_alllayer.json over the mid matrix). Same schema as matrix_ts3_string.json.
    - term_count : union over layers, term -> total feature-hits across all layers (KEYS = all-layer concept set)
    - annot_rate : MEAN of per-layer rates (n_feat/n_annot back-solved so the ratio = mean rate; n_feat kept at
                   the per-layer d_sae so totals don't balloon)
    - layer      : mid layer (for the roster's reference column); axis/params/var_explained carried from mid matrix
    python scripts/alllayer_matrix.py   -> outputs/atlas/comparative/matrix_alllayer.json
"""
from __future__ import annotations
import json, glob, os
import numpy as np
from collections import defaultdict, Counter
from scipy.stats import fisher_exact

BASE = "/Users/annaantipova/Desktop/biomech"
G = f"{BASE}/outputs/atlas/genesets"; C = f"{BASE}/outputs/atlas/comparative"
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"; TS = f"{BASE}/outputs/atlas/ts3_out"
MIN, MAX, ALPHA, TOP = 5, 500, 0.05, 5
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]

print("loading gene sets (5 DBs)...", flush=True)
gs = {nm: {t: set(g) for t, g in json.load(open(f"{G}/{nm}_gene_sets.json")).items() if MIN <= len(g) <= MAX}
      for nm in ("GO_BP", "KEGG", "Reactome")}
tr = {t: set(v) for t, v in json.load(open(f"{G}/trrust_edges.json")).items() if MIN <= len(v) <= MAX}
st = {t: set(v) for t, v in json.load(open(f"{G}/string_edges.json")).items() if MIN <= len(v) <= MAX}
bg = set(json.load(open(f"{G}/background.json"))); bg |= set().union(*st.values()) | set(st); bgn = len(bg)
idx = defaultdict(list); size = {}
for s, d in list(gs.items()) + [("TRRUST", tr), ("STRING", st)]:
    for t, genes in d.items():
        k = (s, t); size[k] = len(genes)
        for g in genes:
            idx[g].append(k)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.arange(1, n + 1)
    q = np.minimum.accumulate((p[o] * n / r)[::-1])[::-1]; out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


def annotate_layer(feats):
    """feats: {fid: [top genes]} -> (n_annotated, Counter(term -> #features hitting it))."""
    recs = []
    for fid, genes in feats.items():
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
            tg = size[k]; b = len(g) - a; c = tg - a; dd = bgn - a - b - c
            _, p = fisher_exact([[a, b], [c, dd]], alternative="greater")
            recs.append((fid, f"{k[0]}:{k[1]}", float(p)))
    if not recs:
        return 0, Counter()
    q = bh([r[2] for r in recs]); ann = defaultdict(list)
    for i, (fid, term, p) in enumerate(recs):
        if q[i] <= ALPHA:
            ann[fid].append(term)
    tc = Counter(t for terms in ann.values() for t in terms)
    return len(ann), tc


def catalogs_for(m):
    d = f"{ALLCAT}/{m}"
    if not (os.path.isdir(d) and glob.glob(f"{d}/feature_catalog_L*.json")):
        d = f"{TS}/{m}"
    return sorted(glob.glob(f"{d}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])


mid = json.load(open(f"{C}/matrix_ts3_string.json"))   # carry axis/params/var_explained/layer for continuity
out = {}; perlayer = {}   # perlayer[model][layer] = sorted list of concepts (for the controls script)
for m in ORDER:
    cats = catalogs_for(m)
    if not cats:
        print(f"  (skip {m}: no catalogs)"); continue
    tc_all = Counter(); rates = []; nfeat_layers = []; perlayer[m] = {}
    for cp in cats:
        cat = json.load(open(cp)); feats = {fid: f["top_genes"] for fid, f in cat["features"].items()}
        nalive = cat.get("n_alive", len(feats)); L = cat["layer"]
        nann, tc = annotate_layer(feats)
        perlayer[m][str(L)] = {"concepts": sorted(tc), "n_alive": int(nalive), "n_annot": int(nann),
                               "d_sae": int(cat.get("d_sae", 0))}
        tc_all.update(tc); rates.append(100 * nann / max(nalive, 1)); nfeat_layers.append(nalive)
    mean_rate = float(np.mean(rates))
    mm = mid.get(m, {})
    nfeat = int(mm.get("n_feat") or nfeat_layers[len(nfeat_layers) // 2])   # per-layer d_sae, keeps totals sane
    out[m] = {
        "n_feat": nfeat, "n_annot": int(round(nfeat * mean_rate / 100)),
        "layer": mm.get("layer", json.load(open(cats[len(cats) // 2]))["layer"]),
        "axis": mm.get("axis", ""), "params": mm.get("params", "?"),
        "var_explained": mm.get("var_explained", 0),
        "n_layers": len(cats), "mean_layer_rate": round(mean_rate, 1),
        "term_count": dict(tc_all),
        "feat_terms": mm.get("feat_terms", {}),   # MID-layer feat->terms (firing/richness are per-feature, mid)
    }
    print(f"  {m}: {len(cats)} layers · all-layer concepts {len(tc_all)} (mid was {len(mm.get('term_count', {}))}) · mean rate {mean_rate:.1f}%", flush=True)

json.dump(out, open(f"{C}/matrix_alllayer.json", "w"))
json.dump(perlayer, open(f"{C}/perlayer_concepts.json", "w"))
print(f"==> matrix_alllayer.json + perlayer_concepts.json ({len(out)} models)", flush=True)
