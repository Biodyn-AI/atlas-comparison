#!/usr/bin/env python
"""Untrained controls: the same analyses on catalogues from SAEs that were never trained (random-SAE)
and from models whose weights were never trained (untrained model), next to the real catalogues.

For each catalogue set and model:
  * middle layer (sorted catalogue files, index n // 2, as bb_common.mid_path): the strict annotator
    (backbone_composition/bb_common.Annotator, cap 200) -> matched gene sets; share of features with at
    least one match;
  * shared core: gene sets matched in >= 8 of the 10 models (random-SAE set only, all ten present);
  * TRRUST: pooled co-firing graph over all layers (top 10 genes, upper case, Ensembl ids dropped,
    pairs in >= 5 features; hypothesis_trrust2.py) scored with the configuration null of
    hypothesis_trrust2.py (stubs shuffled and re-paired), 200 draws.

usage: python controls.py SET ROOT [MODELS]   SET in {real, random_sae, untrained_model}
output: controls_<SET>.json (this folder)
"""
from __future__ import annotations
import glob, itertools, json, os, re, sys, time
from collections import Counter, defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "backbone_composition"))
from bb_common import Annotator, G  # noqa: E402

SET, ROOT = sys.argv[1], sys.argv[2]
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
MODELS = sys.argv[3].split(",") if len(sys.argv) > 3 else ORDER
NPERM, W0, TOPG = 200, 5, 10
t0 = time.time()


def cats(m):
    ps = glob.glob(f"{ROOT}/{m}/feature_catalog_L*.json")
    return sorted(ps, key=lambda p: int(re.search(r"_L(\d+)\.json$", p).group(1)))


TR = json.load(open(f"{G}/trrust_edges.json")); TRUTH = set()
for a, bs in TR.items():
    for b in bs:
        u, v = sorted((str(a).upper(), str(b).upper()))
        if u != v: TRUTH.add((u, v))


def trrust(m, seed):
    cnt = Counter()
    for p in cats(m):
        for f in json.load(open(p))["features"].values():
            s = sorted({str(x).upper() for x in f.get("top_genes", [])[:TOPG] if not str(x).upper().startswith("ENSG")})
            if len(s) >= 2:
                cnt.update(itertools.combinations(s, 2))
    E = [e for e, c in cnt.items() if c >= W0]
    if not E:
        return {"n_pairs": 0, "hits": 0}
    genes = sorted({g for e in E for g in e}); gid = {g: i for i, g in enumerate(genes)}
    a = np.array([gid[e[0]] for e in E], np.int64); b = np.array([gid[e[1]] for e in E], np.int64)
    tkey = np.array(sorted({gid[u] * (1 << 21) + gid[v] for u, v in TRUTH if u in gid and v in gid}), np.int64)
    keys = np.minimum(a, b) * (1 << 21) + np.maximum(a, b)
    obs = int(np.isin(keys, tkey).sum())
    rng = np.random.default_rng(seed); nul = []
    for _ in range(NPERM):
        st = np.concatenate([a, b]); rng.shuffle(st)
        h, t = st[::2], st[1::2]; ok = h != t
        k = np.unique(np.minimum(h[ok], t[ok]) * (1 << 21) + np.maximum(h[ok], t[ok]))
        nul.append(int(np.isin(k, tkey).sum()))
    nul = np.array(nul, float)
    return {"n_pairs": len(E), "hits": obs, "null_mean": round(float(nul.mean()), 3),
            "fold": round(obs / nul.mean(), 2) if nul.mean() > 0 else None,
            "p_emp": round((int((nul >= obs).sum()) + 1) / (NPERM + 1), 5)}


ann = Annotator(200)
res = {"set": SET, "root": {"real": "$ATLAS_TRAINED_CATALOGUES", "random_sae": "$ATLAS_CACHE/random_sae",
                          "untrained_model": "$ATLAS_CACHE/out_untrained"}[SET], "n_perm": NPERM, "models": {}}
sets = {}
for i, m in enumerate(MODELS):
    cs = cats(m)
    if not cs:
        print(f"{m}: no catalogues", flush=True); continue
    mid = cs[len(cs) // 2]
    feats = [f["top_genes"] for f in json.load(open(mid))["features"].values()]
    det = ann.annotate_detail(feats)
    labelled = {fi for v in det.values() for fi, _ in v}
    sets[m] = set(det)
    res["models"][m] = {"n_layers": len(cs), "mid_layer_file": os.path.basename(mid), "mid_features": len(feats),
                        "mid_labelled_share": round(len(labelled) / max(len(feats), 1), 4), "mid_gene_sets": len(det),
                        "trrust": trrust(m, 100 + i)}
    print(m, {k: v for k, v in res["models"][m].items() if k != "mid_layer_file"}, f"[{time.time() - t0:.0f}s]", flush=True)
    json.dump(res, open(os.path.join(HERE, f"controls_{SET}.json"), "w"), indent=1)
if len(sets) == 10:
    cm = Counter(t for s in sets.values() for t in s)
    res["shared_core"] = {f">={k}": sum(1 for v in cm.values() if v >= k) for k in (7, 8, 9, 10)}
    print("shared core", res["shared_core"], flush=True)
json.dump(res, open(os.path.join(HERE, f"controls_{SET}.json"), "w"), indent=1)
