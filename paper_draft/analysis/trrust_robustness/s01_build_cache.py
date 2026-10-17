#!/usr/bin/env python
"""Step 1: read every per-layer catalogue once and store, per model, each feature's filtered top-10
genes as integer ids (int32, padded with -1) plus the layer index. Same filter as
hypothesis_trrust2.py:52-54. Then check that the pooled W>=5 graph built from this cache is exactly
the repo's graph (.cofire_cache.pkl written by hypothesis_trrust3.py) for all ten models.

Output: SCRATCH/genes_<m>.npz, SCRATCH/gene_index.json, OUT/s01_cache_check.json
"""
import json, os, pickle, re, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

os.makedirs(SCRATCH, exist_ok=True)
gid = {}
info = {}
t0 = time.time()
for m in ORDER:
    feats, lays, lnames = [], [], []
    files = catalog_files(m)
    for li, p in enumerate(files):
        lnames.append(int(re.search(r"_L(\d+)\.json$", p).group(1)))
        for f in json.load(open(p))["features"].values():
            s = sorted({str(x).upper() for x in f.get("top_genes", [])[:TOPG]
                        if not str(x).upper().startswith("ENSG")})
            if len(s) < 2:
                continue
            row = [gid.setdefault(g, len(gid)) for g in s]
            feats.append(row + [-1] * (TOPG - len(row)))
            lays.append(li)
    feat = np.array(feats, np.int32); lay = np.array(lays, np.int16)
    np.savez_compressed(cache_path(m), feat=feat, layer=lay, n_layers=len(files),
                        layer_names=np.array(lnames))
    info[m] = {"n_layers": len(files), "layer_names": lnames, "n_features_ge2": int(len(feat)),
               "source_dir": os.path.dirname(files[0]).replace(BASE, "<ATLAS_BASE>")}
    print(f"{m}: {len(files)} layers, {len(feat)} features, {time.time()-t0:.0f}s", flush=True)
    del feats, lays, feat, lay

genes = [None] * len(gid)
for g, i in gid.items():
    genes[i] = g
json.dump(genes, open(f"{SCRATCH}/gene_index.json", "w"))

# ---- check against the repo's own W>=5 graphs
PRED, NFEAT = pickle.load(open(f"{C}/.cofire_cache.pkl", "rb"))
pub = json.load(open(f"{ORIG}/outputs/atlas/comparative/hypothesis_trrust2.json"))["per_model"]
truth, _, _ = load_truth()
tkey = truth_keys(truth, gid)
chk = {}
for m in ORDER:
    feat, lay, L = load_model(m)
    keys, w = graph_from(feat, lay, None, 5)
    ref = np.array(sorted(min(gid[a], gid[b]) * SHIFT + max(gid[a], gid[b]) for a, b in PRED[m]), np.int64)
    refw = np.array([PRED[m][(genes[k // SHIFT], genes[k % SHIFT]) if genes[k // SHIFT] < genes[k % SHIFT]
                      else (genes[k % SHIFT], genes[k // SHIFT])] for k in ref])
    same = bool(len(ref) == len(keys) and np.array_equal(ref, keys) and np.array_equal(refw, w))
    hits = int(np.isin(keys, tkey).sum())
    chk[m] = {"n_features": int(len(feat)), "repo_n_features": int(NFEAT[m]), "n_pairs": int(len(keys)),
              "repo_n_pairs": len(PRED[m]), "identical_graph_and_weights": same, "hits": hits,
              "published_hits": pub.get(m, {}).get("hits"), "published_n_pairs": pub.get(m, {}).get("n_pairs")}
    print(m, chk[m], flush=True)
json.dump({"models": info, "check_vs_repo_W5_graph": chk, "n_genes_indexed": len(genes),
           "n_trrust_undirected": len(truth)},
          open(f"{OUT}/s01_cache_check.json", "w"), indent=1)
print("done", time.time() - t0)
