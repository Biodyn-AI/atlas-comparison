#!/usr/bin/env python
"""Does the co-firing prediction lean on near-duplicate features?

Section 3.4 calls a gene pair "predicted" when it recurs in >= 5 features of a model. A block of
near-duplicate features -- same top genes, firing together -- satisfies that threshold by itself,
so a model with such a block could look like a strong predictor for the wrong reason. The atlas
page shows that Tahoe-x1's co-activation communities are exactly such a block (tight in decoder
space, firing in ~10 % of cells against 0.3 % for the average feature).

For every model this script asks:
  * how much more often community features fire than the rest (from the catalogues);
  * what share of each predicted pair's supporting features are community members, and how many
    predicted pairs are carried mostly (>= 50 %) by them;
  * whether those block-carried pairs hit TRRUST at a different rate from the rest;
  * what survives if support is DEDUPLICATED: features in the same community at the same layer
    count once, so a block of clones can contribute one vote per layer, not thirty.
Pair construction matches hypothesis_trrust2.py exactly (top-10 genes, Ensembl ids dropped,
undirected, W >= 5). Communities come from modules_alllayers.json (atlas_h100/modules_alllayers.py).
    python scripts/pair_redundancy.py -> outputs/atlas/comparative/pair_redundancy.json
"""
from __future__ import annotations

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import glob, itertools, json
import numpy as np
from collections import defaultdict

C = f"{_B}/outputs/atlas/comparative"; G = f"{_B}/outputs/atlas/genesets"
ALLCAT = f"{_B}/outputs/atlas/alllayer_cat"; TS = f"{_B}/outputs/atlas/ts3_out"
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
TOPG, W0 = 10, 5

MODS = json.load(open(f"{C}/modules_alllayers.json"))["graphs"]
TRUTH = set()
for a, bs in json.load(open(f"{G}/trrust_edges.json")).items():
    for b in bs:
        u, v = sorted((a.upper(), b.upper()))
        if u != v:
            TRUTH.add((u, v))


def catalogs(m):
    d = ALLCAT if glob.glob(f"{ALLCAT}/{m}/feature_catalog_L*.json") else TS
    return sorted(glob.glob(f"{d}/{m}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])


out = {"design": "pairs as in hypothesis_trrust2 (top-10 genes, W>=5); communities from modules_alllayers.json",
       "models": {}}
for m in ORDER:
    support = defaultdict(list)          # pair -> [(layer, fid, in_community, community_id)]
    fire_in, fire_out = [], []
    for p in catalogs(m):
        cat = json.load(open(p)); L = str(cat["layer"])
        g = MODS.get(m, {}).get(L, {})
        member = {n["id"]: n["m"] for n in g.get("nodes", [])}
        for fid, f in cat["features"].items():
            fr = float(f.get("activation_frequency", 0))
            (fire_in if int(fid) in member else fire_out).append(fr)
            s = sorted({str(x).upper() for x in f.get("top_genes", [])[:TOPG] if not str(x).upper().startswith("ENSG")})
            if len(s) < 2:
                continue
            for a, b in itertools.combinations(s, 2):
                support[(a, b)].append((L, int(fid), int(fid) in member, member.get(int(fid), -1)))
    pred = {k: v for k, v in support.items() if len(v) >= W0}
    # dedup: one vote per (layer, community); non-members keep their own vote
    dedup = {}
    for k, v in support.items():
        votes = {(L, ("c", cid) if inc else ("f", fid)) for L, fid, inc, cid in v}
        if len(votes) >= W0:
            dedup[k] = len(votes)
    frac = np.array([np.mean([x[2] for x in v]) for v in pred.values()]) if pred else np.array([])
    block = [k for k, v in pred.items() if np.mean([x[2] for x in v]) >= 0.5]
    rest = [k for k in pred if k not in set(block)]
    hit = lambda ks: int(sum(1 for k in ks if k in TRUTH))
    r = {"n_features": len(fire_in) + len(fire_out), "n_community_features": len(fire_in),
         "fire_pct_community": round(100 * float(np.mean(fire_in)), 2) if fire_in else None,
         "fire_pct_rest": round(100 * float(np.mean(fire_out)), 2) if fire_out else None,
         "n_pred": len(pred),
         "mean_share_of_support_from_communities": round(float(frac.mean()), 3) if len(frac) else None,
         "n_pred_block_carried": len(block), "pct_pred_block_carried": round(100 * len(block) / max(len(pred), 1), 1),
         "hits_block_carried": hit(block), "hits_rest": hit(rest),
         "precision_block_carried": round(hit(block) / max(len(block), 1), 5),
         "precision_rest": round(hit(rest) / max(len(rest), 1), 5),
         "n_pred_dedup": len(dedup), "hits_dedup": hit(dedup),
         "precision_full": round(hit(pred) / max(len(pred), 1), 5),
         "precision_dedup": round(hit(dedup) / max(len(dedup), 1), 5)}
    out["models"][m] = r
    print(f"{m:<13} feat {r['n_features']:>6} (community {r['n_community_features']:>5}) fire {r['fire_pct_community']}% vs {r['fire_pct_rest']}% | "
          f"pred {r['n_pred']:>7} block-carried {r['pct_pred_block_carried']:>5}% (prec {r['precision_block_carried']} vs rest {r['precision_rest']}) | "
          f"dedup {r['n_pred_dedup']:>7} hits {r['hits_dedup']} prec {r['precision_dedup']} (full {r['precision_full']})", flush=True)
json.dump(out, open(f"{C}/pair_redundancy.json", "w"), indent=1)
print("==> pair_redundancy.json")
