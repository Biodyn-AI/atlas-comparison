#!/usr/bin/env python
"""Cross-layer feature flow (Igor's persistence view). For each consecutive pair of the 5 depth-matched
layers, persistence = % of features in the earlier layer that have a match in the next layer with top-5
gene-set Jaccard > 0.3. Reconstructs the atlas 'flow' block for all models (adds Tahoe). Reproduces the
prior 7-model values (AIDO exact) from the current ts3_out catalogs.
    python scripts/flow_ts3.py
"""
from __future__ import annotations
import json, glob
from collections import defaultdict

BASE = "/Users/annaantipova/Desktop/biomech"
TS = f"{BASE}/outputs/atlas/ts3_out"; C = f"{BASE}/outputs/atlas/comparative"
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
THR, NG = 0.3, 5


def top_sets(cat):
    return {fid: set(str(x).upper() for x in f["top_genes"][:NG]) for fid, f in cat["features"].items()}


def persistence(A, B):
    """Inverted gene->feature index on layer B; only Jaccard-test B-features sharing >=1 gene."""
    a = top_sets(A); bsets = {fid: g for fid, g in top_sets(B).items() if g}
    if not a:
        return 0.0
    inv = defaultdict(set)
    for fid, g in bsets.items():
        for gene in g:
            inv[gene].add(fid)
    hit = 0
    for ga in a.values():
        if not ga:
            continue
        cand = set()
        for gene in ga:
            cand |= inv.get(gene, set())
        for fid in cand:
            gb = bsets[fid]
            if len(ga & gb) / len(ga | gb) > THR:
                hit += 1
                break
    return round(100 * hit / len(a), 1)


def main():
    flow = {}
    for m in ORDER:
        cats = sorted(glob.glob(f"{TS}/{m}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])
        if not cats:
            continue
        if len(cats) > 6:  # depth-match to 5 layers (0/25/50/75/100%) for parity across models
            idxs = sorted(set(round(x * (len(cats) - 1) / 4) for x in range(5)))
            cats = [cats[i] for i in idxs]
        cs = [json.load(open(p)) for p in cats]
        lay = [c["layer"] for c in cs]
        flow[m] = [{"from": lay[i], "to": lay[i + 1], "persistence": persistence(cs[i], cs[i + 1])}
                   for i in range(len(cs) - 1)]
        print(f"  {m}: layers={lay} persistence={[t['persistence'] for t in flow[m]]}", flush=True)
    flow = {m: flow[m] for m in ORDER if m in flow}
    json.dump(flow, open(f"{C}/flow_ts3.json", "w"))
    print(f"==> flow_ts3.json ({len(flow)} models)")


if __name__ == "__main__":
    main()
