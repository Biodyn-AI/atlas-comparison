#!/usr/bin/env python
"""Cross-layer feature flow over ALL ADJACENT layers (methodologically better than depth-matched, which
skips transformer blocks). persistence(L->L+1) = % of features in L with a match in L+1 at top-5 gene-set
Jaccard > 0.3. Inverted gene->feature index so it's fast on wide models. Reads the all-layer catalogs and
emits a tiny flow_alllayers.json (per model: list of {from,to,persistence} for every consecutive layer).
    python flow_alllayers.py --out out_alllayers
"""
from __future__ import annotations
import argparse, json, glob
from collections import defaultdict

MODELS = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
THR, NG = 0.3, 5


def top_sets(cat):
    return {fid: set(str(x).upper() for x in f["top_genes"][:NG]) for fid, f in cat["features"].items()}


def persistence(A, B):
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
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="out_alllayers"); args = ap.parse_args()
    flow = {}
    for m in MODELS:
        cats = sorted(glob.glob(f"{args.out}/{m}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])
        if len(cats) < 2:
            continue
        cs = [json.load(open(p)) for p in cats]
        lay = [c["layer"] for c in cs]
        flow[m] = [{"from": lay[i], "to": lay[i + 1], "persistence": persistence(cs[i], cs[i + 1])}
                   for i in range(len(cs) - 1)]
        print(f"  {m}: {len(lay)} layers, persistence[min..max]="
              f"{min(t['persistence'] for t in flow[m])}..{max(t['persistence'] for t in flow[m])}", flush=True)
    flow = {m: flow[m] for m in MODELS if m in flow}
    json.dump(flow, open(f"{args.out}/flow_alllayers.json", "w"))
    print(f"==> {args.out}/flow_alllayers.json ({len(flow)} models)")


if __name__ == "__main__":
    main()
