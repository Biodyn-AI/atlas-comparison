#!/usr/bin/env python
"""Per-layer profile for ALL layers (annotation rate, semantic richness, SAE variance, dead fraction) —
turns the atlas depth chart from 5 depth-% snapshots into a full curve over every layer. Reads the
all-layers catalogs (n_alive, fvu, dead_frac) + annotations. Tiny output.
    python depth_profile.py --all --out out_alllayers   -> out_alllayers/depth_alllayers.json
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np

MODELS = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]


def load_ann(p):
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    return d.get("annotations", d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model"); ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="out_alllayers")
    args = ap.parse_args()
    targets = MODELS if args.all else [args.model]
    res = {}
    for m in targets:
        cats = sorted(glob.glob(f"{args.out}/{m}/feature_catalog_L*.json"),
                      key=lambda p: int(re.search(r"L(\d+)", p).group(1)))
        prof = []
        for cp in cats:
            cat = json.load(open(cp)); L = int(cat.get("layer", re.search(r"L(\d+)", cp).group(1)))
            n_alive = int(cat.get("n_alive", len(cat.get("features", {}))))
            ann = load_ann(f"{args.out}/annotations/{m}_L{L:02d}_annotations.json")
            n_annot = sum(1 for v in ann.values() if v)
            tot = sum(len(v) for v in ann.values())
            prof.append({"layer": L,
                         "rate": round(100 * n_annot / max(n_alive, 1), 1),
                         "rich": round(tot / max(n_annot, 1), 2),
                         "var_explained": round(1 - float(cat.get("fvu", 0)), 3),
                         "dead": round(float(cat.get("dead_frac", 0)), 3),
                         "n_alive": n_alive, "n_annot": n_annot})
        res[m] = prof
        print(f"{m}: {len(prof)} layers, rate {prof[0]['rate']}->{prof[-1]['rate']}%", flush=True)
    p = os.path.join(args.out, "depth_alllayers.json"); json.dump(res, open(p, "w"))
    print(f"==> {p}", flush=True)


if __name__ == "__main__":
    main()
