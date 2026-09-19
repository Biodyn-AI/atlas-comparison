#!/usr/bin/env python
"""Ingest the prior atlases' per-layer feature files (layer_XX_features.json, schema:
list of {i,d,f,ma,fc,tg:[{n,a}...]}) into our feature_catalog schema, so his models
join the comparative matrix. Top genes come from `tg` (name field).

    python scripts/atlas_ingest.py --features /tmp/gf_L4.json --model Geneformer --layer 4
"""
import argparse, json, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out", default="outputs/atlas")
    args = ap.parse_args()
    feats_in = json.load(open(args.features))
    feats = {}
    for f in feats_in:
        if f.get("d"):        # dead
            continue
        tg = [str(g["n"]).upper() for g in f.get("tg", [])]
        if not tg:
            continue
        feats[str(f["i"])] = {"top_genes": tg, "max_activation": float(f.get("ma", 0)),
                              "activation_frequency": float(f.get("f", 0))}
    os.makedirs(f"{args.out}/{args.model}", exist_ok=True)
    cat = {"model": args.model, "layer": args.layer, "n_alive": len(feats),
           "n_genes": None, "features": feats, "source": "igor-published-top5"}
    p = f"{args.out}/{args.model}/feature_catalog_L{args.layer}.json"
    json.dump(cat, open(p, "w"))
    print(f"==> {args.model} L{args.layer}: {len(feats)} features -> {p}")


if __name__ == "__main__":
    main()
