#!/usr/bin/env python
"""Build the shared annotation vocabulary once (GO BP / KEGG / Reactome via Enrichr,
TRRUST from the local TSV). Writes data/genesets/{GO_BP,KEGG,Reactome}_gene_sets.json
(term -> [symbols]) and trrust_edges.json (TF -> [targets]). This is the vocabulary
EVERY model's features are annotated against — build it once, reuse for all.

    python data/build_genesets.py --trrust path/to/trrust_human.tsv --out data/genesets
"""
from __future__ import annotations
import argparse, json, os

ENRICHR = {"GO_BP": "GO_Biological_Process_2021",
           "KEGG": "KEGG_2021_Human",
           "Reactome": "Reactome_2022"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trrust", required=True)
    ap.add_argument("--out", default="data/genesets")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import gseapy as gp
    for name, lib in ENRICHR.items():
        d = gp.get_library(name=lib, organism="Human")   # {term: [genes]}
        d = {t: [g.upper() for g in gs] for t, gs in d.items()}
        json.dump(d, open(os.path.join(args.out, f"{name}_gene_sets.json"), "w"))
        print(f"  {name}: {len(d)} terms  ({lib})")

    import pandas as pd
    tr = pd.read_csv(args.trrust, sep="\t", header=None, names=["tf", "tg", "mode", "pmid"])
    edges = {}
    for tf, tg in zip(tr.tf.str.upper(), tr.tg.str.upper()):
        edges.setdefault(tf, set()).add(tg)
    edges = {tf: sorted(t) for tf, t in edges.items()}
    json.dump(edges, open(os.path.join(args.out, "trrust_edges.json"), "w"))
    print(f"  TRRUST: {len(edges)} TFs")

    # background = union of all annotated genes
    bg = set()
    for name in ENRICHR:
        for gs in json.load(open(os.path.join(args.out, f"{name}_gene_sets.json"))).values():
            bg |= set(gs)
    for t in edges.values():
        bg |= set(t)
    json.dump(sorted(bg), open(os.path.join(args.out, "background.json"), "w"))
    print(f"  background: {len(bg)} genes")


if __name__ == "__main__":
    main()
