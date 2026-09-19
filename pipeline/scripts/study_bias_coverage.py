#!/usr/bin/env python
"""How much of what the models feature can any external resource actually check?

The Discussion argues that the failure to find new biology is a limit of the evidence rather than of
the models, and rests that on eight numbers: how many understudied genes the models feature, how many
survive our own confidence filter, and how few of them appear in a genome-scale screen. Those numbers
were computed ad hoc and never written down, so nothing could verify them and no reader could
reproduce them. This script computes and stores them.

Gene symbols are resolved through NCBI synonyms, because a renamed symbol read as "zero publications"
is what put H2AX and STING1 in the least-studied stratum once already.
    python scripts/study_bias_coverage.py -> outputs/atlas/comparative/study_bias_coverage.json
"""
from __future__ import annotations

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere

import glob, gzip, json
import numpy as np
from collections import defaultdict, Counter

C = f"{_B}/outputs/atlas/comparative"; PB = f"{_B}/outputs/atlas/pubmed"; PD = f"{_B}/outputs/atlas/perturb"
ALLCAT, TS = f"{_B}/outputs/atlas/alllayer_cat", f"{_B}/outputs/atlas/ts3_out"
MODELS = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
TOPG, FEW, MANY = 10, 5, 50          # "understudied" = < FEW papers; "well studied" = >= MANY

sym2id, syn2id, biotype = {}, {}, {}
with gzip.open(f"{PB}/Homo_sapiens.gene_info.gz", "rt") as fh:
    hdr = fh.readline().rstrip("\n").split("\t")
    ti = hdr.index("type_of_gene") if "type_of_gene" in hdr else 9
    for line in fh:
        f = line.rstrip("\n").split("\t")
        if f[0] != "9606":
            continue
        s = f[2].upper()
        sym2id.setdefault(s, f[1]); biotype.setdefault(s, f[ti])
        for sy in f[4].split("|"):
            sy = sy.strip().upper()
            if sy and sy != "-":
                syn2id.setdefault(sy, f[1])
cnt = defaultdict(int)
with gzip.open(f"{PB}/gene2pubmed.gz", "rt") as fh:
    fh.readline()
    for line in fh:
        t, g, _ = line.rstrip("\n").split("\t")
        if t == "9606":
            cnt[g] += 1


def papers(sym):
    gid = sym2id.get(sym) or syn2id.get(sym)
    return None if gid is None else cnt.get(gid, 0)


featured = set()
for m in MODELS:
    d = ALLCAT if glob.glob(f"{ALLCAT}/{m}/feature_catalog_L*.json") else TS
    for p in sorted(glob.glob(f"{d}/{m}/feature_catalog_L*.json")):
        for f in json.load(open(p))["features"].values():
            featured.update(str(x).upper() for x in f.get("top_genes", [])[:TOPG]
                            if not str(x).upper().startswith("ENSG"))
tested = {g for c in json.load(open(f"{C}/hypothesis_pairs_full.json"))["pairs"] for g in c["pair"]}

F = {g: papers(g) for g in featured}
Fm = {g: v for g, v in F.items() if v is not None}
Tm = {g: papers(g) for g in tested if papers(g) is not None}
under = {g for g, v in Fm.items() if v < FEW}
well = {g for g, v in Fm.items() if v >= MANY}
under_kept = {g for g, v in Tm.items() if v < FEW}
well_kept = {g for g, v in Tm.items() if v >= MANY}
pc_under = [g for g in under if biotype.get(g) == "protein-coding"]

f = __import__("h5py").File(f"{PD}/K562_gwps_normalized_bulk_01.h5ad", "r")
pert = {str(x.decode() if isinstance(x, bytes) else x).split("_")[1].upper() for x in f["obs/gene_transcript"][:]}
cats = [x.decode() if isinstance(x, bytes) else str(x) for x in f["var/__categories/gene_name"][:]]
meas = {cats[i].upper() for i in f["var/gene_name"][:]}
all_pc = [g for g in sym2id if biotype.get(g) == "protein-coding"]
pc_papers = np.array([cnt.get(sym2id[g], 0) for g in all_pc])
screen_pc = np.array([cnt.get(sym2id[g], 0) for g in pert if g in sym2id and biotype.get(g) == "protein-coding"])

out = {
    "definitions": {"understudied": f"< {FEW} publications in gene2pubmed", "well_studied": f">= {MANY}",
                    "top_genes_per_feature": TOPG, "mapping": "NCBI current symbols plus synonyms"},
    "featured_genes": len(Fm),
    "understudied_featured": len(under),
    "understudied_by_biotype": dict(Counter(biotype.get(g, "unmapped") for g in under).most_common()),
    # strictly ncRNA + pseudogene, which is what "non-coding or pseudogenes" means; small RNAs and
    # unmapped symbols are counted separately in the breakdown rather than folded in
    "understudied_noncoding_or_pseudo": sum(Counter(biotype.get(g, "unmapped") for g in under)[k]
                                            for k in ("ncRNA", "pseudo")),
    "understudied_small_rna": sum(Counter(biotype.get(g, "unmapped") for g in under)[k]
                                  for k in ("snRNA", "snoRNA", "scRNA")),
    "understudied_protein_coding": len(pc_under),
    "protein_coding_understudied_absent_from_screen": len(set(pc_under) - pert - meas),
    "retention": {"understudied_kept": len(under_kept), "understudied_total": len(under),
                  "understudied_pct": round(100 * len(under_kept) / len(under), 1),
                  "well_studied_kept": len(well_kept), "well_studied_total": len(well),
                  "well_studied_pct": round(100 * len(well_kept) / len(well), 1)},
    "screen_bias": {"all_protein_coding": len(all_pc),
                    "all_protein_coding_pct_understudied": round(100 * float((pc_papers < FEW).mean()), 1),
                    "screen_protein_coding": len(screen_pc),
                    "screen_pct_understudied": round(100 * float((screen_pc < FEW).mean()), 1)},
}
out["retention"]["ratio"] = round(out["retention"]["well_studied_pct"] / out["retention"]["understudied_pct"], 1)
json.dump(out, open(f"{C}/study_bias_coverage.json", "w"), indent=1)
r, sb = out["retention"], out["screen_bias"]
print(f"featured genes (mappable): {out['featured_genes']:,}")
print(f"  understudied (<{FEW} papers): {out['understudied_featured']:,}  "
      f"({out['understudied_protein_coding']} protein-coding, {out['understudied_noncoding_or_pseudo']} non-coding/pseudo)")
print(f"  of the protein-coding ones, absent from the screen: {out['protein_coding_understudied_absent_from_screen']}")
print(f"retention into the tested pairs: understudied {r['understudied_kept']}/{r['understudied_total']} "
      f"= {r['understudied_pct']} %, well studied {r['well_studied_pct']} %  ->  {r['ratio']}x harsher")
print(f"screen: {sb['screen_pct_understudied']} % understudied vs {sb['all_protein_coding_pct_understudied']} % "
      f"of all protein-coding genes")
print("==> study_bias_coverage.json")
