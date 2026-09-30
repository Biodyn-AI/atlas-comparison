#!/usr/bin/env python
"""Step 2b (job item 1, classification): which recovered TRRUST edges join two genes of the same
protein / DNA-binding-domain family, or two genes with similar names?

Two checks on every hit in s02_edges.json:
 (a) NAME similarity: length of the shared leading string of the two symbols (e.g. SP1/SPP1 -> 2,
     STAT1/STAT3 -> 4). Reported as >= 2 and >= 3 shared leading characters, plus the repo's
     corrected same-family rule (hypothesis_robust.py:35-49).
 (b) STRUCTURAL family: a HAND-MADE map of the genes that appear in hits to their DNA-binding-domain
     or protein family, from standard TF classifications (general knowledge, not a database file in
     this repo; approximate; genes not in the map are 'other'). Same family = both genes in the same
     group. This catches pairs whose names differ but whose proteins are paralogues (FLI1/ERG,
     RORA/NR1D1, ESR1/AR), which matters most for UCE, whose input is protein sequence.
Output: OUT/s02b_family.json
"""
import json, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

FAMILY = {
    "bZIP": "ATF2 ATF3 ATF4 CREB1 CREM DDIT3 FOSL1 FOSL2 JDP2 JUN JUNB XBP1 CEBPE CEBPG",
    "ETS": "ERG ETS1 ETS2 ETV6 ETV7 FLI1 SPIB SPI1 ELF3",
    "nuclear receptor": "AR ESR1 ESRRA HNF4A NR1D1 NR2F6 NR5A2 PPARG RORA RXRA NR3C1 PGR",
    "STAT": "STAT1 STAT2 STAT3 STAT4 STAT6",
    "IRF": "IRF1 IRF2 IRF3 IRF4 IRF5 IRF7",
    "bHLH-PAS": "HIF1A HIF3A EPAS1",
    "bHLH / bHLH-ZIP": "MYC MITF SREBF2",
    "forkhead": "FOXA2 FOXC1 FOXO1",
    "RUNT": "RUNX1 RUNX3",
    "C2H2 zinc finger": "SP1 SP3 KLF2 KLF4 KLF6 KLF10 EGR1 ZBTB7A BCL6 YY1 WT1 ZEB2",
    "Rel / NF-kB": "NFKB1 REL RELA",
    "IkB / NF-kB inhibitor": "NFKBIA NFKBIZ",
    "SMAD": "SMAD3 SMAD4 SMAD7",
    "homeobox": "HOXA10 HOXC9 HOXB8 MEIS1 PBX2 NKX2-1 EMX2 HNF1A HNF1B POU5F1",
    "GATA": "GATA2 GATA3",
    "HDAC": "HDAC1 HDAC3 HDAC4",
    "CBP/p300": "CREBBP EP300",
    "BET bromodomain": "BRD2 BRD3 BRD7",
    "E2F": "E2F1",
    "HSF": "HSF2",
    "p53": "TP53",
    "RB": "RB1",
    "MHC class II": "HLA-DRA HLA-DMA HLA-DMB HLA-DOA HLA-DQB1 HLA-DRB1 HLA-DPB1",
}
G2F = {g: f for f, gs in FAMILY.items() for g in gs.split()}

d = json.load(open(f"{OUT}/s02_edges.json"))


def lead(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def classify(h):
    a, b = h["pair"]
    fa, fb = G2F.get(a), G2F.get(b)
    struct = fa is not None and fa == fb
    return {"shared_leading_chars": lead(a, b), "struct_family_a": fa, "struct_family_b": fb,
            "same_structural_family": struct}


out = {"family_map": FAMILY, "per_model": {}, "tier2": {}}
for m, r in d["per_model"].items():
    rows = []
    for h in r["hits"]:
        c = classify(h)
        rows.append({"pair": h["pair"], "weight": h["weight"], "class": h["class"], **c})
    n = len(rows)
    samef = [x for x in rows if x["same_structural_family"]]
    name2 = [x for x in rows if x["shared_leading_chars"] >= 2]
    name3 = [x for x in rows if x["shared_leading_chars"] >= 3]
    either = [x for x in rows if x["same_structural_family"] or x["shared_leading_chars"] >= 2]
    out["per_model"][m] = {"n_hits": n, "n_same_structural_family": len(samef),
                           "n_shared_prefix_ge2": len(name2), "n_shared_prefix_ge3": len(name3),
                           "n_same_family_or_prefix_ge2": len(either),
                           "share_same_family_or_prefix_ge2": round(len(either) / n, 3) if n else None,
                           "family_counts": dict(Counter(x["struct_family_a"] for x in samef)),
                           "rows": rows}
    print(m, n, "struct", len(samef), "pref>=2", len(name2), "pref>=3", len(name3), "either", len(either),
          dict(Counter(x["struct_family_a"] for x in samef)))
rows = []
for h in d["tier2_all_ten"]["edges"]:
    rows.append({"pair": h["pair"], "n_models": h["n_models"], "class": h["class"], **classify(h)})
either = [x for x in rows if x["same_structural_family"] or x["shared_leading_chars"] >= 2]
out["tier2"] = {"n": len(rows), "n_same_structural_family": sum(x["same_structural_family"] for x in rows),
                "n_shared_prefix_ge2": sum(x["shared_leading_chars"] >= 2 for x in rows),
                "n_same_family_or_prefix_ge2": len(either), "rows": rows}
print("tier2", out["tier2"]["n"], out["tier2"]["n_same_structural_family"], out["tier2"]["n_shared_prefix_ge2"], len(either))
json.dump(out, open(f"{OUT}/s02b_family.json", "w"), indent=1)
