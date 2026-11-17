#!/usr/bin/env python
"""Step 8: print markdown tables for the notes from the JSON outputs (no new computation, except
the hand grouping of the 31 tier-2 edges, which is my reading and is marked as such)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

e = json.load(open(f"{OUT}/s02_edges.json"))
fb = json.load(open(f"{OUT}/s02b_family.json"))
ABBR = {"Tahoe": "Tah", "scGPT": "scG", "UCE": "UCE", "C2S": "C2S", "Geneformer": "GF", "MaxToki": "MxT", "tGPT": "tGPT",
        "AIDO": "AIDO", "scFoundation": "scF", "GeneCompass": "GC"}
# hand grouping (my reading, general knowledge)
GROUP = {
    ("ATF3", "JUN"): "B immediate-early", ("ATF3", "EGR1"): "B immediate-early", ("ATF3", "KLF6"): "B immediate-early",
    ("STAT1", "STAT2"): "C TF family", ("STAT1", "STAT3"): "C TF family", ("STAT1", "STAT6"): "C TF family",
    ("IRF1", "IRF2"): "C TF family", ("IRF3", "IRF4"): "C TF family", ("FOSL1", "FOSL2"): "C TF family",
    ("RUNX1", "RUNX3"): "C TF family", ("ETS1", "FLI1"): "C TF family", ("CREBBP", "EP300"): "C paralogue co-activators",
}
struct = {tuple(r["pair"]): r for r in fb["tier2"]["rows"]}
print("| # | pair | TRRUST direction (mode) | PMIDs | models (n) | weights | name rule | same DBD class | STRING | group (my reading) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for i, h in enumerate(e["tier2_all_ten"]["edges"], 1):
    p = tuple(h["pair"]); s = struct[p]
    grp = GROUP.get(p, "A TF -> target in a known cell programme")
    w = ", ".join(f"{ABBR[m]} {v}" for m, v in h["weights"].items())
    name = "same family" if h["same_family_rule"] else ("prefix" if h["same_3letter_prefix"] else "-")
    print(f"| {i} | {p[0]}–{p[1]} | {'; '.join(h['directions'])} ({'/'.join(m[:3] for m in h['modes'])}) | {h['n_pmids']} | "
          f"{', '.join(ABBR[m] for m in h['models'])} ({h['n_models']}) | {w} | {name} | "
          f"{'yes' if s['same_structural_family'] else '-'} | {'yes' if h['in_STRING'] else '-'} | {grp} |")
print()
for m, r in e["per_model"].items():
    rows = fb["per_model"][m]["rows"]
    fam = {tuple(x["pair"]): x for x in rows}
    items = []
    for h in r["hits"]:
        x = fam[tuple(h["pair"])]
        flag = ("*" if x["same_structural_family"] else "") + ("^" if x["shared_leading_chars"] >= 2 else "")
        items.append(f"{' / '.join(h['directions'])} ({h['weight']}){flag}")
    print(f"**{m}** ({r['n_hits']} hits): " + "; ".join(items))
    print()
