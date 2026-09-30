#!/usr/bin/env python
"""Step 2 (job items 1 and 5, descriptive part): name every recovered TRRUST edge.

For each of the seven models (5 passing + MaxToki + tGPT) take the repo's pooled W>=5 graph and list
every pair that is a TRRUST edge (undirected, as the test scores it). For each hit record:
  - weight in that model (number of features containing both genes)
  - TRRUST direction(s) present (a->b, b->a, or both), regulation mode(s) and number of distinct
    PubMed ids supporting the edge (from the TRRUST v2 human raw file; its 8,427 directed edges are
    identical to outputs/atlas/genesets/trrust_edges.json, checked)
  - whether both genes are TRRUST regulators ("TF-TF pair")
  - same family by the repo's corrected rule (hypothesis_robust.py:35-49) and same first 3 letters
  - in STRING (string_edges.json) / co-members of a curated GO_BP/KEGG/Reactome set of <=200 genes
  - number of TRRUST regulators that regulate BOTH genes (shared regulators = "co-regulated" in TRRUST)
Also: the tier-2 set of the all-ten corroboration curve (pairs in >= 2 of the ten W>=5 graphs that are
TRRUST edges; published 31, hypothesis_trrust2.json:cross_model_curve.2.hits).
Output: OUT/s02_edges.json
"""
import csv, json, os, sys
from collections import defaultdict, Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

genes = load_genes(); gid = {g: i for i, g in enumerate(genes)}
truth, directed, tfs = load_truth()
tkey = truth_keys(truth, gid)
STRING = load_string(); SEEN = load_seen()

raw = defaultdict(lambda: {"modes": set(), "pmids": set()})
for r in csv.reader(open(TRRUST_RAW), delimiter="\t"):
    k = (r[0].upper(), r[1].upper())
    raw[k]["modes"].add(r[2]); raw[k]["pmids"].update(x for x in r[3].split(";") if x)
regs_of = defaultdict(set)          # target -> set of regulators
for a, b in directed:
    regs_of[b].add(a)


def annotate(a, b):
    d_ab, d_ba = (a, b) in directed, (b, a) in directed
    dirs = []
    if d_ab: dirs.append(f"{a}->{b}")
    if d_ba: dirs.append(f"{b}->{a}")
    modes, pm = set(), set()
    for k in ((a, b), (b, a)):
        if k in raw:
            modes |= raw[k]["modes"]; pm |= raw[k]["pmids"]
    both_tf = a in tfs and b in tfs
    fam = same_family(a, b); pref = a[:3] == b[:3]
    if both_tf and (fam or pref):
        cls = "TF-TF same family/prefix"
    elif both_tf:
        cls = "TF-TF different family"
    else:
        cls = "TF -> non-TF target"
    return {"pair": [a, b], "directions": dirs, "bidirectional": d_ab and d_ba,
            "tf": sorted(x for x, d in ((a, d_ab), (b, d_ba)) if d),
            "modes": sorted(modes), "n_pmids": len(pm),
            "both_trrust_regulators": both_tf, "same_family_rule": fam, "same_3letter_prefix": pref,
            "class": cls, "in_STRING": (a, b) in STRING, "in_curated_set_le200": (a, b) in SEEN,
            "n_shared_regulators": len((regs_of[a] - {b}) & (regs_of[b] - {a}))}


G5 = {}
for m in ORDER:
    feat, lay, L = load_model(m)
    keys, w = graph_from(feat, lay, None, 5)
    G5[m] = dict(zip(keys.tolist(), w.tolist()))

res = {"per_model": {}, "notes": "pair order is alphabetical; directions from TRRUST"}
for m in SEVEN:
    hits = []
    for k, w in G5[m].items():
        if np.isin(k, tkey):
            a, b = sorted((genes[k // SHIFT], genes[k % SHIFT]))
            hits.append({**annotate(a, b), "weight": int(w)})
    hits.sort(key=lambda h: -h["weight"])
    cls = Counter(h["class"] for h in hits)
    tfc = Counter(t for h in hits for t in h["tf"])
    res["per_model"][m] = {"n_hits": len(hits), "class_counts": dict(cls),
                           "n_bidirectional": sum(h["bidirectional"] for h in hits),
                           "n_in_STRING": sum(h["in_STRING"] for h in hits),
                           "n_in_curated_set": sum(h["in_curated_set_le200"] for h in hits),
                           "n_with_shared_regulator": sum(h["n_shared_regulators"] > 0 for h in hits),
                           "median_pmids": float(np.median([h["n_pmids"] for h in hits])) if hits else None,
                           "tf_hit_counts": dict(tfc.most_common()),
                           "hits": hits}
    print(m, len(hits), dict(cls), tfc.most_common(5), flush=True)

# ---- tier-2 of the all-ten curve
allp = Counter()
for m in ORDER:
    for k in G5[m]:
        allp[k] += 1
tier = []
for k, n in allp.items():
    if n >= 2 and np.isin(k, tkey):
        a, b = sorted((genes[k // SHIFT], genes[k % SHIFT]))
        ms = sorted(m for m in ORDER if k in G5[m])
        tier.append({**annotate(a, b), "n_models": n, "models": ms,
                     "weights": {m: int(G5[m][k]) for m in ms}, "total_weight": int(sum(G5[m][k] for m in ms))})
tier.sort(key=lambda h: (-h["n_models"], -h["total_weight"]))
res["tier2_all_ten"] = {"n": len(tier), "published_hits": 31,
                        "class_counts": dict(Counter(h["class"] for h in tier)),
                        "n_bidirectional": sum(h["bidirectional"] for h in tier),
                        "n_in_STRING": sum(h["in_STRING"] for h in tier),
                        "tf_counts": dict(Counter(t for h in tier for t in h["tf"]).most_common()),
                        "model_participation": dict(Counter(m for h in tier for m in h["models"])),
                        "edges": tier}
# tier >=1 union over the seven named models
print("tier2", len(tier), res["tier2_all_ten"]["class_counts"])
for h in tier:
    print(f"  {h['pair'][0]:>8}-{h['pair'][1]:<8} {h['n_models']} {','.join(h['models']):<40} {h['directions']} "
          f"{h['modes']} pm={h['n_pmids']} {h['class']} STRING={h['in_STRING']}")
json.dump(res, open(f"{OUT}/s02_edges.json", "w"), indent=1)
