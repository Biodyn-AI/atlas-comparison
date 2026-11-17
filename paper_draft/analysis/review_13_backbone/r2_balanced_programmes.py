#!/usr/bin/env python
"""(f) Primary programme rule (weighted hit-gene Jaccard, complete linkage, J>=0.3) recomputed with MODEL-BALANCED hit
weights: each model's hit counts for a concept are normalised to sum 1 before pooling, so tGPT / C2S (which give the
most hits for 78 of 78 concepts, r1_checks.json:c_model_giving_most_hits_count) cannot dominate the similarity.
Same clustering helper (bb_common.cluster_matrix), same 78 concepts. Also J>=0.5 and average/single linkage.
(g) per-model concept counts at 0 % and mid (from s1_slices_summary.json), to explain the 'all 7 agree' contrast.
Output: r2_balanced_programmes.json
"""
import json, os, sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
BC = os.path.join(os.path.dirname(HERE), "backbone_composition")
sys.path.insert(0, BC)
import bb_common as b                       # noqa: E402
from s2_mid_composition import load_slice  # noqa: E402

s2 = json.load(open(f"{BC}/s2_mid_composition.json"))["backbone"]
terms = sorted(s2["concepts"])
per = load_slice("mid")
balw = {}
for t in terms:
    acc = defaultdict(float)
    for m in b.ORDER:
        if t not in per[m]["terms"]: continue
        h = Counter()
        for fi, hs in per[m]["terms"][t]: h.update(hs)
        tot = sum(h.values())
        for g, c in h.items(): acc[g] += c / tot
    balw[t] = dict(acc)
out = {"programmes_model_balanced_weighted_jaccard": {}}
for method in ("complete", "average", "single"):
    for thr in (0.3, 0.5):
        cl = b.cluster_matrix(terms, lambda a, c: b.weighted_jaccard(balw[a], balw[c]), thr, method)
        cls = []
        for c in cl:
            pooled = Counter()
            for t in c:
                for g, v in balw[t].items(): pooled[g] += v
            tot = sum(pooled.values())
            fT = sum(v for g, v in pooled.items() if g in b.T_GENES) / tot
            fTO = sum(v for g, v in pooled.items() if g in (b.T_GENES | b.O_ALL)) / tot
            cls.append("protein_synthesis" if fT >= 0.5 else "other_housekeeping" if fTO >= 0.5 else "specific")
        out["programmes_model_balanced_weighted_jaccard"][f"{method}|J>={thr}"] = {
            "n_programmes": len(cl), "n_by_class": dict(Counter(cls)),
            "sizes": [len(c) for c in cl]}
out["pooled_primary_for_comparison"] = {k.split("|", 1)[1]: v["n_programmes"] for k, v in s2["programmes"].items()
                                        if k.startswith("hit_weighted|")}
S1 = json.load(open(f"{BC}/s1_slices_summary.json"))["slices"]
out["per_model_n_concepts"] = {nm: S1[nm]["n_concepts"] for nm in ("d0", "d25", "mid", "d75", "d100", "d0_block1")}
json.dump(out, open(f"{HERE}/r2_balanced_programmes.json", "w"), indent=1)
print(json.dumps(out, indent=1))
