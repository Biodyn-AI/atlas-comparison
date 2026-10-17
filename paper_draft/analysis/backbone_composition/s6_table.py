#!/usr/bin/env python
"""Supplementary table of the 78 (one row per concept): source, n_models, missing models, gene-content class,
content label, weighted protein-synthesis share, programme id (hit_weighted | complete | J>=0.3),
number of depth slices at >=8, model counts at 0/25/50/75/100 %. Output: backbone78_table.tsv"""
import json
import bb_common as b

s2 = json.load(open(f"{b.OUT}/s2_mid_composition.json"))["backbone"]
s3 = json.load(open(f"{b.OUT}/s3_depth.json"))
prog = {}
for i, p in enumerate(s2["programmes"]["hit_weighted|complete|J>=0.3"]["clusters"], 1):
    for t in p["members"]:
        prog[t] = (i, p["representative"].split(":", 1)[1])
cnt = s3["headline78_min_count_over_depths"]["counts_per_concept"]
rows = []
for t, c in s2["concepts"].items():
    src, name = t.split(":", 1)
    rows.append([name, src, c["n_models"], ",".join(c["missing"]) or "-", c["class_weighted"], c["class"],
                 c["dominant_label"], c["frac_T_weighted"], c["frac_T_union"], prog[t][0], prog[t][1],
                 sum(1 for v in cnt[t] if v >= 8), "/".join(map(str, cnt[t])), c["keyword_rule"],
                 " ".join(g for g, _ in c["top_hit_genes"][:6])])
rows.sort(key=lambda r: (-r[2], r[9], r[0]))
hdr = ["concept", "source", "n_models", "missing", "class_weighted", "class_union", "content_label",
       "frac_T_weighted", "frac_T_union", "programme_id", "programme_representative", "n_depths_ge8",
       "counts_0/25/50/75/100", "repo_keyword_rule", "top_hit_genes"]
with open(f"{b.OUT}/backbone78_table.tsv", "w") as f:
    f.write("\t".join(hdr) + "\n")
    for r in rows:
        f.write("\t".join(map(str, r)) + "\n")
print(len(rows), "rows ==> backbone78_table.tsv")
