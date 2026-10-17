#!/usr/bin/env python
"""Why do MaxToki and GeneCompass lack every protein-synthesis concept at the mid layer?
For each slice and model: how many features have >=1 / >=3 cytosolic ribosomal-protein genes (bb_common.RP_GENES)
in their top-10, how many distinct RP genes appear in any feature's top-20, and whether the model has
KEGG Ribosome or GO cytoplasmic translation (0-2). Output: s2b_rp_presence.json
"""
import json
import bb_common as b

SCR = b.CACHE + "/bb_cache"
out = {}
for nm in ["d0", "d25", "d50", "mid", "d75", "d100"]:
    per = json.load(open(f"{SCR}/slice_{nm}.json"))
    row = {}
    for m in b.ORDER:
        feats, _ = b.load_cat(b.layer_path(m, per[m]["layer"]))
        row[m] = {"layer": per[m]["layer"], "n_feat": len(feats),
                  "feats_with_any_RP_in_top10": sum(1 for f in feats if any(g.upper() in b.RP_GENES for g in f[:10])),
                  "feats_with_ge3_RP_in_top10": sum(1 for f in feats if sum(g.upper() in b.RP_GENES for g in f[:10]) >= 3),
                  "distinct_RP_genes_in_any_top20": len(set(g.upper() for f in feats for g in f[:20]) & b.RP_GENES),
                  "has_Ribosome_or_cyto_translation": sum(1 for t in per[m]["terms"] if t in ("KEGG:Ribosome", "GO_BP:cytoplasmic translation (GO:0002181)"))}
    out[nm] = row
    print(nm, {m: (r["feats_with_ge3_RP_in_top10"], r["distinct_RP_genes_in_any_top20"]) for m, r in row.items()})
out["n_RP_genes_in_background"] = len(b.RP_GENES)
print("==>", b.dump(out, "s2b_rp_presence.json"))
