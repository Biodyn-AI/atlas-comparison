"""Filler in the top-gene lists, counted on real list entries only.

The tie-tail detector (ie_common.tie_tail_with_firing) flags trailing slots of a top-20 list that are
alphabetical blocks, united with a hard bound from firing counts (slots n..19 of a feature that fired
on n < 20 positions). In the published catalogues those slots held filler genes; in the corrected
catalogues a list stops at the last gene with non-zero activation, so flagged slots past the end of a
short list are empty, not filler. This counts a feature as having filler in its top 10 only when a
flagged slot among the first ten holds a gene.
usage: ATLAS_BASE=<base> python filler_check.py <out.json>
"""
import json
import os
import sys
from collections import Counter

import numpy as np

from ie_common import load_model, universe_ranks, tie_tail_with_firing

MODELS = ["AIDO", "scGPT", "scFoundation", "GeneCompass", "tGPT", "MaxToki", "Geneformer", "UCE", "C2S", "Tahoe"]
res = {"base": os.environ.get("ATLAS_CATALOGUE_RELEASE", "runs-v2"), "rule": "feature counted if any of slots 0-9 is flagged (block rule U firing bound) and holds a gene",
       "models": {}}
for m in MODELS:
    d = load_model(m, cache=False); T20 = d["T20"]; rank = universe_ranks(d["names"])
    tail, tail_block, _ = tie_tail_with_firing(T20, rank, d["nfire"])
    pos = np.arange(20)[None, :]
    flag = (pos >= (20 - tail)[:, None]) & (T20 >= 0)
    in10 = flag[:, :10].any(1)
    short = (T20 < 0).any(1)
    tg = Counter(T20[flag].tolist())
    res["models"][m] = {"n_features": int(len(T20)), "share_filler_in_top10": round(float(in10.mean()), 4),
                        "share_filler_any_slot": round(float(flag.any(1).mean()), 4),
                        "share_lists_shorter_than_20": round(float(short.mean()), 4),
                        "detector_share_tail_into_top10": round(float((tail >= 11).mean()), 4),
                        "top_filler_genes": [[d["names"][g], c, round(float(rank[g]) / len(d["names"]), 3)]
                                             for g, c in tg.most_common(5)],
                        "per_layer_share_filler_in_top10": {str(L): round(float(in10[d["layer"] == L].mean()), 4)
                                                            for L in sorted(set(d["layer"].tolist()))}}
    print(m, res["models"][m]["share_filler_in_top10"], res["models"][m]["detector_share_tail_into_top10"],
          res["models"][m]["share_lists_shorter_than_20"], flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
