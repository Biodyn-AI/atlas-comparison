#!/usr/bin/env python
"""Possible false positives of the tie-tail detector: flagged features (tail reaches the top-10) that
fired on >= 100 token positions (a real tie fill needs the feature to hit fewer than 20 distinct genes,
which is rare at that firing count). Lists the most common genes in their flagged tails.
-> tail_fp_check.json"""
import json
from collections import Counter
import numpy as np
from ie_common import *
out = {}
for m in ORDER:
    d = load_model(m); rank = universe_ranks(d["names"])
    tail, _, _ = tie_tail_with_firing(d["T20"], rank, d["nfire"])
    fl = tail >= 11; hi = fl & (d["nfire"] >= 100)
    c = Counter()
    for i in np.where(hi)[0]:
        c.update(d["names"][g] for g in d["T20"][i, 20 - tail[i]:] if g >= 0)
    out[m] = {"flagged_top10": int(fl.sum()), "flagged_and_fired_ge100": int(hi.sum()),
              "share": round(float(hi.sum() / max(fl.sum(), 1)), 4),
              "genes_in_those_tails": c.most_common(12)}
    print(m, out[m]["flagged_top10"], out[m]["flagged_and_fired_ge100"], c.most_common(8), flush=True)
json.dump(out, open(f"{OUTDIR}/tail_fp_check.json", "w"), indent=1)
