#!/usr/bin/env python
"""Descriptive: how close is each model's real token-id order to alphabetical order?
Spearman correlation between the alphabetical rank and the token-id rank over the model's featured
genes that have a token id; coverage = share of the model's upper-case symbols with a token id.
-> token_order_check.json"""
import json
import numpy as np
from scipy.stats import spearmanr
from ie_common import *
TOK = token_orders(); out = {}
for m, tk in TOK.items():
    d = load_model(m); _, un = upper_map(d["names"])
    un = [u for u in un if not u.startswith("ENSG")]
    has = [u for u in un if u in tk]
    alpha = np.argsort(np.argsort(np.array(has, dtype=object)))
    ids = np.array([tk[u] for u in has])
    rho = spearmanr(alpha, ids).correlation
    out[m] = {"source": TOKEN_SOURCE[m], "n_symbols": len(un), "with_token": len(has),
              "coverage": round(len(has) / len(un), 4), "spearman_alpha_vs_token": round(float(rho), 4)}
    print(m, out[m])
json.dump(out, open(f"{OUTDIR}/token_order_check.json", "w"), indent=1)
