#!/usr/bin/env python
"""Review check for notes/11_input_encoding.md T-IE-5p / T-IE-5m (esm_cos).

lexical.py computes the OBSERVED esm_cos on one random 10,000-pair subsample, so the observed value
carries sampling noise of about the size of the null SD. Here the observed mean cosine is computed on
ALL within-feature pairs (clean lists, both genes embedded; same unit-normalised float16 ESM2 matrix
and same filter as lexical.py) and compared with the stored null mean / SD from lexical.json.
z_allpairs = (obs_all - null_mean) / null_sd; the stored null SD includes subsample noise, so this z
is conservative (too small), not inflated.
    ATLAS_BASE=<copy> python esm_allpairs.py -> esm_allpairs.json
"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
IE = os.path.join(os.path.dirname(HERE), "input_encoding")
sys.path.insert(0, IE)
from ie_common import *   # noqa

KEYS = json.load(open(UCE_KEYS)); EID = {k.upper(): i for i, k in enumerate(KEYS)}
ESM = np.load(f"{SCR}/uce_esm2_emb_f16.npy").astype(np.float32)
ESM /= np.linalg.norm(ESM, axis=1, keepdims=True); ESM = ESM.astype(np.float16)
LEX = json.load(open(f"{IE}/lexical.json"))["models"]
out = {"note": "observed esm_cos on all pairs vs stored null (lexical.json)", "models": {}}
for m in ORDER:
    d = load_model(m); raw2u, un = upper_map(d["names"]); rank = universe_ranks(d["names"])
    tail, _, _ = tie_tail_with_firing(d["T20"], rank, d["nfire"])
    keep = np.arange(20)[None, :] < (20 - tail)[:, None]
    Xc, kc = filtered_top10(d["T20"], raw2u, un, keep); Lc = d["layer"][kc]
    esm = np.array([EID.get(u, -1) for u in un], np.int64)
    M = {}
    for scope, X in (("mid", Xc[Lc == mid_layer(m)]), ("pooled", Xc)):
        A, B = pairs_of(X); ea, eb = esm[A], esm[B]; ok = (ea >= 0) & (eb >= 0); ea, eb = ea[ok], eb[ok]
        s = 0.0
        for c0 in range(0, len(ea), 20000):
            s += float(np.einsum("ij,ij->i", ESM[ea[c0:c0 + 20000]].astype(np.float32),
                                 ESM[eb[c0:c0 + 20000]].astype(np.float32)).sum())
        obs = s / max(len(ea), 1)
        st = LEX[m][scope]["clean"]["metrics"]["esm_cos"]
        M[scope] = {"n_pairs_embedded": int(len(ea)), "obs_allpairs": round(obs, 6), "obs_subsample_stored": st["obs"],
                    "null_mean_stored": st["null_mean"], "null_sd_stored": st["null_sd"], "z_stored": st["z"],
                    "z_allpairs": round((obs - st["null_mean"]) / st["null_sd"], 2) if st["null_sd"] else None}
    out["models"][m] = M
    print(m, {k: (v["z_stored"], v["z_allpairs"]) for k, v in M.items()}, flush=True)
    json.dump(out, open(f"{HERE}/esm_allpairs.json", "w"), indent=1)
    del d, Xc
