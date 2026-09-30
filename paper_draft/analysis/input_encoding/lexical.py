#!/usr/bin/env python
"""Parts (a) (b) (c) (d) (f) of notes/11_input_encoding.md.

For each of the ten models, in two scopes (all layers pooled; the mid layer of stats_final.py) and
two variants ("raw" = exactly the top-10 filter of hypothesis_trrust3.py; "clean" = the same filter
after blanking the tie-tail slots found by ie_common.tie_tail), measure over all within-feature gene
pairs:
  p3 / p2        share of pairs whose upper-case symbols share the first 3 / 2 characters
  lcp            mean longest-common-prefix length (characters)            [subsample of pairs]
  lev            mean Levenshtein distance / max(len)                       [subsample of pairs]
  adj1/5/20      share of pairs within 1 / 5 / 20 positions in the model's catalogue index order
                 (python-sorted gene strings = alphabetical; see ie_common)
  tok1/5/20      same, in the model's REAL token-id order where its vocabulary is on disk
                 (share among pairs whose two genes both have a token id)
  hg / hg_cov    share of pairs in the same HGNC gene group (all pairs / pairs with both genes grouped)
Null: slot permutation across the scope's features (every gene keeps its exact feature count and every
feature its size; within-feature duplicates removed by random swaps). p = (#null >= obs + 1)/(R + 1)
(for lev: #null <= obs).  Per-layer profiles for six models (part d).  Hub genes, tie-tail
validation against firing counts and Ensembl-id counts (part f).
    ATLAS_BASE=<copy> python lexical.py [R_POOL R_MID R_LAYER]  -> lexical.json
"""
from __future__ import annotations

import json, sys, time
from collections import Counter
import numpy as np
from ie_common import *

R_POOL = int(sys.argv[1]) if len(sys.argv) > 1 else 50
R_MID = int(sys.argv[2]) if len(sys.argv) > 2 else 200
R_LAYER = int(sys.argv[3]) if len(sys.argv) > 3 else 50
ONLY = sys.argv[4].split(",") if len(sys.argv) > 4 else ORDER
NLEV = 10000
LAYER_MODELS = ["C2S", "tGPT", "UCE", "Tahoe", "Geneformer", "scGPT"]
OUT = f"{OUTDIR}/lexical.json" if ONLY == ORDER else f"{OUTDIR}/lexical_{'_'.join(ONLY)}.json"

H, ENS2SYM, HVER = hgnc()
TOK = token_orders()
# UCE input tokens: ESM2 protein embeddings (uce_esm2_extract.py), unit-normalised, float16
ESM_KEYS = json.load(open(UCE_KEYS)); ESM_ID = {k.upper(): i for i, k in enumerate(ESM_KEYS)}
ESM = np.load(f"{SCR}/uce_esm2_emb_f16.npy").astype(np.float32)
ESM /= np.linalg.norm(ESM, axis=1, keepdims=True); ESM = ESM.astype(np.float16)
t00 = time.time()


def model_attrs(m, d):
    names = d["names"]; rank = universe_ranks(names)
    raw2u, un = upper_map(names)
    nU = len(un)
    c3, c2 = {}, {}
    p3 = np.array([c3.setdefault(u[:3], len(c3)) for u in un], np.int64)
    p2 = np.array([c2.setdefault(u[:2], len(c2)) for u in un], np.int64)
    arank = np.full(nU, np.iinfo(np.int64).max, np.int64)
    np.minimum.at(arank, raw2u, rank)
    tk = TOK.get(m)
    trank = np.full(nU, -1, np.int64)
    if tk is not None:
        has = np.array([u in tk for u in un])
        ids = np.array([tk.get(u, -1) for u in un])
        o = np.argsort(ids[has], kind="stable"); r = np.empty(has.sum(), np.int64); r[o] = np.arange(has.sum())
        trank[has] = r
    K = 4; Gk = np.full((nU, K), -1, np.int64); ngrp = 0; nmap = 0; nmulti = 0
    for i, u in enumerate(un):
        h = H.get(u)
        if h is None: continue
        nmap += 1
        if h[1]:
            ngrp += 1; g = sorted(h[1])
            if len(g) > K: nmulti += 1
            Gk[i, :min(K, len(g))] = g[:K]
    Cc, Ln = char_codes(un)
    esm = np.array([ESM_ID.get(u, -1) for u in un], np.int64)
    return dict(esm=esm, names=names, rank=rank, raw2u=raw2u, un=un, p3=p3, p2=p2, arank=arank, trank=trank,
                Gk=Gk, Cc=Cc, Ln=Ln, hgnc_mapped=nmap, hgnc_grouped=ngrp, hgnc_gt4groups=nmulti)


def metrics(A, B, at, rng, lev=True):
    out = {}
    out["p3"] = float(np.mean(at["p3"][A] == at["p3"][B]))
    out["p2"] = float(np.mean(at["p2"][A] == at["p2"][B]))
    da = np.abs(at["arank"][A] - at["arank"][B])
    for w in (1, 5, 20): out[f"adj{w}"] = float(np.mean(da <= w))
    del da
    ta, tb = at["trank"][A], at["trank"][B]; ok = (ta >= 0) & (tb >= 0)
    if ok.any():
        dt = np.abs(ta[ok] - tb[ok])
        for w in (1, 5, 20): out[f"tok{w}"] = float(np.mean(dt <= w))
        out["tok_cov"] = float(ok.mean())
    del ta, tb, ok
    Ga, Gb = at["Gk"][A], at["Gk"][B]
    same = np.zeros(len(A), bool)
    for i in range(Ga.shape[1]):
        gi = Ga[:, i]; v = gi >= 0
        for j in range(Gb.shape[1]):
            same |= v & (gi == Gb[:, j])
    both = (Ga[:, 0] >= 0) & (Gb[:, 0] >= 0)
    out["hg"] = float(same.mean()); out["hg_cov"] = float(same[both].mean()) if both.any() else None
    out["hg_both_frac"] = float(both.mean())
    del Ga, Gb, same, both
    if lev:
        idx = rng.choice(len(A), min(NLEV, len(A)), replace=False)
        a, b = A[idx], B[idx]
        CA, CB = at["Cc"][a], at["Cc"][b]
        eq = (CA == CB) & (CA != 0)
        out["lcp"] = float(np.cumprod(eq, 1).sum(1).mean())
        out["lev"] = float(levenshtein_norm(CA, at["Ln"][a], CB, at["Ln"][b]).mean())
        ea, eb = at["esm"][a], at["esm"][b]; ok = (ea >= 0) & (eb >= 0)
        ea, eb = ea[ok], eb[ok]; cs = []
        for c0 in range(0, len(ea), 2000):
            cs.append(np.einsum("ij,ij->i", ESM[ea[c0:c0 + 2000]].astype(np.float32), ESM[eb[c0:c0 + 2000]].astype(np.float32)))
        out["esm_cos"] = float(np.concatenate(cs).mean()) if cs else None
        out["esm_cov"] = float(ok.mean())
    return out


def run_scope(X, at, R, rng, lev=True):
    A, B = pairs_of(X)
    obs = metrics(A, B, at, rng, lev)
    nul = {k: [] for k in obs}
    for r in range(R):
        Y = slot_permute(X, rng); a, b = pairs_of(Y)
        mm = metrics(a, b, at, rng, lev)
        for k in nul:
            if mm.get(k) is not None: nul[k].append(mm[k])
    res = {}
    for k, v in obs.items():
        if v is None or k in ("tok_cov", "hg_both_frac", "esm_cov"):
            res[k] = v; continue
        res[k] = empirical(v, nul[k], greater=(k != "lev"))
    return {"n_features": int(len(X)), "n_pairs": int(len(A)), "metrics": res}


RES = {"design": {
    "filter": "hypothesis_trrust3.py build(): top-10 genes, upper-case, drop ENSG*, unique, >= 2 genes",
    "universe": "python-sorted raw gene strings seen in any top-20 list of the model (catalogue index order of pipeline.py)",
    "tie_tail": f"trailing run of top-20 slots whose genes form alphabetical blocks (>= {MINBLOCK} genes, rank gaps <= {GAP})",
    "clean": "tie-tail slots blanked before the same filter",
    "null": "slot permutation across features (exact gene counts and feature sizes); duplicates fixed by random swaps",
    "p": "(#null >= obs + 1)/(R + 1); for lev #null <= obs",
    "esm_cos": "mean cosine of the UCE input tokens (ESM2 protein embeddings, unit-normalised) over the lcp/lev pair subsample, pairs with both genes embedded",
    "R_pool": R_POOL, "R_mid": R_MID, "R_layer": R_LAYER, "n_pairs_for_lcp_lev": NLEV, "seed": 0,
    "hgnc": {"path": HGNC_PATH, **HVER}, "token_sources": TOKEN_SOURCE}, "models": {}}

for m in ONLY:
    t0 = time.time()
    rng = np.random.default_rng(0)
    d = load_model(m)
    at = model_attrs(m, d)
    T20 = d["T20"]
    nf = d["nfire"]
    tail, tail_block, inb = tie_tail_with_firing(T20, at["rank"], nf)   # block rule U firing bound
    keep = np.arange(20)[None, :] < (20 - tail)[:, None]
    Xraw, kr = filtered_top10(T20, at["raw2u"], at["un"])
    Xcln, kc = filtered_top10(T20, at["raw2u"], at["un"], keep)
    L_raw, L_cln = d["layer"][kr], d["layer"][kc]
    mid = mid_layer(m)
    R = {"n_features_all": int(len(T20)), "n_features_filtered_raw": int(len(Xraw)),
         "n_features_filtered_clean": int(len(Xcln)), "universe_size_reconstructed": len(at["names"]),
         "mid_layer": mid, "layers": sorted(int(x) for x in np.unique(d["layer"])),
         "n_positions_per_layer": {str(k): v for k, v in d["npos"].items()},
         "hgnc": {"n_symbols": len(at["un"]), "mapped": at["hgnc_mapped"], "with_group": at["hgnc_grouped"],
                  "gt4_groups_truncated": at["hgnc_gt4groups"]},
         "token_order": TOKEN_SOURCE.get(m, "not available on disk; alphabetical order only")}
    # ---- tie-tail description & validation against firing counts
    top10_tail = (tail >= 11)
    val = {"share_features_tail_into_top10_block_rule_only": round(float((tail_block >= 11).mean()), 4),"share_features_tail_into_top10": round(float(top10_tail.mean()), 4),
           "share_features_tail_ge3": round(float((tail >= 3).mean()), 4),
           "mean_tail_len": round(float(tail.mean()), 3)}
    if (nf >= 0).all():
        known = (nf > 0) & (nf < 20)
        if known.any():
            pos = np.arange(20)[None, :]
            certain = known[:, None] & (pos >= nf[:, None])          # positions that MUST be zero-ties
            flagged = pos >= (20 - tail_block)[:, None]            # block rule only (independent of firing)
            val["n_features_firing_lt20"] = int(known.sum())
            val["certain_tie_slots"] = int(certain.sum())
            val["certain_tie_slots_flagged_by_block_rule_share"] = round(float(flagged[certain].mean()), 4)
        val["share_features_firing_lt10"] = round(float(((nf > 0) & (nf < 10)).mean()), 4)
        for lo, hi in ((1, 10), (10, 100), (100, 1000), (1000, 10 ** 9)):
            s = (nf >= lo) & (nf < hi)
            if s.any():
                val[f"tail_into_top10_share_firing_{lo}_{hi if hi < 10**9 else 'inf'}"] = \
                    {"n": int(s.sum()), "share_block_rule": round(float((tail_block >= 11)[s].mean()), 4),
                     "share_final": round(float(top10_tail[s].mean()), 4)}
    # which genes fill the tails
    tg = Counter()
    for pos_ in range(20):
        sel = (20 - tail) <= pos_
        tg.update(T20[sel, pos_].tolist())
    val["top_tail_genes"] = [[at["names"][g], c, round(float(at["rank"][g]) / len(at["names"]), 3)]
                             for g, c in tg.most_common(12) if g >= 0]
    # tail slots that are ribosomal / mitochondrial genes (possible real blocks mislabelled as ties)
    tail_ids = np.concatenate([T20[(20 - tail) <= p, p] for p in range(20)])
    tn = [at["names"][g].upper() for g in tail_ids[tail_ids >= 0]]
    val["tail_slots_total"] = len(tn)
    val["tail_slots_RPL_RPS_MT_share"] = round(sum(1 for g in tn if g.startswith(("RPL", "RPS", "MT-"))) / max(len(tn), 1), 4)
    R["tie_tail"] = val
    # ---- hubs (pooled, filtered top-10), share of their occurrences in tie tails
    cnt = Counter(Xraw[Xraw >= 0].tolist()); cntc = Counter(Xcln[Xcln >= 0].tolist())
    U10 = np.where(T20[:, :10] >= 0, at["raw2u"][np.maximum(T20[:, :10], 0)], -1)
    intail10 = np.arange(10)[None, :] >= (20 - tail)[:, None]
    hubs = []
    for u, c in cnt.most_common(10):
        occ = (U10 == u); hubs.append([at["un"][u], c, round(float(intail10[occ].mean()), 4) if occ.any() else None])
    R["hubs_raw"] = hubs
    R["hubs_clean"] = [[at["un"][u], c] for u, c in cntc.most_common(10)]
    # ---- (f) Ensembl ids
    isens_raw = np.array([g.upper().startswith("ENSG") for g in at["names"]])
    T10 = T20[:, :10]; e10 = (T10 >= 0) & isens_raw[np.maximum(T10, 0)]
    ecnt = Counter(T10[e10].tolist())
    ens_univ = [g for g in at["names"] if g.upper().startswith("ENSG")]
    ens_feat = sorted({at["names"][g] for g in ecnt})
    R["ensembl"] = {"share_top10_slots": round(float(e10.sum() / max((T10 >= 0).sum(), 1)), 4),
                    "share_features_with_ensg_in_top10": round(float(e10.any(1).mean()), 4),
                    "distinct_ensg_in_top10": len(ecnt),
                    "ensg_in_universe": len(ens_univ), "universe_share": round(len(ens_univ) / len(at["names"]), 4),
                    "top_ensg": [[at["names"][g], c] for g, c in ecnt.most_common(5)],
                    "share_ensg_top10_slots_in_tie_tail": round(float(intail10[e10].mean()), 4) if e10.any() else None,
                    "ensg_in_top10_with_hgnc_symbol": sum(1 for e in ens_feat if e.split(".")[0] in ENS2SYM),
                    "ensg_in_universe_with_hgnc_symbol": sum(1 for e in ens_univ if e.split(".")[0] in ENS2SYM),
                    "share_top10_slots_after_clean": round(float((e10 & ~intail10).sum() / max(((T10 >= 0) & ~intail10).sum(), 1)), 4)}
    print(f"[{m}] loaded {len(T20)} feats, universe {len(at['names'])}, tail->top10 {val['share_features_tail_into_top10']}, "
          f"hubs {hubs[:3]}  ({time.time()-t0:.0f}s)", flush=True)
    # ---- scopes
    R["pooled"] = {"raw": run_scope(Xraw, at, R_POOL, rng), "clean": run_scope(Xcln, at, R_POOL, rng)}
    print(f"   pooled done ({time.time()-t0:.0f}s)  raw p3 {R['pooled']['raw']['metrics']['p3']['fold']}x "
          f"clean p3 {R['pooled']['clean']['metrics']['p3']['fold']}x", flush=True)
    R["mid"] = {"raw": run_scope(Xraw[L_raw == mid], at, R_MID, rng), "clean": run_scope(Xcln[L_cln == mid], at, R_MID, rng)}
    print(f"   mid done ({time.time()-t0:.0f}s)", flush=True)
    if m in LAYER_MODELS:
        pl = {}
        for Ly in R["layers"]:
            xr, xc = Xraw[L_raw == Ly], Xcln[L_cln == Ly]
            sel = d["layer"] == Ly
            pl[str(Ly)] = {"tail_into_top10_share": round(float(top10_tail[sel].mean()), 4),
                           "raw": run_scope(xr, at, R_LAYER, rng, lev=False) if len(xr) > 1 else None,
                           "clean": run_scope(xc, at, R_LAYER, rng, lev=False) if len(xc) > 1 else None}
        R["per_layer"] = pl
        print(f"   per-layer done ({time.time()-t0:.0f}s)", flush=True)
    RES["models"][m] = R
    json.dump(RES, open(OUT, "w"), indent=1)
    del d, at, T20, Xraw, Xcln
print(f"==> {OUT}  total {time.time()-t00:.0f}s")
