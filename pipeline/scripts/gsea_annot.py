#!/usr/bin/env python
"""A second annotation method — rank-based (GSEA-spirit) instead of Fisher-on-a-cutoff. Fisher asks 'do the
top-5 genes overlap a set more than chance' (counts, hard cutoff). This asks 'are a set's genes concentrated
near the TOP of the feature's ranked genes' (position-aware, no hard cutoff — uses the ranked top-20). For each
feature × candidate set with >=2 hits among the top-20, a rank-sum z-test vs the uniform null (mean rank 10.5);
lower mean rank = enriched at the top. BH<0.05. Reports per model: annotation rate, distinct concepts, median z,
and the % of concepts it shares with the Fisher(top-5) call — does the method change the picture?
    python scripts/gsea_annot.py   -> outputs/atlas/comparative/gsea_annot.json
"""
from __future__ import annotations
import json, glob, math
import numpy as np
from collections import defaultdict, Counter

BASE = "/Users/annaantipova/Desktop/biomech"
G = f"{BASE}/outputs/atlas/genesets"; TS = f"{BASE}/outputs/atlas/ts3_out"; C = f"{BASE}/outputs/atlas/comparative"
MIN, MAX, ALPHA, N = 5, 500, 0.05, 20
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]

print("loading gene sets...", flush=True)
gs = {nm: {t: set(g) for t, g in json.load(open(f"{G}/{nm}_gene_sets.json")).items() if MIN <= len(g) <= MAX}
      for nm in ("GO_BP", "KEGG", "Reactome")}
tr = {t: set(v) for t, v in json.load(open(f"{G}/trrust_edges.json")).items() if MIN <= len(v) <= MAX}
st = {t: set(v) for t, v in json.load(open(f"{G}/string_edges.json")).items() if MIN <= len(v) <= MAX}
bg = set(json.load(open(f"{G}/background.json"))); bg |= set().union(*st.values()) | set(st)
idx = defaultdict(list)
for s, d in list(gs.items()) + [("TRRUST", tr), ("STRING", st)]:
    for t, genes in d.items():
        for g in genes:
            idx[g].append((s, t))


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.arange(1, n + 1)
    q = np.minimum.accumulate((p[o] * n / r)[::-1])[::-1]; out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


def ranksum_p(ranks):
    """One-sided p that these k ranks (from 1..N) are smaller (nearer the top) than uniform expectation."""
    k = len(ranks); mu = k * (N + 1) / 2.0
    var = k * (N + 1) * (N - k) / 12.0
    if var <= 0:
        return 1.0
    z = (mu - sum(ranks)) / math.sqrt(var)          # lower rank-sum = enriched at top -> positive z
    return 0.5 * math.erfc(z / math.sqrt(2))


def gsea_annotate(feats_ranked):
    recs = []
    for fid, genes in feats_ranked.items():
        pos = {}
        for r, g in enumerate(genes[:N], 1):
            if g in bg and g not in pos:
                pos[g] = r
        if len(pos) < 3:
            continue
        setranks = defaultdict(list)
        for g, r in pos.items():
            for k in idx.get(g, ()):
                setranks[k].append(r)
        for k, ranks in setranks.items():
            if len(ranks) < 2:
                continue
            recs.append((fid, f"{k[0]}:{k[1]}", ranksum_p(ranks),
                         (N + 1) / 2 - np.mean(ranks)))   # 'z-ish' top-shift
    if not recs:
        return {}, []
    q = bh([r[2] for r in recs]); ann = defaultdict(list); shifts = []
    for i, (fid, term, p, sh) in enumerate(recs):
        if q[i] <= ALPHA:
            ann[fid].append(term); shifts.append(sh)
    return ann, shifts


mat = json.load(open(f"{C}/matrix_ts3_string.json"))   # for Fisher(top-5) concept comparison
out = {"models": {}}
for m in ORDER:
    cats = sorted(glob.glob(f"{TS}/{m}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])
    if not cats:
        continue
    cat = json.load(open(cats[len(cats) // 2])); feats_in = cat["features"]; nfeat = cat.get("n_alive", len(feats_in))
    feats = {fid: [str(x).upper() for x in f["top_genes"]] for fid, f in feats_in.items()}
    ann, shifts = gsea_annotate(feats)
    gsea_terms = set(t for v in ann.values() for t in v)
    fisher_terms = set(mat[m]["term_count"]) if m in mat else set()
    jac = round(len(gsea_terms & fisher_terms) / max(len(gsea_terms | fisher_terms), 1), 3)
    out["models"][m] = {"annot_rate": round(100 * len(ann) / max(nfeat, 1), 1), "n_concepts": len(gsea_terms),
                        "med_shift": round(float(np.median(shifts)), 2) if shifts else 0,
                        "concept_jaccard_vs_fisher": jac}
    print(f"  {m}: GSEA {out['models'][m]['annot_rate']}% · {len(gsea_terms)} concepts · Jaccard vs Fisher {jac}", flush=True)
json.dump(out, open(f"{C}/gsea_annot.json", "w"))
print("==> gsea_annot.json", flush=True)
