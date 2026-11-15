#!/usr/bin/env python
"""Candidate new links with independent support.

Starts from the released pairs that no paper names together (hypothesis_pubmed.json:
never_comentioned; gene2pubmed without the 60-gene cap, 98 pairs) and asks of each pair:
  1. CRISPR screens (Replogle et al. 2022, K562 genome-wide and RPE1 essential-gene, normalised
     pseudobulk; hypothesis_perturb.py read-out): knocking one gene down, where is the other in the
     within-row ranking of |change|? Per ordered direction, the percentile and an empirical p against
     NNULL responsiveness-matched random genes (20 bins of mean |change|, as hypothesis_perturb.py).
     BH over all tested directions; a pair is supported if any direction has q <= 0.05.
  2. Co-expression in the 6,000 atlas cells (log1p CP10k, genes in >= 10 cells): Pearson r of the
     two genes, its percentile among 1,000,000 random gene pairs, and the largest r within one cell
     type (>= 50 cells). A pair counts as co-expression if r is in the top 5 % of all pairs.
  3. Artefact flags: salivary / antimicrobial transcripts (ambient RNA in this corpus), stress and
     immediate-early genes, heat shock, haemoglobin, mitochondrial-encoded genes, ribosomal proteins.
Also the group-level rate for the 98 pairs against the same null, to compare with the full list.

    python novel_candidates.py   (ATLAS_BASE = _fixed_base)  -> novel_candidates.json
"""
from __future__ import annotations
import json, os, re, sys, time
from collections import defaultdict
import numpy as np
import h5py
import scipy.sparse as sp

BASE = os.environ["ATLAS_BASE"]
C = f"{BASE}/outputs/atlas/comparative"; PD = f"{BASE}/outputs/atlas/perturb"
CORPUS = f"{BASE}/external/perturb/tabula_3tissue_6k.h5ad"
HERE = os.path.dirname(os.path.abspath(__file__))
NNULL, NBIN, TOPQ = 2000, 20, 0.05
rng = np.random.default_rng(0)

PAIRS = json.load(open(f"{C}/hypothesis_pubmed.json"))["never_comentioned"]

FLAGS = [
    ("salivary/antimicrobial", r"^(HTN\d$|PRB\d$|PRH\d|STATH$|SMR3|MUC7$|CST[1-5]$|PIP$|ZG16B$|AMY1|BPIF|DEFA\d|DEFB\d|LYZ$|LTF$)"),
    ("immediate-early/stress", r"^(FOS|JUN|EGR\d$|ATF3$|IER\d|DUSP1$|NR4A\d$|KLF[246]$|ZFP36)"),
    ("heat shock", r"^(HSPA\d|HSPB\d|HSPD1|HSPE1|HSPH1|HSP90|DNAJ[ABC]\d)"),
    ("haemoglobin", r"^HB[ABDGEQMZ]\d?$"),
    ("mitochondrial-encoded", r"^MT-"),
    ("ribosomal protein", r"^(RPL\d|RPS\d|RPLP\d)"),
]


def flags(g):
    return [n for n, rx in FLAGS if re.match(rx, g)]


def load_perturb(fn):
    f = h5py.File(f"{PD}/{fn}", "r")
    gt = [x.decode() if isinstance(x, bytes) else str(x) for x in f["obs/gene_transcript"][:]]
    pert = [s.split("_")[1].upper() for s in gt]
    cats = [x.decode() if isinstance(x, bytes) else str(x) for x in f["var/__categories/gene_name"][:]]
    genes = [cats[i].upper() for i in f["var/gene_name"][:]]
    X = f["X"][:]; X[~np.isfinite(X)] = 0.0
    byg = defaultdict(list)
    for i, g in enumerate(pert): byg[g].append(i)
    names = sorted(byg)
    M = np.empty((len(names), X.shape[1]), np.float32)
    for i, g in enumerate(names): M[i] = X[byg[g]].mean(axis=0)
    A = np.abs(M)
    order = np.argsort(A, axis=1); pct = np.empty_like(A)
    ranks = np.arange(A.shape[1], dtype=np.float32) / (A.shape[1] - 1)
    for i in range(A.shape[0]): pct[i, order[i]] = ranks
    resp = A.mean(axis=0)
    edges = np.quantile(resp, np.linspace(0, 1, NBIN + 1)[1:-1]); gbin = np.digitize(resp, edges)
    bins = defaultdict(list)
    for j, b in enumerate(gbin): bins[b].append(j)
    return {g: i for i, g in enumerate(names)}, {g: i for i, g in enumerate(genes)}, pct, gbin, {b: np.array(v) for b, v in bins.items()}


def bh(p):
    p = np.asarray(p, float); n = len(p)
    if n == 0: return p
    o = np.argsort(p); r = np.arange(1, n + 1)
    q = np.minimum.accumulate((p[o] * n / r)[::-1])[::-1]; out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


def main():
    t0 = time.time()
    res = [{"pair": c["pair"], "n_models": c["n_models"], "models": c["models"], "papers": c.get("papers"),
            "flags": sorted(set(flags(c["pair"][0].upper()) + flags(c["pair"][1].upper()))), "perturb": []}
           for c in PAIRS]
    group = {}
    for screen, fn in (("K562", "K562_gwps_normalized_bulk_01.h5ad"), ("RPE1", "rpe1_normalized_bulk_01.h5ad")):
        pix, gix, pct, gbin, bins = load_perturb(fn)
        print(f"{screen} loaded [{time.time() - t0:.0f}s]", flush=True)
        obs_all, null_all = [], []
        for r in res:
            a, b = r["pair"][0].upper(), r["pair"][1].upper()
            for u, v in ((a, b), (b, a)):
                if u in pix and v in gix:
                    i, j = pix[u], gix[v]
                    o = float(pct[i, j])
                    mem = bins[gbin[j]]
                    nul = pct[i, rng.choice(mem, NNULL, replace=True)]
                    p = (int((nul >= o).sum()) + 1) / (NNULL + 1)
                    r["perturb"].append({"screen": screen, "knockdown": u, "measured": v,
                                         "percentile": round(o, 4), "p": p})
                    obs_all.append(o >= 1 - TOPQ); null_all.append(float((nul >= 1 - TOPQ).mean()))
        group[screen] = {"n_directions": len(obs_all),
                         "top5_rate": round(float(np.mean(obs_all)), 4) if obs_all else None,
                         "null_top5_rate": round(float(np.mean(null_all)), 4) if null_all else None}
    allp = [d for r in res for d in r["perturb"]]
    q = bh([d["p"] for d in allp])
    for d, qq in zip(allp, q): d["q"] = float(qq)
    for r in res:
        r["perturb_supported"] = any(d["q"] <= 0.05 for d in r["perturb"])
        r["perturb_tested"] = bool(r["perturb"])

    # co-expression in the atlas cells
    import anndata as ad
    a = ad.read_h5ad(CORPUS)
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    X = X.astype(np.float64); lib = np.asarray(X.sum(1)).ravel()
    X = sp.diags(1e4 / np.maximum(lib, 1)) @ X; X.data = np.log1p(X.data)
    det = np.asarray((X > 0).sum(0)).ravel() >= 10
    names = np.array([str(s).upper() for s in a.var["feature_name"].values])[det]
    Xd = X[:, det].toarray().astype(np.float32)
    Z = (Xd - Xd.mean(0)) / (Xd.std(0) + 1e-8); n = Z.shape[0]
    pos = {}
    for i, g in enumerate(names): pos.setdefault(g, i)
    ri = rng.integers(0, Z.shape[1], (1_000_000, 2)); ri = ri[ri[:, 0] != ri[:, 1]]
    rr = np.empty(len(ri), np.float32)
    for s in range(0, len(ri), 50_000):
        b = ri[s:s + 50_000]; rr[s:s + len(b)] = (Z[:, b[:, 0]] * Z[:, b[:, 1]]).sum(0) / n
    rr.sort(); thr95 = float(rr[int(0.95 * len(rr))])
    ct = a.obs["cell_type"].astype(str).values if "cell_type" in a.obs else None
    cts = [c for c in (sorted(set(ct)) if ct is not None else []) if (ct == c).sum() >= 50]
    for r in res:
        g1, g2 = r["pair"][0].upper(), r["pair"][1].upper()
        if g1 in pos and g2 in pos:
            i, j = pos[g1], pos[g2]
            rv = float((Z[:, i] * Z[:, j]).sum() / n)
            best, best_ct = None, None
            for c in cts:
                m = ct == c; x, y = Xd[m, i], Xd[m, j]
                if x.std() > 0 and y.std() > 0:
                    v = float(np.corrcoef(x, y)[0, 1])
                    if best is None or v > best: best, best_ct = v, c
            r["coexpr"] = {"r": round(rv, 4), "percentile": round(float(np.searchsorted(rr, rv) / len(rr)), 4),
                           "top5pct": rv >= thr95, "max_r_within_cell_type": None if best is None else round(best, 3),
                           "cell_type": best_ct}
        else:
            r["coexpr"] = None
    out = {"source": "hypothesis_pubmed.json never_comentioned (no shared paper in gene2pubmed, no cap)",
           "n_pairs": len(res), "null": f"{NNULL} responsiveness-matched random genes per direction; BH over directions",
           "coexpr_top5_threshold_r": round(thr95, 4), "group": group, "pairs": res}
    json.dump(out, open(os.path.join(HERE, "novel_candidates.json"), "w"), indent=1)
    sup = [r for r in res if r["perturb_supported"]]
    print(f"pairs {len(res)}; tested in a screen {sum(r['perturb_tested'] for r in res)}; supported {len(sup)}; group {group}")
    for r in sup:
        cx = r["coexpr"]
        print(" ", "-".join(r["pair"]), r["n_models"], r["models"], r["flags"],
              [(d["screen"], d["knockdown"], d["percentile"], round(d["q"], 4)) for d in r["perturb"] if d["q"] <= 0.05],
              None if cx is None else (cx["r"], cx["percentile"], cx["top5pct"], cx["max_r_within_cell_type"], cx["cell_type"]))
    print(f"==> novel_candidates.json [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
