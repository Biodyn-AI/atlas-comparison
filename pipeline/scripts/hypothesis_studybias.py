#!/usr/bin/env python
"""Does the models' predictive power depend on how well studied a gene is?

Motivation. The models feature 4,003 genes with fewer than 5 papers, but our pair list keeps only
141 of them (3.5%, against 16.8% of well-studied genes), so the earlier "no new biology" result may
be a property of our filter rather than of the models. The direct fix -- validate understudied genes
causally -- is blocked: of the 793 protein-coding understudied genes the models feature, 752 are
absent from the Replogle screen entirely (the screen's perturbed set has 0.4% understudied genes
against 9.9% of all protein-coding genes).

So we ask the strongest question the data can answer. Among pairs that ARE testable, does the causal
signal weaken as the responding gene gets less studied? A flat profile means the models' predictions
do not depend on a gene's fame, which is what would license extrapolating them to genes no external
resource covers. A declining profile means the opposite.

Two changes from hypothesis_perturb.py: the co-firing threshold drops from >=5 features to >=2 (so
low-abundance genes are not pre-filtered), and every pair carries the publication count of its
responding gene. The causal read-out and null are unchanged: percentile of |dB| within perturbation
A, against a responsiveness-matched replacement for B.
    python scripts/hypothesis_studybias.py [NPERM] -> outputs/atlas/comparative/hypothesis_studybias.json
"""
from __future__ import annotations
import h5py, gzip, json, glob, sys, itertools
import numpy as np
from collections import defaultdict

BASE = "/Users/annaantipova/Desktop/biomech"
C = f"{BASE}/outputs/atlas/comparative"; PD = f"{BASE}/outputs/atlas/perturb"; PB = f"{BASE}/outputs/atlas/pubmed"
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"; TS = f"{BASE}/outputs/atlas/ts3_out"
PASSED = ["Tahoe", "scGPT", "UCE", "C2S", "Geneformer"]
TOPG, WLOW, KMOD = 10, 2, 2
NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200
TOPQ, NBIN = 0.05, 20
rng = np.random.default_rng(0)

# ---- publication counts -----------------------------------------------------
sym2id, syn2id = {}, {}
with gzip.open(f"{PB}/Homo_sapiens.gene_info.gz", "rt") as fh:
    fh.readline()
    for line in fh:
        f = line.rstrip("\n").split("\t")
        if f[0] != "9606": continue
        sym2id.setdefault(f[2].upper(), f[1])
        for sy in f[4].split("|"):
            sy = sy.strip().upper()
            if sy and sy != "-": syn2id.setdefault(sy, f[1])
cnt = defaultdict(int)
with gzip.open(f"{PB}/gene2pubmed.gz", "rt") as fh:
    fh.readline()
    for line in fh:
        t, g, _ = line.rstrip("\n").split("\t")
        if t == "9606": cnt[g] += 1
def papers(sym):
    """publication count, or None when the symbol maps to no NCBI gene (clone IDs like AL450998.2)"""
    gid = sym2id.get(sym) or syn2id.get(sym)
    return None if gid is None else cnt.get(gid, 0)

# ---- co-firing pairs at the LOW threshold -----------------------------------
def graph(m):
    c = defaultdict(int)
    d = ALLCAT if glob.glob(f"{ALLCAT}/{m}/feature_catalog_L*.json") else TS
    for p in sorted(glob.glob(f"{d}/{m}/feature_catalog_L*.json")):
        for f in json.load(open(p))["features"].values():
            s = sorted({str(x).upper() for x in f.get("top_genes", [])[:TOPG]
                        if not str(x).upper().startswith("ENSG")})
            if len(s) < 2: continue
            for a, b in itertools.combinations(s, 2): c[(a, b)] += 1
    return {k: v for k, v in c.items() if v >= WLOW}

allp = defaultdict(int)
for m in PASSED:
    g = graph(m)
    print(f"  {m}: {len(g)} pairs at W>={WLOW}", flush=True)
    for e in g: allp[e] += 1
pairs = {e: n for e, n in allp.items() if n >= KMOD}
pgenes = {x for e in pairs for x in e}
u = [g for g in pgenes if (papers(g) or 0) < 5 and papers(g) is not None]
print(f"\npairs corroborated by >={KMOD} of the validated models: {len(pairs)} over {len(pgenes)} genes")
print(f"  understudied (<5 papers) genes now retained: {len(u)}  (was 141 at W>=5)", flush=True)


def run(fn, label):
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
    pix = {g: i for i, g in enumerate(names)}; gix = {g: i for i, g in enumerate(genes)}
    A = np.abs(M)
    order = np.argsort(A, axis=1); pct = np.empty_like(A)
    ranks = np.arange(A.shape[1], dtype=np.float32) / (A.shape[1] - 1)
    for i in range(A.shape[0]): pct[i, order[i]] = ranks
    resp = A.mean(axis=0)
    edges = np.quantile(resp, np.linspace(0, 1, NBIN + 1)[1:-1])
    gbin = np.digitize(resp, edges)
    bins = defaultdict(list)
    for j, b in enumerate(gbin): bins[b] = bins[b] + [j]
    bins = {b: np.array(v) for b, v in bins.items()}

    rows, cols, pubB = [], [], []
    for (a, b) in pairs:
        for uu, vv in ((a, b), (b, a)):
            if uu in pix and vv in gix:
                rows.append(pix[uu]); cols.append(gix[vv])
                pb = papers(vv); pubB.append(-1 if pb is None else pb)
    rows = np.array(rows); cols = np.array(cols); pubB = np.array(pubB)
    print(f"\n{label}: {len(rows)} testable ordered pairs", flush=True)
    if len(rows) < 100: return None
    obs = pct[rows, cols]

    def null_rate(sel):
        sr, sc = rows[sel], cols[sel]; out = []
        for _ in range(NPERM):
            rc = np.empty_like(sc)
            for b, mem in bins.items():
                m2 = gbin[sc] == b
                if m2.any(): rc[m2] = rng.choice(mem, m2.sum(), replace=True)
            out.append(float((pct[sr, rc] >= 1 - TOPQ).mean()))
        return np.array(out)

    res = {}
    allsel = np.ones(len(rows), bool)
    nr = null_rate(allsel); r = float((obs >= 1 - TOPQ).mean())
    res["overall"] = {"n": int(len(rows)), "rate": round(r, 4), "null": round(float(nr.mean()), 4),
                      "fold": round(r / max(float(nr.mean()), 1e-9), 2),
                      "z": round(float((r - nr.mean()) / (nr.std() or 1)), 2),
                      "p": round((int((nr >= r).sum()) + 1) / (NPERM + 1), 5)}
    print(f"  overall: n={len(rows)}  {100*r:.2f}% vs {100*nr.mean():.2f}%  -> "
          f"{res['overall']['fold']}x, p={res['overall']['p']}", flush=True)

    qs = [-1, 0, 25, 50, 100, 400, 10**9]
    labels = ["unmapped*", "<25", "25-49", "50-99", "100-399", ">=400"]
    strata = {}
    print(f"  by how studied the RESPONDING gene is (papers); "
          f"unmapped* = clone-ID/lncRNA symbols with no NCBI gene record:", flush=True)
    for lo, hi, lab in zip(qs[:-1], qs[1:], labels):
        sel = (pubB == -1) if lab == "unmapped*" else ((pubB >= lo) & (pubB < hi))
        if sel.sum() < 100:
            print(f"    {lab:>9}: n={int(sel.sum())} (too few)", flush=True); continue
        rr = float((obs[sel] >= 1 - TOPQ).mean()); nn = null_rate(sel)
        strata[lab] = {"n": int(sel.sum()), "rate": round(rr, 4), "null": round(float(nn.mean()), 4),
                       "fold": round(rr / max(float(nn.mean()), 1e-9), 2),
                       "z": round(float((rr - nn.mean()) / (nn.std() or 1)), 2),
                       "p": round((int((nn >= rr).sum()) + 1) / (NPERM + 1), 5)}
        d = strata[lab]
        print(f"    {lab:>9}: n={d['n']:>5}  {100*rr:>5.2f}% vs {100*nn.mean():>5.2f}%  -> "
              f"{d['fold']}x, z={d['z']}, p={d['p']}", flush=True)
    res["by_papers"] = strata
    return res


out = {"threshold_features": WLOW, "min_models": KMOD, "n_perm": NPERM,
       "n_pairs": len(pairs), "n_genes": len(pgenes), "n_understudied_genes": len(u),
       "note": "understudied genes cannot be tested directly: 752 of the 793 protein-coding ones "
               "are absent from the Replogle screen. This measures whether predictive power depends "
               "on how studied the responding gene is, among the genes the screen does cover."}
out["K562"] = run("K562_gwps_normalized_bulk_01.h5ad", "K562")
out["RPE1"] = run("rpe1_normalized_bulk_01.h5ad", "RPE1")
json.dump(out, open(f"{C}/hypothesis_studybias.json", "w"), indent=1)
print("\n==> hypothesis_studybias.json", flush=True)
