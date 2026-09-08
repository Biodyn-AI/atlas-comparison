#!/usr/bin/env python
"""(a) Is the rescued backbone robust, or an argmax over one lucky config/threshold? Sweep several CALIBRATED
annotators x sharing thresholds (>=7/8/9/10) and report, for each cell, the real shared-count vs the random-gene
null (fold). (b) How trivial is it? Classify the >=8 backbone concepts as housekeeping/ubiquitous machinery
(translation, ER targeting, splicing, proteasome, OXPHOS, cycle, mRNA processing) vs specific (immune, signalling,
tissue). Reports both -> outputs/atlas/comparative/recalibrate_robust.json
"""
from __future__ import annotations
import json, glob
import numpy as np
from scipy.stats import hypergeom
from collections import defaultdict, Counter

BASE = "/Users/annaantipova/Desktop/biomech"
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"; TS = f"{BASE}/outputs/atlas/ts3_out"
ALPHA, NPERM = 0.05, 20
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
N = len(ORDER); rng = np.random.default_rng(0)

RAW = {nm: {t: set(x.upper() for x in g) for t, g in json.load(open(f"{G}/{nm}_gene_sets.json")).items()}
       for nm in ("GO_BP", "KEGG", "Reactome")}
BG = set(x.upper() for x in json.load(open(f"{G}/background.json"))); BGN = len(BG); BG_ARR = np.array(sorted(BG))


def index(mx):
    gs = {}
    for src in RAW:
        for t, genes in RAW[src].items():
            if 5 <= len(genes) <= mx:
                gs[f"{src}:{t}"] = genes
    idx = defaultdict(list); size = {}
    for t, genes in gs.items():
        size[t] = len(genes)
        for g in genes:
            idx[g].append(t)
    return idx, size


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.arange(1, n + 1)
    q = np.minimum.accumulate((p[o] * n / r)[::-1])[::-1]; out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


def annotate(feat_lists, idx, size, top, amin, ormin):
    A = []; TG = []; LG = []; TERM = []
    for genes in feat_lists:
        g = set(str(x).upper() for x in genes[:top]) & BG
        if len(g) < 3:
            continue
        cand = Counter()
        for gene in g:
            for k in idx.get(gene, ()):
                cand[k] += 1
        for k, a in cand.items():
            if a < amin:
                continue
            A.append(a); TG.append(size[k]); LG.append(len(g)); TERM.append(k)
    if not A:
        return set()
    A = np.array(A); TG = np.array(TG); LG = np.array(LG)
    p = hypergeom.sf(A - 1, BGN, TG, LG)
    if ormin > 0:
        a = A.astype(float); lg = LG.astype(float); tg = TG.astype(float)
        orr = (a * (BGN - tg - (lg - a))) / np.maximum((lg - a) * (tg - a), 1e-9)
        keep = (bh(p) <= ALPHA) & (orr >= ormin)
    else:
        keep = bh(p) <= ALPHA
    return set(TERM[i] for i in np.where(keep)[0])


def load_feats(m):
    d = f"{ALLCAT}/{m}"
    if not glob.glob(f"{d}/feature_catalog_L*.json"):
        d = f"{TS}/{m}"
    cats = sorted(glob.glob(f"{d}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])
    return [f["top_genes"] for f in json.load(open(cats[len(cats) // 2]))["features"].values()]

FEATS = {m: load_feats(m) for m in ORDER}; NFEAT = {m: len(FEATS[m]) for m in ORDER}

CONFIGS = [  # (name, top, amin, max, OR)
    ("top10 a3 max200", 10, 3, 200, 0),
    ("top10 a3 max150", 10, 3, 150, 0),
    ("top12 a3 max200", 12, 3, 200, 0),
    ("top10 a3 max300", 10, 3, 300, 0),
    ("top10 a3 max200 OR5", 10, 3, 200, 5),
]
THRESH = [7, 8, 9, 10]

grid = []
winner_ge8_terms = None
for name, top, amin, mx, ormin in CONFIGS:
    idx, size = index(mx)
    real = {m: annotate(FEATS[m], idx, size, top, amin, ormin) for m in ORDER}
    cm = Counter()
    for m in ORDER:
        for t in real[m]:
            cm[t] += 1
    real_ge = {k: sum(1 for v in cm.values() if v >= k) for k in THRESH}
    if name == "top10 a3 max200":
        winner_ge8_terms = sorted([t for t, v in cm.items() if v >= 8])
    # null
    null_ge = {k: [] for k in THRESH}
    for _ in range(NPERM):
        cc = Counter()
        for m in ORDER:
            rs = annotate([BG_ARR[rng.integers(0, BGN, top)].tolist() for _ in range(NFEAT[m])], idx, size, top, amin, ormin)
            for t in rs:
                cc[t] += 1
        for k in THRESH:
            null_ge[k].append(sum(1 for v in cc.values() if v >= k))
    row = {"config": name}
    for k in THRESH:
        nm_ = float(np.mean(null_ge[k])); row[f">={k}"] = {"real": real_ge[k], "null": round(nm_, 1),
                                                            "fold": round(real_ge[k] / max(nm_, 0.3), 1)}
    grid.append(row)
    print(f"{name:22} " + " ".join(f">={k}:{row[f'>={k}']['real']}v{row[f'>={k}']['null']}({row[f'>={k}']['fold']}x)" for k in THRESH), flush=True)

# (b) triviality of the winner's >=8 backbone
HK = ["translat", "ribosom", "rrna", "trna", "srp", "cotranslational", "targeting to membrane", "targeting to er",
      "protein targeting", "nonsense-mediated", "mrna catabolic", "mrna processing", "mrna splic", "spliceosom",
      "rna processing", "proteasom", "oxidative phosphoryl", "respiratory", "electron transport", "mitochond",
      "dna replication", "dna repair", "cell cycle", "mitotic", "chromatin", "histone", "ubiquitin", "chaperone",
      "peptide biosynth", "amide biosynth", "gene expression", "rrna processing", "ribonucleoprotein"]


def is_hk(term):
    t = term.split(":", 1)[1].lower()
    return any(h in t for h in HK)

hk = [t for t in winner_ge8_terms if is_hk(t)]
sp = [t for t in winner_ge8_terms if not is_hk(t)]
triv = {"n_ge8": len(winner_ge8_terms), "housekeeping": len(hk), "specific": len(sp),
        "pct_housekeeping": round(100 * len(hk) / max(len(winner_ge8_terms), 1), 1),
        "specific_terms": [t.split(":", 1)[1] for t in sp][:40]}

out = {"grid": grid, "thresholds": THRESH, "n_perm": NPERM, "triviality": triv}
json.dump(out, open(f"{C}/recalibrate_robust.json", "w"), indent=1)
print(f"\n(b) >=8 backbone: {len(winner_ge8_terms)} concepts | housekeeping {len(hk)} ({triv['pct_housekeeping']}%) | specific {len(sp)}")
print("   specific examples:", triv["specific_terms"][:12])
print("==> recalibrate_robust.json")
