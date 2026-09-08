#!/usr/bin/env python
"""Three cross-model findings for the atlas, all LOCAL:
 (1) emergence   — concept-acquisition curve vs model scale (+ big-only vs universal category skew).
 (2) rosetta     — cross-model feature matching (same top-genes in different architectures = shared dict).
 (3) curriculum  — relative depth at which each concept category first appears, averaged across models.
    python scripts/findings.py [emergence|rosetta|curriculum|all]
"""
from __future__ import annotations
import sys, json, glob
import numpy as np
from collections import defaultdict, Counter

BASE = "/Users/annaantipova/Desktop/biomech"
TS = f"{BASE}/outputs/atlas/ts3_out"; C = f"{BASE}/outputs/atlas/comparative"
PAR = {"AIDO": 10, "tGPT": 50, "scGPT": 50, "MaxToki": 217, "Geneformer": 316, "UCE": 650, "C2S": 2000, "Tahoe": 3000, "scFoundation": 100, "GeneCompass": 104}
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
import re
RULES = [("translation", "translat|ribosom|peptide chain|aminoacyl|rrna|elongation"),
         ("cell-cycle/DNA", "cell cycle|mitotic|dna replicat|dna repair|chromosom|spindle|dna metabolic|telomere|nucleosome"),
         ("RNA-processing", "splic|mrna|rna processing|snrna|transcription|polymerase|nonsense-mediated"),
         ("mito/OXPHOS", "respiratory electron|atp synth|oxidative phosph|mitochond|electron transport|tca"),
         ("immune", "immune|neutrophil|interferon|cytokine|mhc|antigen|complement|inflamm|lymphocyte|interleukin|degranulation|leukocyte"),
         ("membrane/transport", "transport|endocytos|golgi|vesicle|slc|transmembrane|secretion|traffick|lysosom|endosom"),
         ("signaling", "signal|mapk|kinase|receptor|wnt|notch|gpcr|phosphoryl|pathway|rho gtpase"),
         ("metabolism", "metabol|biosynth|catabol|glycol|lipid|fatty acid|cholesterol|amino acid|nucleotide|heme"),
         ("PPI-hub", "ppi ")]


def catof(term):
    s = term.replace("STRING:", "PPI ").replace("GO_BP:", "").replace("Reactome:", "").replace("KEGG:", "").lower()
    for nm, pat in RULES:
        if re.search(pat, s):
            return nm
    return "other"


def dist(terms):
    c = Counter(catof(t) for t in terms); n = max(len(terms), 1)
    return {k: round(100 * v / n, 1) for k, v in c.most_common()}


def emergence(mat):
    models = [m for m in ORDER if m in mat]
    cm = defaultdict(set)
    for m in models:
        for t in mat[m]["term_count"]:
            cm[t].add(m)
    N = len(models); allc = set(cm)
    order = sorted(models, key=lambda m: PAR[m])
    cum, curve = set(), []
    for m in order:
        cum |= set(mat[m]["term_count"])
        curve.append({"model": m, "params_M": PAR[m], "cum_concepts": len(cum),
                      "cum_frac": round(100 * len(cum) / len(allc), 1)})
    uni = np.array([len(ms) for ms in cm.values()])
    mn = np.array([min(PAR[x] for x in ms) for ms in cm.values()])
    corr = round(float(np.corrcoef(uni, np.log10(mn))[0, 1]), 3)
    big = {"UCE", "C2S", "Tahoe"}
    bigonly = [t for t, ms in cm.items() if ms <= big]
    universal = [t for t, ms in cm.items() if len(ms) == N]
    # per-universality-level mean min-size
    lvl = defaultdict(list)
    for t, ms in cm.items():
        lvl[len(ms)].append(min(PAR[x] for x in ms))
    minsize_by_k = {str(k): round(float(np.median(lvl[k])), 1) for k in sorted(lvl)}
    return {"order": order, "curve": curve, "corr_uni_minsize": corr, "n_total": len(allc),
            "n_bigonly": len(bigonly), "bigonly_pct": round(100 * len(bigonly) / len(allc), 1),
            "n_universal": len(universal), "cat_universal": dist(universal), "cat_bigonly": dist(bigonly),
            "median_minsize_by_universality": minsize_by_k}


def load_topgenes(m, mat, ng=5):
    mid = mat[m]["layer"]
    for cp in glob.glob(f"{TS}/{m}/feature_catalog_L*.json"):
        j = json.load(open(cp))
        if j.get("layer") == mid:
            return {fid: set(str(x).upper() for x in f["top_genes"][:ng]) for fid, f in j["features"].items()}
    return {}


def rosetta(mat):
    models = [m for m in ORDER if m in mat]
    tg = {m: load_topgenes(m, mat) for m in models}
    THR = 0.4
    # pairwise: fraction of A-features with a Jaccard>=THR twin in B
    J = [[0.0] * len(models) for _ in models]
    for i, a in enumerate(models):
        for j, b in enumerate(models):
            if i == j:
                J[i][j] = 1.0; continue
            A = tg[a]; B = list(tg[b].values())
            inv = defaultdict(set)
            for fid, g in tg[b].items():
                for gene in g:
                    inv[gene].add(fid)
            hit = 0
            for ga in A.values():
                if not ga:
                    continue
                cand = set()
                for gene in ga:
                    cand |= inv.get(gene, set())
                if any(len(ga & tg[b][fid]) / len(ga | tg[b][fid]) >= THR for fid in cand):
                    hit += 1
            J[i][j] = round(100 * hit / max(len(A), 1), 1)
    # universal features: cluster identical-ish top-gene sets shared by many models (by gene-set signature)
    sig = defaultdict(set)  # frozenset(top3 genes) -> models
    examples = defaultdict(lambda: defaultdict(list))
    for m in models:
        for fid, g in tg[m].items():
            if len(g) >= 3:
                key = frozenset(sorted(g)[:3])
                sig[key].add(m)
                examples[key][m].append(fid)
    shared = sorted(((k, ms) for k, ms in sig.items() if len(ms) >= 4), key=lambda x: -len(x[1]))
    top_shared = [{"genes": sorted(k), "n_models": len(ms), "models": sorted(ms)} for k, ms in shared[:20]]
    return {"models": models, "match_pct": J, "thr": THR,
            "n_shared_ge4": len(shared), "top_shared_features": top_shared}


def curriculum(mat):
    from scipy.stats import fisher_exact
    G = f"{BASE}/outputs/atlas/genesets"
    MIN, MAXS, ALPHA, TOP = 5, 500, 0.05, 5
    gs = {nm: {t: set(g) for t, g in json.load(open(f"{G}/{nm}_gene_sets.json")).items() if MIN <= len(g) <= MAXS}
          for nm in ("GO_BP", "KEGG", "Reactome")}
    st = {t: set(v) for t, v in json.load(open(f"{G}/string_edges.json")).items() if MIN <= len(v) <= MAXS}
    bg = set(json.load(open(f"{G}/background.json"))); bg |= set().union(*st.values()) | set(st); bgn = len(bg)
    idx = defaultdict(list); size = {}
    for s, dd in list(gs.items()) + [("STRING", st)]:
        for t, genes in dd.items():
            k = (s, t); size[k] = len(genes)
            for gene in genes:
                idx[gene].append(k)

    def bh(p):
        p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.arange(1, n + 1)
        q = np.minimum.accumulate((p[o] * n / r)[::-1])[::-1]; out = np.empty(n); out[o] = np.clip(q, 0, 1); return out

    def annotate(feats):
        recs = []
        for fid, genes in feats.items():
            g = set(genes) & bg
            if len(g) < 3:
                continue
            cand = Counter()
            for gene in g:
                for k in idx.get(gene, ()):
                    cand[k] += 1
            for k, a in cand.items():
                if a < 2:
                    continue
                tgn = size[k]; b = len(g) - a; c = tgn - a; d = bgn - a - b - c
                _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
                recs.append((f"{k[0]}:{k[1]}", float(p)))
        if not recs:
            return set()
        q = bh([r[1] for r in recs])
        return {recs[i][0] for i in range(len(recs)) if q[i] <= ALPHA}

    models = [m for m in ORDER if m in mat]
    # per concept: list of relative depths (0..1) at which it appears, across models
    depth_of = defaultdict(list)
    for m in models:
        cats = sorted(glob.glob(f"{TS}/{m}/feature_catalog_L*.json"), key=lambda p: json.load(open(p))["layer"])
        if len(cats) > 6:
            idxs = sorted(set(round(x * (len(cats) - 1) / 4) for x in range(5)))
            cats = [cats[i] for i in idxs]
        L = len(cats)
        first = {}  # concept -> earliest rel depth in THIS model
        for li, cp in enumerate(cats):
            cat = json.load(open(cp))
            feats = {fid: [str(x).upper() for x in f["top_genes"]][:TOP] for fid, f in cat["features"].items()}
            cons = annotate(feats)
            rel = li / (L - 1) if L > 1 else 0.0
            for t in cons:
                if t not in first:
                    first[t] = rel
        for t, rel in first.items():
            depth_of[t].append(rel)
        print(f"  curriculum {m}: {L} layers done", flush=True)
    # aggregate by category
    by_cat = defaultdict(list)
    for t, rels in depth_of.items():
        by_cat[catof(t)].append(float(np.mean(rels)))
    cats_mean = sorted(((c, round(float(np.mean(v)), 3), len(v)) for c, v in by_cat.items() if len(v) >= 20),
                       key=lambda x: x[1])
    return {"models": models, "category_first_depth": [{"cat": c, "mean_rel_depth": d, "n": n} for c, d, n in cats_mean]}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    mat = json.load(open(f"{C}/matrix_ts3_string.json"))
    out = json.load(open(f"{C}/findings.json")) if glob.glob(f"{C}/findings.json") else {}
    if which in ("all", "emergence"):
        out["emergence"] = emergence(mat); print("emergence done", flush=True)
    if which in ("all", "rosetta"):
        out["rosetta"] = rosetta(mat); print("rosetta done", flush=True)
    if which in ("all", "curriculum"):
        out["curriculum"] = curriculum(mat); print("curriculum done", flush=True)
    json.dump(out, open(f"{C}/findings.json", "w"))
    print(f"==> findings.json keys: {sorted(out)}", flush=True)


if __name__ == "__main__":
    main()
