"""Shared helpers for the TRRUST robustness analysis (paper_draft/analysis/trrust_robustness).

Everything here re-implements, line for line, the graph construction and the null of the repo's
pipeline/scripts/hypothesis_trrust2.py / hypothesis_trrust3.py / hypothesis_robust.py, so that the
numbers are on the same ruler as the published ones:

  * catalogue source: ALLCAT/<m>/feature_catalog_L*.json if present, else TS/<m>/...
    (hypothesis_trrust2.py:49)
  * per feature: top_genes[:10], upper-cased, Ensembl ids (ENSG...) dropped, de-duplicated; the
    feature is skipped if < 2 genes remain (hypothesis_trrust2.py:52-54)
  * pair weight = number of features (pooled over the chosen layers) containing both genes
    (hypothesis_trrust2.py:56); predicted if weight >= W (W0 = 5, hypothesis_trrust2.py:25)
  * truth = TRRUST edges made undirected, self-pairs dropped (hypothesis_trrust2.py:29-33)
  * null = stub-matching configuration model; stubs shuffled and re-paired, self-loops dropped,
    duplicate pairs collapsed (config_shuffle, hypothesis_trrust2.py:60-69)
  * fold = hits / null mean; p = (#null >= obs + 1) / (N + 1) (hypothesis_trrust2.py:100-106)
  * same-family rule = hypothesis_robust.py:35-49 (copied verbatim below)

The only difference: genes are indexed over ALL genes in any catalogue (not only genes in the W>=5
union), because layer subsets and other thresholds change the union. Hit counts do not depend on
the index; the null only ever draws genes that are in the model's own graph.

Inputs: ATLAS_BASE = repository root with the release archives unpacked (pipeline/scripts/fetch_runs.sh);
defaults to the root of this checkout. ATLAS_CACHE = folder for intermediate files. TRRUST_RAW = the
TRRUST v2 human raw file trrust_rawdata.human.tsv (https://www.grnpedia.org/trrust/), used only to annotate.
"""
from __future__ import annotations

import glob
import itertools
import json
import os

import numpy as np

_ANALYSIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # paper_draft/analysis
# repository root with the release archives unpacked (pipeline/scripts/fetch_runs.sh)
BASE = os.environ.get("ATLAS_BASE", os.path.abspath(os.path.join(_ANALYSIS, "..", "..")))
# folder for intermediate files
CACHE = os.environ.get("ATLAS_CACHE", os.path.join(_ANALYSIS, ".cache"))
C = f"{BASE}/outputs/atlas/comparative"
G = f"{BASE}/outputs/atlas/genesets"
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"
TS = f"{BASE}/outputs/atlas/ts3_out"
SCRATCH = os.path.join(CACHE, "trrust_rob")
OUT = os.path.dirname(os.path.abspath(__file__))
# original repo (read-only) for published JSON values
ORIG = BASE
# local copy of the TRRUST v2 human raw file (TF, target, mode, PMIDs); used only to annotate
TRRUST_RAW = os.environ.get("TRRUST_RAW", os.path.join(CACHE, "trrust_rawdata.human.tsv"))

ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
PASSED = ["Tahoe", "scGPT", "UCE", "C2S", "Geneformer"]
SEVEN = ["Tahoe", "scGPT", "UCE", "C2S", "Geneformer", "MaxToki", "tGPT"]
TOPG = 10
SHIFT = 1 << 21


def catalog_files(m):
    d = ALLCAT if glob.glob(f"{ALLCAT}/{m}/feature_catalog_L*.json") else TS
    return sorted(p for p in glob.glob(f"{d}/{m}/feature_catalog_L*.json")
                  if not os.path.basename(p).startswith("._"))


def load_truth():
    """TRRUST, exactly as hypothesis_trrust2.py:29-33 (undirected, self-pairs dropped).
    Also returns the directed edge set and the regulator (TF) set."""
    tr = json.load(open(f"{G}/trrust_edges.json"))
    und, directed = set(), set()
    for a, bs in tr.items():
        for b in bs:
            A, B = str(a).upper(), str(b).upper()
            if A != B:
                directed.add((A, B))
                u, v = sorted((A, B))
                und.add((u, v))
    tfs = {str(a).upper() for a in tr}
    return und, directed, tfs


def load_string():
    s = set()
    for a, bs in json.load(open(f"{G}/string_edges.json")).items():
        for b in bs:
            u, v = sorted((str(a).upper(), str(b).upper()))
            if u != v:
                s.add((u, v))
    return s


def load_seen():
    seen = set()
    for src in ("GO_BP", "KEGG", "Reactome"):
        for t, genes in json.load(open(f"{G}/{src}_gene_sets.json")).items():
            gl = sorted({str(x).upper() for x in genes})
            if len(gl) <= 200:
                seen.update(itertools.combinations(gl, 2))
    return seen


def same_family(a, b):
    """verbatim copy of hypothesis_robust.py:35-49"""
    a, b = a.upper(), b.upper()
    if a == b:
        return True
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    stem = a[:n].rstrip("-")
    return len(stem) >= 3 and len(a) - n <= 3 and len(b) - n <= 3


# ---------------------------------------------------------------- cache of feature gene lists
def cache_path(m):
    return f"{SCRATCH}/genes_{m}.npz"


def load_genes():
    return json.load(open(f"{SCRATCH}/gene_index.json"))


def load_model(m):
    """returns (feat array int32 (n_feat, 10) padded with -1, layer index per feature, n_layers)"""
    z = np.load(cache_path(m))
    return z["feat"], z["layer"], int(z["n_layers"])


COMBOS = np.array(list(itertools.combinations(range(TOPG), 2)), np.int64)


def pair_keys(feat):
    """all within-feature unordered pairs as int64 keys lo*2^21+hi (one entry per feature per pair)"""
    out = []
    for i0 in range(0, len(feat), 50000):
        f = feat[i0:i0 + 50000].astype(np.int64)
        a = f[:, COMBOS[:, 0]]
        b = f[:, COMBOS[:, 1]]
        ok = (a >= 0) & (b >= 0)
        lo = np.minimum(a, b)[ok]
        hi = np.maximum(a, b)[ok]
        out.append(lo * SHIFT + hi)
    return np.concatenate(out) if out else np.empty(0, np.int64)


def graph_from(feat, layer, keep_layers=None, W=5):
    """co-firing graph: unique pair keys with weight >= W, and their weights"""
    if keep_layers is not None:
        feat = feat[np.isin(layer, np.asarray(sorted(keep_layers)))]
    k = pair_keys(feat)
    u, c = np.unique(k, return_counts=True)
    sel = c >= W
    return u[sel], c[sel]


def truth_keys(truth, gid):
    return np.array(sorted({min(gid[a], gid[b]) * SHIFT + max(gid[a], gid[b])
                            for a, b in truth if a in gid and b in gid}), np.int64)


def decode(keys):
    return (keys // SHIFT).astype(np.int64), (keys % SHIFT).astype(np.int64)


def config_shuffle_keys(keys, rng):
    """repo's config_shuffle (hypothesis_trrust2.py:60-69) on int64 keys"""
    a, b = decode(keys)
    stubs = np.concatenate([a, b]).astype(np.int32)
    rng.shuffle(stubs)
    h = stubs[::2]; t = stubs[1::2]
    ok = h != t
    h, t = h[ok], t[ok]
    lo = np.minimum(h, t).astype(np.int64); hi = np.maximum(h, t).astype(np.int64)
    return np.unique(lo * SHIFT + hi)


def score(keys, tkey, nperm, seed=0, return_null=False):
    rng = np.random.default_rng(seed)
    obs = int(np.isin(keys, tkey).sum())
    if len(keys) == 0:
        return {"n_pairs": 0, "hits": 0}
    nul = np.array([int(np.isin(config_shuffle_keys(keys, rng), tkey).sum()) for _ in range(nperm)], float)
    d = summarize(obs, nul, len(keys))
    if return_null:
        return d, nul
    return d


def summarize(obs, nul, n_pairs=None):
    m = float(nul.mean()); s = float(nul.std())
    d = {"hits": int(obs), "null_mean": round(m, 3), "null_sd": round(s, 3),
         "fold": round(obs / m, 3) if m > 0 else None,
         "z": round((obs - m) / s, 2) if s > 0 else None,
         "p_emp": round((int((nul >= obs).sum()) + 1) / (len(nul) + 1), 5),
         "n_perm": int(len(nul))}
    if n_pairs is not None:
        d = {"n_pairs": int(n_pairs), **d}
    return d


def bh(pvals):
    """Benjamini-Hochberg q-values (step-up), input list -> list"""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        prev = min(prev, p[i] * n / rank)
        q[i] = prev
    return [round(float(x), 5) for x in q]
