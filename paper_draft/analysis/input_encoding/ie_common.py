"""Shared loaders for the input-encoding analysis (notes/11_input_encoding.md).

Everything reads the atlas release files (ATLAS_BASE = repository root with the release archives
unpacked by pipeline/scripts/fetch_runs.sh; defaults to the root of this checkout). Catalogue choice, top-10 cut, upper-casing, Ensembl-id drop and the
>= 2 genes rule are copied from pipeline/scripts/hypothesis_trrust3.py:build() so that our numbers
are on the same ruler as the published co-firing test.

New here:
  * model universe U_m = python-sorted set of the raw gene strings seen in ANY top-20 list of ANY
    layer of model m. pipeline/atlas_h100/common/pipeline.py:build_catalog indexes genes by
    `uniq = np.array(sorted(set(syms)))` (the same python string order) and picks top genes by
    `np.argsort(G[:, f])[::-1][:20]`, so ranks in U_m are the catalogue's own index order
    (U_m is a subset of the true index: genes never in any top-20 are missing; coverage is reported).
  * tie-tail detector: a run of alphabetical neighbours (in U_m) at the END of a feature's top-20
    list. A feature that fires on fewer than 20 genes has per-gene mean exactly 0 for all other genes;
    argsort breaks that tie by index position, i.e. by alphabetical position.
"""
from __future__ import annotations

import csv, glob, json, os, pickle, re
import numpy as np

_ANALYSIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # paper_draft/analysis
# repository root with the release archives unpacked (pipeline/scripts/fetch_runs.sh)
BASE = os.environ.get("ATLAS_BASE", os.path.abspath(os.path.join(_ANALYSIS, "..", "..")))
# folder for intermediate files
CACHE = os.environ.get("ATLAS_CACHE", os.path.join(_ANALYSIS, ".cache"))
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"; TS = f"{BASE}/outputs/atlas/ts3_out"
C = f"{BASE}/outputs/atlas/comparative"; G = f"{BASE}/outputs/atlas/genesets"
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
PASSED = ["Tahoe", "scGPT", "UCE", "C2S", "Geneformer"]
TOPG, W0 = 10, 5
OUTDIR = os.path.dirname(os.path.abspath(__file__))
SCR = os.path.join(CACHE, "input_encoding")
os.makedirs(SCR, exist_ok=True)

# HGNC complete set (https://www.genenames.org/download/)
HGNC_PATH = os.environ.get("HGNC_PATH", os.path.join(CACHE, "hgnc_complete_set.txt"))
HF = os.path.expanduser("~/.cache/huggingface/hub")
TGPT_TOK = f"{HF}/models--lixiangchun--transcriptome-gpt-1024-8-16-64/snapshots/29e29c6aba21f69fe510333fba5dde7447732a24/tokenizer.json"
GF_DIR = f"{HF}/models--ctheodoris--Geneformer/snapshots/ad8f66dfcda3ebbd148d916c01f31339c5b95a15/geneformer"
# vocab.json of the scGPT whole-human checkpoint
SCGPT_VOCAB = os.environ.get("SCGPT_VOCAB", os.path.join(CACHE, "scgpt_vocab.json"))
# gene order of the 19,264-gene panel (OS_scRNA_gene_index.19264.tsv, AIDO.Cell / scFoundation)
SF_INDEX = os.environ.get("AIDO_GENE_INDEX", os.path.join(CACHE, "OS_scRNA_gene_index.19264.tsv"))
UCE_KEYS = f"{SCR}/uce_esm2_keys.json"

GAP, MINBLOCK, TAILFRAC = 2, 3, 0.75   # tie-tail detector: sorted ranks within GAP link; blocks of >= MINBLOCK genes


def cat_dir(m):
    return ALLCAT if glob.glob(f"{ALLCAT}/{m}/feature_catalog_L*.json") else TS


def catalog_paths(m):
    return sorted(glob.glob(f"{cat_dir(m)}/{m}/feature_catalog_L*.json"))


def mid_layer(m):
    """stats_final.py:feats_mid - catalogues sorted by layer, take cats[len(cats)//2]."""
    ps = catalog_paths(m)
    return int(re.search(r"_L(\d+)\.json$", ps[len(ps) // 2]).group(1))


def est_npos(freq):
    """number of token positions the SAE was run on: smallest N with freq*N integral for all features."""
    f = np.unique(freq[freq > 0]); fmin = f[0]
    for c in range(1, 3000):
        N = c / fmin; x = f * N
        if np.all(np.abs(x - np.round(x)) < 1e-3 * np.maximum(1, x * 1e-3)):
            return int(round(N))
    return None


def load_model(m, cache=True):
    """-> dict(names=[raw str], T20=int32[F,20] (-1 pad), layer=int16[F], freq=f64[F], nfire=int64[F] (-1 unknown),
    npos={layer: N})"""
    cp = f"{SCR}/load_{m}.pkl"
    if cache and os.path.exists(cp):
        return pickle.load(open(cp, "rb"))
    sid = {}; names = []; rows = []; lay = []; fr = []; nf = []; npos = {}
    for p in catalog_paths(m):
        c = json.load(open(p)); L = int(c["layer"])
        feats = list(c["features"].values())
        f_ = np.array([f["activation_frequency"] for f in feats], float)
        N = est_npos(f_); npos[L] = N
        for f in feats:
            r = []
            for g in f.get("top_genes", [])[:20]:
                g = str(g)
                if g not in sid: sid[g] = len(names); names.append(g)
                r.append(sid[g])
            rows.append(r + [-1] * (20 - len(r)))
        lay += [L] * len(feats); fr.append(f_)
        nf.append(np.round(f_ * N).astype(np.int64) if N else np.full(len(feats), -1, np.int64))
        del c
    d = {"names": names, "T20": np.array(rows, np.int32), "layer": np.array(lay, np.int16),
         "freq": np.concatenate(fr), "nfire": np.concatenate(nf), "npos": npos}
    if cache: pickle.dump(d, open(cp, "wb"))
    return d


def universe_ranks(names):
    """rank of each raw name in the python-sorted universe (catalogue index order)."""
    order = sorted(range(len(names)), key=lambda i: names[i])
    rk = np.empty(len(names), np.int64); rk[np.array(order)] = np.arange(len(names))
    return rk


def tie_tail(T20, rank):
    """length of the trailing run (positions 19, 18, ...) whose genes sit in an alphabetical block:
    a group of >= MINBLOCK genes of the same top-20 list whose universe ranks chain with gaps <= GAP."""
    F, K = T20.shape
    valid = T20 >= 0
    R = np.where(valid, rank[np.maximum(T20, 0)], np.int64(1) << 40)
    o = np.argsort(R, axis=1, kind="stable"); Rs = np.take_along_axis(R, o, 1)
    brk = np.ones((F, K), bool); brk[:, 1:] = (Rs[:, 1:] - Rs[:, :-1]) > GAP
    comp = np.cumsum(brk, 1)                                   # component label per sorted slot
    size = np.zeros((F, K + 1), np.int64)
    np.add.at(size, (np.repeat(np.arange(F), K), comp.ravel()), 1)
    inb_sorted = np.take_along_axis(size, comp, 1) >= MINBLOCK
    inb_sorted &= np.take_along_axis(valid, o, 1)
    inb = np.empty_like(inb_sorted); np.put_along_axis(inb, o, inb_sorted, 1)
    # tail start p* = first position p that is a block gene and from which >= TAILFRAC of the
    # remaining slots p..19 are block genes (tolerates the odd tied gene from another index region)
    suf = np.cumsum(inb[:, ::-1], 1)[:, ::-1]                    # block genes in p..K-1
    frac = suf / (K - np.arange(K))[None, :]
    cond = inb & (frac >= TAILFRAC) & ((K - np.arange(K))[None, :] >= MINBLOCK)
    first = np.where(cond.any(1), cond.argmax(1), K)
    tail = K - first
    return tail.astype(np.int64), inb


def tie_tail_with_firing(T20, rank, nfire):
    """block rule, united with the hard bound from firing counts: a feature that fired on n < 20
    positions has at most n genes with non-zero mean, so slots n..19 are certainly zero-ties."""
    tail, inb = tie_tail(T20, rank)
    known = (nfire > 0) & (nfire < 20)
    hard = np.where(known, 20 - nfire, 0)
    return np.maximum(tail, hard), tail, inb


def upper_map(names):
    """raw id -> upper-case id (uid), plus uid names."""
    un = {}; unames = []; m = np.empty(len(names), np.int64)
    for i, g in enumerate(names):
        u = g.upper()
        if u not in un: un[u] = len(unames); unames.append(u)
        m[i] = un[u]
    return m, unames


def filtered_top10(T20, rawmap_u, unames, keep_mask=None):
    """hypothesis_trrust3.build filter on the first 10 slots: upper-case, drop ENSG*, unique, >= 2 genes.
    keep_mask (F,20 bool) optionally blanks slots (e.g. tie tails) BEFORE the filter.
    -> X int64[F', 10] of uids (-1 pad, sorted per row), idx of kept features."""
    T = T20[:, :TOPG].copy()
    if keep_mask is not None: T[~keep_mask[:, :TOPG]] = -1
    isens = np.array([u.startswith("ENSG") for u in unames])
    U = np.where(T >= 0, rawmap_u[np.maximum(T, 0)], -1)
    U[(U >= 0) & isens[np.maximum(U, 0)]] = -1
    BIG = np.int64(1) << 40
    S = np.sort(np.where(U >= 0, U, BIG), 1)
    dup = np.zeros_like(S, bool); dup[:, 1:] = S[:, 1:] == S[:, :-1]
    S[dup] = BIG
    S = np.sort(S, 1); S[S == BIG] = -1
    n = (S >= 0).sum(1); keep = np.where(n >= 2)[0]
    return S[keep], keep


IU, JU = np.triu_indices(TOPG, 1)


def pairs_of(X):
    A = X[:, IU].ravel(); B = X[:, JU].ravel(); ok = (A >= 0) & (B >= 0)
    return A[ok], B[ok]


def slot_permute(X, rng, max_iter=200):
    """bipartite null: shuffle gene slots across features, keeping every gene's count and every
    feature's size exactly; within-feature duplicates removed by random swaps."""
    Y = X.copy(); v = Y >= 0
    vals = Y[v]; rng.shuffle(vals); Y[v] = vals
    pos = np.flatnonzero(v.ravel())
    flat = Y.ravel()
    for it in range(max_iter):
        BIG = np.int64(1) << 40
        Z = np.where(Y >= 0, Y, BIG + np.arange(Y.shape[1]))
        o = np.argsort(Z, 1); Zs = np.take_along_axis(Z, o, 1)
        d = np.zeros_like(Zs, bool); d[:, 1:] = Zs[:, 1:] == Zs[:, :-1]
        if not d.any(): break
        rr, cc = np.nonzero(d)
        bad = rr * Y.shape[1] + o[rr, cc]
        tgt = pos[rng.integers(0, len(pos), len(bad))]
        a = flat[bad].copy(); flat[bad] = flat[tgt]; flat[tgt] = a
    return Y


# ---------------------------------------------------------------- name-level annotations
def hgnc():
    """symbol(upper) -> (hgnc approved symbol, [group ids], ensembl id). Approved symbols first, then
    unique previous / alias symbols that are not approved symbols of another gene."""
    appr = {}; alias = {}; ens2sym = {}
    with open(HGNC_PATH, newline="") as fh:
        r = csv.DictReader(fh, delimiter="\t")
        rows = list(r)
    for x in rows:
        s = x["symbol"].upper()
        gids = [int(t) for t in x["gene_group_id"].strip('"').split("|") if t.strip().isdigit()]
        appr[s] = (x["symbol"], gids, x["ensembl_gene_id"])
        if x["ensembl_gene_id"]: ens2sym[x["ensembl_gene_id"]] = x["symbol"]
    for x in rows:
        for col in ("prev_symbol", "alias_symbol"):
            for a in x[col].strip('"').split("|"):
                a = a.strip().upper()
                if a and a not in appr: alias.setdefault(a, set()).add(x["symbol"].upper())
    out = dict(appr)
    for a, ss in alias.items():
        if len(ss) == 1: out[a] = appr[next(iter(ss))]
    ver = max(x["date_modified"] for x in rows if x["date_modified"])
    return out, ens2sym, {"n_rows": len(rows), "max_date_modified": ver}


def token_orders():
    """model -> {symbol_upper: token id} for the models whose real vocabulary is on disk."""
    out = {}
    t = json.load(open(TGPT_TOK))["model"]["vocab"]
    out["tGPT"] = {k.upper(): int(v) for k, v in t.items()}                 # adapter upper-cases vocab keys
    td = pickle.load(open(f"{GF_DIR}/token_dictionary_gc104M.pkl", "rb"))
    nd = pickle.load(open(f"{GF_DIR}/gene_name_id_dict_gc104M.pkl", "rb"))
    gf = {}
    for s, e in nd.items():
        if e in td: gf.setdefault(str(s).upper(), int(td[e]))
    out["Geneformer"] = gf; out["MaxToki"] = gf                            # shared 20,275 vocab (adapters/maxtoki.py)
    out["scGPT"] = {k.upper(): int(v) for k, v in json.load(open(SCGPT_VOCAB)).items()}
    sf = {}
    with open(SF_INDEX) as fh:
        next(fh)
        for line in fh:
            g, i = line.rstrip("\n").split("\t"); sf.setdefault(g.upper(), int(i))
    out["scFoundation"] = sf; out["AIDO"] = sf
    if os.path.exists(UCE_KEYS):
        out["UCE"] = {k.upper(): i for i, k in enumerate(json.load(open(UCE_KEYS)))}
    return out


TOKEN_SOURCE = {
    "tGPT": "tokenizer.json of lixiangchun/transcriptome-gpt-1024-8-16-64 (WordLevel; ids in corpus-frequency order)",
    "Geneformer": "token_dictionary_gc104M.pkl (ids in Ensembl-id order) via gene_name_id_dict_gc104M.pkl",
    "MaxToki": "same gc104M dictionaries as Geneformer (adapters/maxtoki.py: shared 20,275 vocab)",
    "scGPT": "projects/biotensor/data/scgpt_human/vocab.json (60,697 tokens; assumed = whole-human vocab)",
    "scFoundation": "OS_scRNA_gene_index.19264.tsv gene position index",
    "AIDO": "OS_scRNA_gene_index.19264.tsv gene position index (same list as scFoundation)",
    "UCE": "key order of Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt (UCE per-species token order)",
}


def levenshtein_norm(CA, LA, CB, LB):
    """vectorised Levenshtein distance / max(len) for char-code arrays [P, L] with lengths."""
    P, L = CA.shape
    prev = np.tile(np.arange(L + 1, dtype=np.int16), (P, 1))
    res = np.zeros(P, np.int16)
    for i in range(1, L + 1):
        cur = np.empty_like(prev); cur[:, 0] = i
        ai = CA[:, i - 1][:, None]
        sub = prev[:, :-1] + (ai != CB).astype(np.int16)
        dele = prev[:, 1:] + 1
        best = np.minimum(sub, dele)
        for j in range(1, L + 1):
            cur[:, j] = np.minimum(best[:, j - 1], cur[:, j - 1] + 1)
        hit = LA == i
        if hit.any(): res[hit] = cur[hit, LB[hit]]
        prev = cur
    return res / np.maximum(LA, LB)


def char_codes(unames, L=16):
    C_ = np.zeros((len(unames), L), np.uint8); ln = np.zeros(len(unames), np.int64)
    for i, g in enumerate(unames):
        b = g.encode("ascii", "replace")[:L]; C_[i, :len(b)] = np.frombuffer(b, np.uint8); ln[i] = max(len(b), 1)
    return C_, ln


def empirical(obs, null, greater=True):
    null = np.asarray(null, float); mu = float(null.mean()); sd = float(null.std())
    if greater:
        p = (int((null >= obs).sum()) + 1) / (len(null) + 1)
    else:
        p = (int((null <= obs).sum()) + 1) / (len(null) + 1)
    z = (obs - mu) / sd if sd > 0 else None
    return {"obs": round(float(obs), 6), "null_mean": round(mu, 6), "null_sd": round(sd, 6),
            "fold": round(obs / mu, 3) if mu > 0 else None, "z": round(z, 2) if z is not None else None,
            "p_emp": round(p, 5), "n_perm": len(null), "direction": "greater" if greater else "less"}
