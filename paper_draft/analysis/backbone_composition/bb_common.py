#!/usr/bin/env python
"""Shared code for the backbone-composition analysis (notes 13).

The calibrated annotator here is copied from the repo so that every number is on the same ruler
as the published ones:
  * `annotate_exact`  = recalibrate_final.py:39-56 / backbone_terms.py:43-59 / stats_final.py:45-62,
                        line for line (top-10 genes & background, >=3 in a GO_BP/KEGG/Reactome set of
                        5..MX genes, hypergeometric upper tail, BH over the candidate pairs, q<=0.05).
  * `annotate_detail` = the same rule, but it also keeps WHICH feature passed for WHICH term and
                        which of its top-10 genes fell in the set (the "hit genes"). Checked against
                        `annotate_exact` on every model before use.
  * `annotate_fast`   = depth_backbone_fast.py:56-70 (sparse incidence product; the repo verified it
                        gives the same concepts). Used only for the random-gene nulls.
  * random genes      = BG_ARR[rng.integers(0, BGN, TOP)] per feature, rng = default_rng(seed), exactly
                        as recalibrate_final.py:80 / stats_final.py:92 / depth_backbone_fast.py:119.
  * layer choice      = headline mid: cats[len(cats)//2] (recalibrate_final.py:64);
                        depth slice f: cats[round(f*(n-1))] (depth_backbone_fast.py:85).
Catalogue root: alllayer_cat if the model has one, else ts3_out (recalibrate_final.py:59-63).

Inputs: ATLAS_BASE = repository root with the release archives unpacked (pipeline/scripts/fetch_runs.sh);
defaults to the root of this checkout. ATLAS_CACHE = folder for intermediate files.
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp
from scipy.stats import hypergeom

_ANALYSIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # paper_draft/analysis
# repository root with the release archives unpacked (pipeline/scripts/fetch_runs.sh)
BASE = os.environ.get("ATLAS_BASE", os.path.abspath(os.path.join(_ANALYSIS, "..", "..")))
# folder for intermediate files
CACHE = os.environ.get("ATLAS_CACHE", os.path.join(_ANALYSIS, ".cache"))
G = f"{BASE}/outputs/atlas/genesets"
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"
TS = f"{BASE}/outputs/atlas/ts3_out"
ORIG_COMPARATIVE = f"{BASE}/outputs/atlas/comparative"
OUT = os.path.dirname(os.path.abspath(__file__))

ALPHA, TOP, AMIN = 0.05, 10, 3
ORDER = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE", "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
# adapters where layer index 0 is the raw token embedding (notes 03 section 1.2, notes 07 section 2.6)
EMBED_L0 = ["AIDO", "tGPT", "GeneCompass"]

RAW = {src: {t: set(x.upper() for x in g) for t, g in json.load(open(f"{G}/{src}_gene_sets.json")).items()}
       for src in ("GO_BP", "KEGG", "Reactome")}
BG = set(x.upper() for x in json.load(open(f"{G}/background.json")))
BGN = len(BG)
BG_ARR = np.array(sorted(BG))
GIX = {g: i for i, g in enumerate(BG_ARR.tolist())}


def bh(p):  # recalibrate_final.py:34-36
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = np.arange(1, n + 1)
    q = np.minimum.accumulate((p[o] * n / r)[::-1])[::-1]; out = np.empty(n); out[o] = np.clip(q, 0, 1); return out


class Annotator:
    """Calibrated annotator at a given set-size cap MX (5 <= |set| <= MX)."""

    def __init__(self, mx=200, top=TOP, amin=AMIN):
        self.mx, self.top, self.amin = mx, top, amin
        self.gs = {}
        for src in ("GO_BP", "KEGG", "Reactome"):
            for t, g in RAW[src].items():
                if 5 <= len(g) <= mx:
                    self.gs[f"{src}:{t}"] = g
        self.idx = defaultdict(list); self.size = {}
        for t, genes in self.gs.items():
            self.size[t] = len(genes)
            for g in genes:
                self.idx[g].append(t)
        # sparse incidence (depth_backbone_fast.py:42-48)
        self.TERMS = sorted(self.gs)
        self.TSIZE = np.array([len(self.gs[t]) for t in self.TERMS], dtype=np.int64)
        r, c = [], []
        for ti, t in enumerate(self.TERMS):
            for g in self.gs[t]:
                j = GIX.get(g)
                if j is not None:
                    r.append(j); c.append(ti)
        self.M = sp.csr_matrix((np.ones(len(r), np.int32), (r, c)), shape=(len(BG_ARR), len(self.TERMS)))

    # ---- exact copy of recalibrate_final.annotate ----
    def annotate_exact(self, feat_lists):
        A = []; TG = []; LG = []; TERM = []
        for genes in feat_lists:
            g = set(str(x).upper() for x in genes[:self.top]) & BG
            if len(g) < 3:
                continue
            cand = Counter()
            for gene in g:
                for k in self.idx.get(gene, ()):
                    cand[k] += 1
            for k, a in cand.items():
                if a < self.amin:
                    continue
                A.append(a); TG.append(self.size[k]); LG.append(len(g)); TERM.append(k)
        if not A:
            return set()
        q = bh(hypergeom.sf(np.array(A) - 1, BGN, np.array(TG), np.array(LG)))
        return set(TERM[i] for i in np.where(q <= ALPHA)[0])

    # ---- same rule, keeps feature and hit genes ----
    def annotate_detail(self, feat_lists):
        """returns {term: [(feature_index, (hit genes sorted)), ...]} for passing (feature, term) pairs"""
        A = []; TG = []; LG = []; TERM = []; FI = []; HITS = []
        for fi, genes in enumerate(feat_lists):
            g = set(str(x).upper() for x in genes[:self.top]) & BG
            if len(g) < 3:
                continue
            cand = Counter()
            for gene in g:
                for k in self.idx.get(gene, ()):
                    cand[k] += 1
            for k, a in cand.items():
                if a < self.amin:
                    continue
                A.append(a); TG.append(self.size[k]); LG.append(len(g)); TERM.append(k); FI.append(fi)
                HITS.append(tuple(sorted(g & self.gs[k])))
        out = defaultdict(list)
        if not A:
            return out
        q = bh(hypergeom.sf(np.array(A) - 1, BGN, np.array(TG), np.array(LG)))
        for i in np.where(q <= ALPHA)[0]:
            out[TERM[i]].append((FI[i], HITS[i]))
        return out

    # ---- depth_backbone_fast.annotate ----
    def annotate_fast(self, feat_lists):
        rows, cols = [], []; lg = np.zeros(len(feat_lists), np.int64)
        for i, genes in enumerate(feat_lists):
            g = set(str(x).upper() for x in genes[:self.top]) & BG
            if len(g) < 3: continue
            lg[i] = len(g)
            for x in g: rows.append(i); cols.append(GIX[x])
        if not rows: return set()
        F = sp.csr_matrix((np.ones(len(rows), np.int32), (rows, cols)), shape=(len(feat_lists), len(BG_ARR)))
        A = (F @ self.M).tocoo(); keep = A.data >= self.amin
        if not keep.any(): return set()
        a = A.data[keep].astype(np.int64); fi = A.row[keep]; ti = A.col[keep]
        q = bh(hypergeom.sf(a - 1, BGN, self.TSIZE[ti], lg[fi]))
        return set(self.TERMS[j] for j in np.unique(ti[q <= ALPHA]))


def cats_of(m):
    d = ALLCAT if glob.glob(f"{ALLCAT}/{m}/feature_catalog_L*.json") else TS
    paths = glob.glob(f"{d}/{m}/feature_catalog_L*.json")
    return sorted(paths, key=lambda p: int(re.search(r"_L(\d+)\.json$", p).group(1)))


def load_cat(path):
    cat = json.load(open(path))
    return [x["top_genes"] for x in cat["features"].values()], cat["layer"]


def mid_path(m):
    cats = cats_of(m); return cats[len(cats) // 2]


def depth_path(m, f):
    cats = cats_of(m); return cats[int(round(f * (len(cats) - 1)))]


def layer_path(m, L):
    for p in cats_of(m):
        if int(re.search(r"_L(\d+)\.json$", p).group(1)) == L:
            return p
    raise FileNotFoundError(f"{m} L{L}")


def count_models(sets):
    cm = Counter()
    for s in sets.values():
        for t in s:
            cm[t] += 1
    return cm


def tiers(cm, ks=(7, 8, 9, 10)):
    return {f">={k}": sum(1 for v in cm.values() if v >= k) for k in ks}


def random_feats(rng, n):  # recalibrate_final.py:80
    return [BG_ARR[rng.integers(0, BGN, TOP)].tolist() for _ in range(n)]


# ---------------- gene-content classes ----------------
# Protein-synthesis machinery T (defined BEFORE looking at the concept classifications):
#   (a) cytosolic ribosomal proteins by HGNC symbol: ^RPL\d / ^RPS\d / ^RPLP[0-2] / RPSA / FAU / UBA52
#       (excludes the S6 kinases RPS6K*, and *BP* binding proteins)
#   (b) translation initiation / elongation factors by symbol: ^EIF\d, ^EEF\d
#   (c) every gene in GO_BP "ribosome biogenesis (GO:0042254)"  (raw file, no size cap)
#   (d) every gene in GO_BP "nuclear-transcribed mRNA catabolic process, nonsense-mediated decay (GO:0000184)"
RP_RE = re.compile(r"^(RPL\d+[A-Z]?\d*[A-Z]?\d*|RPS\d+[A-Z]?\d*[A-Z]?\d*|RPLP[0-2]|RPSA)$")
TF_RE = re.compile(r"^(EIF\d|EEF\d)")
GO_RIBOBIO = "ribosome biogenesis (GO:0042254)"
GO_NMD = "nuclear-transcribed mRNA catabolic process, nonsense-mediated decay (GO:0000184)"


def is_rp(g):
    return bool(RP_RE.match(g)) and not g.startswith("RPS6K") and "BP" not in g or g in ("FAU", "UBA52")


RP_GENES = set(g for g in BG if is_rp(g))
TF_GENES = set(g for g in BG if TF_RE.match(g))
RIBOBIO_GENES = RAW["GO_BP"][GO_RIBOBIO]
NMD_GENES = RAW["GO_BP"][GO_NMD]
T_STRICT = RP_GENES | TF_GENES                      # symbol rule only
T_GENES = T_STRICT | RIBOBIO_GENES | NMD_GENES      # primary rule

# Other housekeeping families O (by symbol), for a broader "housekeeping" class:
O_RULES = {
    "oxphos_mito": r"^(NDUF|SDH[ABCD]$|UQCR|CYC1$|CYCS$|COX\d|COX\d+[A-Z]\d*$|ATP5|MT-)",
    "mito_ribosome": r"^MRP[LS]\d",
    "proteasome": r"^PSM[A-G]\d",
    "histone": r"^(H1-|H2A|H2B|H3-|H3C|H4C|H4-|HIST)",
    "chaperone": r"^(HSPA\d|HSPB\d|HSPD1|HSPE1|HSPH1|HSP90|DNAJ[ABC]\d|CCT\d|TCP1$)",
    "spliceosome": r"^(SNRP|SF3[AB]\d|PRPF\d|SRSF\d|HNRNP)",
    "replication_mitosis": r"^(MCM\d$|PCNA$|TOP2A$|MKI67$|CDK1$|CCNA2$|CCNB[12]$|CDC20$|BUB1B?$|AURK[AB]$|PLK1$|TYMS$|RRM2$|UBE2C$|CENP[AEF]$|TUBB$|TUBA1B$|STMN1$)",
}
O_GENES = {k: set(g for g in BG if re.match(v, g)) for k, v in O_RULES.items()}
O_ALL = set().union(*O_GENES.values()) - T_GENES


# Descriptive gene-family labels (symbol rules), used only to NAME programmes by content. Checked in order.
LABEL_RULES = [
    ("translation_machinery", None),  # T_GENES
    ("MHC_class_II", r"^(HLA-D|CD74$)"),
    ("MHC_class_I", r"^(HLA-[ABCEFG]$|B2M$)"),
    ("histone", O_RULES["histone"]),
    ("oxphos_mito", O_RULES["oxphos_mito"]),
    ("proteasome", O_RULES["proteasome"]),
    ("heat_shock_chaperone", O_RULES["chaperone"]),
    ("metallothionein", r"^(MT1[A-Z]?$|MT2A$|MT3$|MT4$)"),
    ("complement", r"^(C1Q[ABC]$|C[2-9]$|CFB$|CFD$|C1R$|C1S$)"),
    ("muscle_contractile", r"^(MYL\d|MYH\d|TPM\d|TNN[TIC]\d|ACTA2$|ACTG2$|ACTC1$|DES$|CNN1$)"),
    ("chemokine", r"^(CXCL\d|CCL\d|CXCR\d|CCR\d)"),
    ("interferon_ISG", r"^(IFI|IFITM|MX[12]$|ISG|OAS|XAF1$|IRF\d)"),
    ("immediate_early_AP1", r"^(FOS|JUN|EGR\d$|ATF3$|IER\d)"),
    ("S100_alarmin", r"^S100"),
    ("cathepsin", r"^CTS[A-Z]$"),
    ("keratin", r"^KRT\d"),
    ("calmodulin_Ca_pump", r"^(CALM\d$|ATP2[AB]\d$|ATP1[AB]\d$)"),
    ("salivary_antimicrobial", r"^(HTN\d$|PRB\d$|PRH\d$|STATH$|DEFA\d|DEFB\d|LYZ$|LTF$|SMR3|MUC7$|JCHAIN$)"),
    ("plasma_protein", r"^(FG[ABG]$|APO[A-H]\d?$|AHSG$|ALB$|SERPINA\d)"),
    ("cell_cycle", O_RULES["replication_mitosis"]),
]


def label_gene(g):
    if g in T_GENES: return "translation_machinery"
    for name, rx in LABEL_RULES[1:]:
        if re.match(rx, g): return name
    return "other"


def dominant_label(hitcounter):
    """(label, weighted share) of the most common non-'other' label, and the share of 'other'"""
    lab = Counter()
    tot = sum(hitcounter.values())
    for g, c in hitcounter.items():
        lab[label_gene(g)] += c
    other = lab.pop("other", 0)
    if not lab: return "other", 0.0, round(other / max(tot, 1), 3)
    k, v = lab.most_common(1)[0]
    if v / max(tot, 1) < 0.25:   # no labelled family carries a quarter of the hits
        return "mixed", round(v / max(tot, 1), 3), round(other / max(tot, 1), 3)
    return k, round(v / max(tot, 1), 3), round(other / max(tot, 1), 3)


def gene_family(g):
    if g in RP_GENES: return "ribosomal_protein"
    if g in TF_GENES: return "EIF_EEF"
    if g in T_GENES: return "ribosome_biogenesis_or_NMD"
    for k, s in O_GENES.items():
        if g in s: return k
    return "other"


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)


def cluster_terms(terms, gene_sets, thr, method="complete"):
    """Cluster `terms` by Jaccard of gene_sets[t]; clusters whose pairs all (complete) / on average (average) /
    via a chain (single) have Jaccard >= thr. Returns list of clusters (lists of terms), largest first."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    terms = list(terms)
    n = len(terms)
    if n == 0: return []
    if n == 1: return [terms]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = 1.0 - jaccard(gene_sets[terms[i]], gene_sets[terms[j]])
    Z = linkage(squareform(D, checks=False), method=method)
    lab = fcluster(Z, t=1.0 - thr + 1e-12, criterion="distance")
    cl = defaultdict(list)
    for t, l in zip(terms, lab):
        cl[l].append(t)
    return sorted(cl.values(), key=lambda c: (-len(c), sorted(c)[0]))


def weighted_jaccard(ca, cb):
    """Ruzicka similarity of two Counters: sum(min)/sum(max)"""
    keys = set(ca) | set(cb)
    num = sum(min(ca.get(k, 0), cb.get(k, 0)) for k in keys)
    den = sum(max(ca.get(k, 0), cb.get(k, 0)) for k in keys)
    return num / den if den else 0.0


def cluster_matrix(terms, simfun, thr, method="complete"):
    """Like cluster_terms but with an arbitrary pairwise similarity simfun(a, b) in [0, 1]."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    terms = list(terms); n = len(terms)
    if n == 0: return []
    if n == 1: return [terms]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = 1.0 - simfun(terms[i], terms[j])
    Z = linkage(squareform(D, checks=False), method=method)
    lab = fcluster(Z, t=1.0 - thr + 1e-12, criterion="distance")
    cl = defaultdict(list)
    for t, l in zip(terms, lab):
        cl[l].append(t)
    return sorted(cl.values(), key=lambda c: (-len(c), sorted(c)[0]))


def dump(obj, name):
    p = f"{OUT}/{name}"
    json.dump(obj, open(p, "w"), indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    return p
