"""One place where gene symbols and feature catalogues are handled.

Why this exists. The same three operations -- read a catalogue, take a feature's top genes, map a
symbol to a publication count -- were written out separately in ~20 scripts, with small differences
that were invisible until they changed a result:

  * `NPUB.get(sym, 0)` conflated "symbol absent from gene_info" with "zero publications", so genes
    whose symbol had been renamed (H2AFX -> H2AX with 2,979 papers, TMEM173 -> STING1 with 2,104)
    landed in the *least studied* stratum. Those genes respond to almost any perturbation, which
    inflated that stratum to 1.49x. Use `GeneIndex.papers()`, which returns None when unmapped.
  * ENSG identifiers are 19.5% of Tahoe-x1's top-gene slots and 0% of five other models'. Taking
    top-N *before* dropping them gives Tahoe ~8.3 usable symbols per feature where UCE gets 10, so
    "top-10 genes" silently means different things per model. `top_genes()` drops them first.

Import it rather than re-deriving any of this:

    from genelib import GeneIndex, top_genes, catalogs, mid_catalog
"""
from __future__ import annotations
import glob, gzip, json, os
from collections import defaultdict

BASE = os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")
ALLCAT = f"{BASE}/outputs/atlas/alllayer_cat"
TS = f"{BASE}/outputs/atlas/ts3_out"
PUBMED = f"{BASE}/outputs/atlas/pubmed"

MODELS = ["AIDO", "C2S", "Geneformer", "MaxToki", "UCE",
          "scGPT", "tGPT", "scFoundation", "GeneCompass", "Tahoe"]
DISPLAY = {"AIDO": "AIDO.Cell", "C2S": "C2S-Scale", "Geneformer": "Geneformer-V2", "Tahoe": "Tahoe-x1"}


# ---------------------------------------------------------------- symbols ---
def norm(sym) -> str:
    """Canonical form of a gene symbol as stored in the catalogues."""
    return str(sym).strip().upper()


def is_ensembl(sym) -> bool:
    return norm(sym).startswith("ENSG")


def top_genes(feature: dict, n: int | None = None, dedup: bool = True) -> list[str]:
    """A feature's top genes: normalised, Ensembl ids dropped FIRST, then truncated to n.

    Dropping before truncating is what keeps `n` comparable across models -- see the module
    docstring. Order is preserved (it encodes attribution rank); `dedup` removes repeats.
    """
    out = [norm(g) for g in feature.get("top_genes", []) if not is_ensembl(g)]
    if dedup:
        out = list(dict.fromkeys(out))
    return out[:n] if n else out


# ------------------------------------------------------------- catalogues ---
def catalogs(model: str) -> list[str]:
    """Every layer catalogue for a model, ordered by layer. Prefers the all-layer extraction."""
    d = ALLCAT if glob.glob(f"{ALLCAT}/{model}/feature_catalog_L*.json") else TS
    paths = glob.glob(f"{d}/{model}/feature_catalog_L*.json")
    return sorted(paths, key=lambda p: json.load(open(p))["layer"])


def mid_catalog(model: str) -> str:
    """The mid-depth catalogue -- the layer every fixed-depth analysis reads."""
    c = catalogs(model)
    return c[len(c) // 2]


def features(path: str):
    """Iterate (feature_id, feature_dict) from one catalogue."""
    return json.load(open(path))["features"].items()


def layer_of(path: str) -> int:
    return json.load(open(path))["layer"]


def iter_features(model: str, all_layers: bool = True):
    """Iterate (layer, feature_id, feature_dict) over one model's catalogues."""
    paths = catalogs(model) if all_layers else [mid_catalog(model)]
    for p in paths:
        lay = layer_of(p)
        for fid, f in features(p):
            yield lay, fid, f


# ------------------------------------------------------------- literature ---
class GeneIndex:
    """Symbol -> NCBI GeneID -> publication count, tolerant of renamed symbols.

    `papers(sym)` returns an int, or **None** when the symbol maps to no NCBI gene (Ensembl-style
    clone ids such as AL450998.2). None is not zero: it means unknown, and callers must keep it in
    its own stratum rather than folding it into the least-studied one.
    """

    def __init__(self, taxid: str = "9606"):
        self.taxid = taxid
        self.current: dict[str, str] = {}
        self.synonym: dict[str, str] = {}
        self.biotype: dict[str, str] = {}
        self._papers: dict[str, int] = defaultdict(int)
        self._loaded = False

    def load(self):
        if self._loaded:
            return self
        with gzip.open(f"{PUBMED}/Homo_sapiens.gene_info.gz", "rt") as fh:
            hdr = fh.readline().rstrip("\n").split("\t")
            ti = hdr.index("type_of_gene") if "type_of_gene" in hdr else 9
            for line in fh:
                f = line.rstrip("\n").split("\t")
                if f[0] != self.taxid:
                    continue
                sym = norm(f[2])
                self.current.setdefault(sym, f[1])
                self.biotype.setdefault(sym, f[ti])
                for s in f[4].split("|"):
                    s = norm(s)
                    if s and s != "-":
                        self.synonym.setdefault(s, f[1])
        with gzip.open(f"{PUBMED}/gene2pubmed.gz", "rt") as fh:
            fh.readline()
            for line in fh:
                t, gid, _ = line.rstrip("\n").split("\t")
                if t == self.taxid:
                    self._papers[gid] += 1
        self._loaded = True
        return self

    def gene_id(self, sym) -> str | None:
        s = norm(sym)
        return self.current.get(s) or self.synonym.get(s)

    def papers(self, sym) -> int | None:
        """Publication count, or None when the symbol maps to no NCBI gene."""
        gid = self.gene_id(sym)
        return None if gid is None else self._papers.get(gid, 0)

    def type_of(self, sym) -> str:
        """Biotype, or 'unmapped' -- never silently 'protein-coding'."""
        s = norm(sym)
        if s in self.current:
            return self.biotype.get(s, "unknown")
        return "unmapped" if self.gene_id(s) is None else "renamed"

    def stratum(self, sym, edges=(25, 50, 100, 400)) -> str:
        """Publication stratum, with unmapped symbols kept separate from zero-publication ones."""
        p = self.papers(sym)
        if p is None:
            return "unmapped"
        lo = 0
        for e in edges:
            if p < e:
                return f"{lo}-{e - 1}"
            lo = e
        return f">={edges[-1]}"
