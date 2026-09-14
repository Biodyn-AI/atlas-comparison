#!/usr/bin/env python
"""Phase-2 step 4-7 — multi-label annotation + external validation of novel candidates.
For each candidate SAE-feature signature (its top genes), it (1) MULTI-LABEL annotates
against the full KNOWN vocabulary (TRRUST + GO + Reactome + Hallmark) by top-gene
enrichment (hypergeometric + Bonferroni) — so a MIXED feature (glycolysis + T-cell) is
labelled with BOTH and correctly called 'known', not a false novel; and (2) for the
features that match NOTHING known, VALIDATES against EXTERNAL ground truth not in
pretraining (ChEA / ENCODE ChIP-seq + TF-perturbation signatures) — a match there =
an externally-supported candidate module (a real hypothesis); no match either = unexplained.

    conda activate scprint
    python scripts/mlp_validate.py                 # uses outputs/singlecell/novel_candidates.npz
    python scripts/mlp_validate.py --demo          # runs on example signatures (no npz needed)
"""

import os as _os
_B = _os.environ.get("ATLAS_BASE", "/Users/annaantipova/Desktop/biomech")   # set ATLAS_BASE to run this anywhere
import argparse, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = _B
KNOWN_DIRS = [f"{BASE}/external/genesets"]                    # GO/Reactome/Hallmark
EXTERNAL_DIR = f"{BASE}/external/genesets_external"          # ChEA / ENCODE ChIP + TF-perturbation
UNIVERSE = 20000                                             # ~ protein-coding genes

DEMO = {  # real candidate signatures observed in the last geneformer_novel run
    "feat1644": "ENO1,TPI1,TUBA1B,UQCRQ,NDUFB11,UBE2L3,MZT2B,PCBP1,LDHA,PGK1,ALDOA,GAPDH",
    "feat800":  "IL7R,TPI1,LDHB,CD3E,CCL5,CTSW,FCER1G,ARPC5,CD3D,GZMK,NKG7,CST7",
    "feat994":  "TAP1,RNH1,HSP90AA1,ATP6AP2,HLA-DRB1,HLA-DMA,CD74,B2M,PSMB9,TAPBP",
    "feat3919": "PSMB10,ZNHIT1,NDUFS7,CASP1,PSMA5,RNF181,PCMT1,PSMB8,PSMA7,PSME2",
    "feat_x":   "GENEA1,GENEB2,GENEC3,GENED4,GENEE5,GENEF6,GENEG7,GENEH8",  # deliberately fake -> unexplained
}


def load_gmts(dirs):
    fams = {}
    files = [f for d in ([dirs] if isinstance(dirs, str) else dirs) for f in glob.glob(f"{d}/*.gmt")]
    for gm in files:
        fam = os.path.basename(gm).replace(".gmt", "")
        for line in open(gm):
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            genes = frozenset(g.upper() for g in p[2:] if g)
            if 5 <= len(genes) <= 2000:
                fams[f"{fam}::{p[0]}"[:80]] = genes
    return fams


def enrich(query, sets, N=UNIVERSE, alpha=0.05, min_overlap=3):
    from scipy.stats import hypergeom
    q = set(query); nq = len(q); hits = []
    ntests = len(sets)
    for name, s in sets.items():
        k = len(q & s)
        if k < min_overlap:
            continue
        p = hypergeom.sf(k - 1, N, len(s), nq)
        padj = min(p * ntests, 1.0)
        if padj < alpha:
            hits.append((name, k, padj))
    return sorted(hits, key=lambda x: x[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--npz", default=f"{BASE}/outputs/singlecell/novel_candidates.npz")
    ap.add_argument("--top", type=int, default=3); A = ap.parse_args()
    import numpy as np

    print("==> loading vocabularies...")
    known = load_gmts(KNOWN_DIRS)
    # add TRRUST as TF-target sets
    import pandas as pd
    tr = pd.read_csv(f"{BASE}/external/single_cell_mechinterp/external/networks/trrust_human.tsv",
                     sep="\t", header=None, names=["tf", "tg", "m", "p"])
    for tf, g in tr.groupby(tr.tf.str.upper()):
        s = frozenset(x.upper() for x in g.tg)
        if len(s) >= 5:
            known[f"TRRUST::{tf}"] = s
    external = load_gmts(EXTERNAL_DIR)
    print(f"    KNOWN vocab: {len(known)} sets | EXTERNAL GT: {len(external)} sets (ChIP + TF-perturbation)")

    if A.demo:
        cands = [(k, v.split(",")) for k, v in DEMO.items()]
    else:
        d = np.load(A.npz, allow_pickle=True)
        cands = [(f"feat{int(f)}(L{int(l)})", s.split(","))
                 for f, l, s in zip(d["feats"], d["layer"], d["sigs"])]
    print(f"==> validating {len(cands)} candidate signatures\n")

    n_known = n_supported = n_unexplained = 0
    for name, genes in cands:
        kh = enrich(genes, known)
        if kh:
            n_known += 1
            labs = "; ".join(f"{h[0].split('::')[-1][:34]} (k={h[1]},p={h[2]:.1e})" for h in kh[:A.top])
            print(f"[KNOWN — multi-label]  {name}\n     {labs}")
        else:
            eh = enrich(genes, external)
            if eh:
                n_supported += 1
                labs = "; ".join(f"{h[0].split('::')[-1][:34]} (k={h[1]},p={h[2]:.1e})" for h in eh[:A.top])
                print(f"[NOVEL-to-vocab, EXTERNALLY SUPPORTED — candidate hypothesis]  {name}\n     ext: {labs}")
            else:
                n_unexplained += 1
                print(f"[UNEXPLAINED — no known or external match]  {name}  top: {', '.join(genes[:6])}")
    print(f"\n==> {n_known} KNOWN (multi-label), {n_supported} externally-supported hypotheses, "
          f"{n_unexplained} unexplained (of {len(cands)}).")
    print("    Shortlist = the EXTERNALLY-SUPPORTED set: novel vs our literature vocab but backed by "
          "ChIP/perturbation → real candidate modules to test.")


if __name__ == "__main__":
    main()
