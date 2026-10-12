#!/usr/bin/env python
"""Held-out corpus with the main corpus's cells removed.

build_ts_corpus.py --seed 1 draws the held-out corpus independently of the seed-0 main corpus, so
the two share cells (419 of 6,000). The held-out replication needs cells no SAE was trained on, so
those shared cells are dropped here.

    python make_heldout_disjoint.py
        -> external/perturb/tabula_3tissue_6k_heldout_disjoint.h5ad
"""
from __future__ import annotations
import argparse
import anndata as ad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="external/perturb/tabula_3tissue_6k.h5ad")
    ap.add_argument("--heldout", default="external/perturb/tabula_3tissue_6k_heldout.h5ad")
    ap.add_argument("--out", default="external/perturb/tabula_3tissue_6k_heldout_disjoint.h5ad")
    a = ap.parse_args()
    main_ids = set(ad.read_h5ad(a.main, backed="r").obs_names)
    h = ad.read_h5ad(a.heldout)
    keep = ~h.obs_names.isin(main_ids)
    print(f"held-out {h.n_obs} cells, {int((~keep).sum())} shared with the main corpus, keeping {int(keep.sum())}")
    print(h.obs.loc[keep, "compartment"].value_counts().to_string())
    h[keep].copy().write_h5ad(a.out)
    print(f"==> {a.out}")


if __name__ == "__main__":
    main()
