#!/usr/bin/env python
"""Run the VALID regulatory-logic test (external CRISPRi perturbation-response) on any
model via its adapter. This is the headline cross-model job: is the 6.2% TF-logic null
universal, or does ESM prior (UCE) / knowledge prior (GeneCompass) / GRN design (scPRINT)
break it? Data: Replogle K562 CRISPRi h5ad (Zenodo rec 7041849, obs['perturbation']).

    python run_perturbation.py --model scPRINT --perturb data/replogle_k562_essential.h5ad \
        --trrust data/trrust_human.tsv --n-tfs 48 --n-kd 40 --device cuda
"""
from __future__ import annotations
import argparse, importlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_model import ADAPTERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(ADAPTERS))
    ap.add_argument("--perturb", required=True, help="Replogle CRISPRi h5ad")
    ap.add_argument("--trrust", required=True)
    ap.add_argument("--layer", type=int, default=None, help="default = middle of adapter.layers")
    ap.add_argument("--n-tfs", type=int, default=48)
    ap.add_argument("--n-kd", type=int, default=40)
    ap.add_argument("--n-ctrl", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="out/perturbation")
    args = ap.parse_args()

    import scanpy as sc
    from common.perturbation import run_perturbation

    mod, cls = ADAPTERS[args.model]
    adapter = getattr(importlib.import_module(mod), cls)()
    adapter.load(device=args.device)
    layer = args.layer if args.layer is not None else adapter.layers[len(adapter.layers) // 2]

    adata = sc.read_h5ad(args.perturb)
    print(f"==> {args.model}: perturbation-response at layer {layer} on {adata.n_obs} cells")
    run_perturbation(adapter, adata, args.trrust, layer, args.out,
                     n_tfs=args.n_tfs, n_kd=args.n_kd, n_ctrl=args.n_ctrl, device=args.device)


if __name__ == "__main__":
    main()
