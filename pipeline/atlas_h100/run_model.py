#!/usr/bin/env python
"""Run one model end-to-end on H100: adapter -> activations -> TopK SAE -> catalog
-> annotation. Outputs under out/<model>/. Validate on AIDO first (trusted forward),
then add UCE/GeneCompass/etc.

    python run_model.py --model AIDO --corpus data/corpus.h5ad --device cuda
    python run_model.py --model AIDO --smoke        # 32 cells, quick end-to-end check
"""
from __future__ import annotations
import argparse, importlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ADAPTERS = {
    "AIDO":        ("adapters.aido", "AIDOAdapter"),
    "scPRINT":     ("adapters.scprint", "ScPrintAdapter"),
    "tGPT":        ("adapters.tgpt", "TGPTAdapter"),         # written from repo (VERIFY on H100)
    "UCE":         ("adapters.uce", "UCEAdapter"),           # written from repo (VERIFY on H100)
    "scGPT":       ("adapters.scgpt", "ScGPTAdapter"),       # Igor's model — for reproducing his 6.2% with our pipe
    "Geneformer":  ("adapters.geneformer", "GeneformerAdapter"),  # V2-316M; rank-value BertForMaskedLM (tested on V1)
    "MaxToki":     ("adapters.maxtoki", "MaxTokiAdapter"),   # theodoris-lab, gene-token (Apache)
    "Tahoe":       ("adapters.tahoe", "TahoeAdapter"),       # Tahoe-x1, transformer MLM
    "C2S":         ("adapters.c2s", "C2SAdapter"),           # C2S-Scale-2B, Gemma-2 cell-sentence
    "scBERT":      ("adapters.scbert", "ScBERTAdapter"),        # written from repo (VERIFY on H100)
    "GeneCompass": ("adapters.genecompass", "GeneCompassAdapter"),  # written from repo (VERIFY — most speculative)
    "scFoundation": ("adapters.scfoundation", "ScFoundationAdapter"),  # xTrimoGene MAE (VERIFY on H100)
    "CellPLM":     ("adapters._stubs", "CellPLMAdapter"),       # recipe only (cell-token — needs separate treatment)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(ADAPTERS))
    ap.add_argument("--corpus", default="data/corpus.h5ad")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="out")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--geneset-dir", default="data/genesets")
    ap.add_argument("--smoke", action="store_true", help="32 cells, quick sanity run")
    ap.add_argument("--all-layers", action="store_true", help="extract + SAE at EVERY layer (not just 5 depth-matched)")
    ap.add_argument("--max-positions", type=int, default=4_000_000,
                    help="cap saved positions/layer (use ~500000 for all-layers to keep disk sane)")
    ap.add_argument("--skip-annotate", action="store_true")
    args = ap.parse_args()

    import scanpy as sc
    from common import pipeline
    from common.annotate import annotate_all

    mod, cls = ADAPTERS[args.model]
    adapter = getattr(importlib.import_module(mod), cls)()
    adapter.load(device=args.device)

    if args.all_layers:                       # every block, not just the 5 depth-matched
        total = getattr(adapter, "n_layers", None) or (max(adapter.layers) + 1)
        adapter.layers = tuple(range(total))
        print(f"    all-layers mode: {total} layers")

    adata = sc.read_h5ad(args.corpus)
    if args.smoke:
        adata = adata[:32].copy()
    print(f"==> {args.model}: {adata.n_obs} cells, layers {list(adapter.layers)}")

    out_dir = os.path.join(args.out, args.model)
    pipeline.extract(adapter, adata, out_dir, batch_size=args.batch_size, max_positions=args.max_positions)
    pipeline.build_catalog(adapter, out_dir, device=args.device)

    if not args.skip_annotate:
        annotate_all(os.path.join(out_dir, "feature_catalog_L*.json"),
                     args.geneset_dir, os.path.join(args.out, "annotations"))
    print(f"==> done: {out_dir}")


if __name__ == "__main__":
    main()
