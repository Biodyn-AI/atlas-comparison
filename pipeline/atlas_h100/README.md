# Extraction and SAE training (GPU side)

This folder is the code that ran on the H100 host. It takes a model checkpoint and the shared
corpus, extracts the residual stream at every layer, trains one TopK sparse autoencoder per
layer, and writes a feature catalogue per layer. The per-model and aggregate analyses that need
the activations or the SAE weights also live here. Everything that only needs the catalogues
runs locally and is in `../scripts/`.

## Layout

```
run_model.py            one model end to end: adapter -> activations -> SAE per layer -> catalogue
build_ts_corpus.py      the shared corpus from cellxgene-census (seed 0; seed 1 is the held-out draw)
adapters/               one file per model; base.py is the contract every adapter implements
common/sae.py           the TopK SAE and its training loop (k = 32, 4 x d_model, MSE, Adam, 4 epochs)
common/pipeline.py      extract() saves activations, build_catalog() trains the SAE and writes the catalogue
common/annotate.py      the permissive annotator (top-5 genes, five databases); the calibrated one is local
common/layers.py        depth matching: which layer indices are 0 / 25 / 50 / 75 / 100 % of depth
configs/models.yaml     the ten models with size, depth, input type, prior and checkpoint
data/build_genesets.py  the gene-set vocabulary (GO_BP, KEGG, Reactome via Enrichr; STRING; TRRUST)
environment.yml         the conda environment of the GPU host
```

The remaining scripts are the analyses that need activations or weights, run per model or over
all models once the catalogues exist:

| script | what it computes | result file (local) |
|---|---|---|
| `cell_cka.py` | per-cell mean-pooled embeddings at every layer; linear CKA between models; feature firing frequencies | `cka_ts3.json`, `freq_ts3.json` |
| `cka_layers.py` | layer-by-layer CKA within one model | `cka_layers.json` |
| `cka_svd_null.py` | the cell-shuffle null for CKA and the random-direction null for the SVD comparison, all ten models | `cka_svd_null.json` |
| `svd_vs_sae.py` | SAE variance explained against top-k PCA at k = 32; the retracted "novel to SVD" share | `<out>/svd/<M>_L<NN>_svd.json`, copied to `outputs/atlas/svd/` and assembled into `atlas_full_notf.json` (`controls2.py` reads it from there) |
| `svd_projection_k.py` | how much of each SAE direction lies in the top-k PCA subspace, for k = 1 … 256 | `svd_projection_k.json` |
| `seed_variance.py`, `seed_variance_alllayer.py` | retrain the SAE with three seeds and repeat the counts | summarised in `controls.json` |
| `gsea_prerank.py` | a rank-based annotation over the full ~19,000-gene ranking of every feature, as a check on the cutoff-based one (needs the activations, so it runs here) | `gsea_annot.json` |
| `depth_profile.py` | annotation rate, concept count, variance explained and dead fraction at every layer | `depth_alllayers.json` |
| `layer_explorer.py`, `build_explorer_slim.py`, `tsne_coords.py`, `coact_embed.py` | the feature maps of the site's Layer explorer | `explorer_slim_*.json` |
| `coactivation.py`, `modules_alllayers.py` | co-activation graphs and their communities per layer | `modules_alllayers.json` |
| `circuits_adjacent.py`, `circuits.py` | feature coupling between consecutive layers | `circuits_adjacent.json` |
| `flow_alllayers.py` | feature persistence between every adjacent layer | `flow_alllayers.json` |
| `nonlinearity.py` | linear versus MLP decodability of tissue and cell type at every layer (`--layers all`) | `nonlinearity_alllayers.json` |
| `tissue_from_emb.py` | tissue binding of embeddings at every layer | `tissue_alllayers.json` |
| `tissue_specificity.py` | per-feature tissue specificity at the depth-matched layers; exploratory, read by no published number and not re-run on the corrected SAEs | `out_ts3/tissue/<M>_tissue.json` |
| `celltype_difficulty.py`, `hardcell_agreement.py` | which cell types and which cells are hard to decode, and whether models agree | `celltype_difficulty.json`, `hardcell_agreement.json` |

## What was run

The published catalogues come from the corrected extraction (release `runs-v2`), run on the pod:

```bash
python build_ts_corpus.py --per-tissue 2000 --seed 0 --out external/perturb/tabula_3tissue_6k.h5ad
python build_ts_corpus.py --per-tissue 2000 --seed 1 --out external/perturb/tabula_3tissue_6k_heldout.h5ad
python make_heldout_disjoint.py      # the held-out draw minus the 419 cells it shares with the corpus
bash run_full_fixed.sh               # run_model.py --all-layers --per-cell 168 --seed 0, all ten models
bash run_heldout_fixed.sh            # the same on the held-out draw, five depth-matched layers
bash run_seeds_fixed.sh              # seed_variance.py and seed_variance_alllayer.py, seeds 0-2
bash run_downstream_fixed.sh         # every analysis in the table above
```

`--per-cell 168` reads all 6,000 cells and keeps at most 168 gene-token positions per cell, drawn
with seed 0, which gives about 1M positions per layer, the SAE's training budget. Each layer's
activations are streamed to disk as float16, or float32 where values exceed the float16 range
(tGPT). `build_catalog()` writes `out_fixed/<M>/feature_catalog_L<NN>.json` (top-20 genes per
feature by mean activation, ties broken stably, genes with zero mean activation never listed) and
`out_fixed/<M>/sae_L<NN>.pt` (weights, configuration, training statistics; the SAE seeds torch).

The superseded v1 catalogues were made with `--max-positions 500000` and no `--per-cell`: positions
were taken in corpus order until the cap, so models that read many genes per cell saw few cells
(AIDO.Cell 32, UCE 492, scFoundation about 160) and tGPT saw all 6,000. That run also reused a
stale UCE preprocessing cache, so UCE's held-out catalogue was built on corpus cells; the adapter
now keys the cache on the cells it is given.

The catalogues are published in the release archives (see `../../docs/REPRODUCE.md`). The SAE
weight files (about 17 GB) and the activations stay on the pod and are not archived here.

## Adapter contract

An adapter implements `iter_activations(adata, batch_size)`, which yields per batch a dict
`{layer_index: float32 array [n_positions, d_model]}` and the HGNC symbols of those positions.
Only gene-token positions are yielded; CLS, padding and other special tokens are dropped. The
rest of the pipeline is model-agnostic. Adapter-specific dependencies are commented in
`environment.yml`; Tahoe-x1 needed its own environment (torch 2.6.0, flash-attn 2.7.4).

## Two environments

Extraction ran under Python 3.12, PyTorch 2.13 (CUDA 13), NumPy 1.26.4 on an NVIDIA H100. The
local analysis ran under Python 3.13 with NumPy 2.3.4. The NumPy major version differs; a reader
reproducing the analyses needs only the local environment (`../../requirements.txt`).
