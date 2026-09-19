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
| `svd_vs_sae.py` | SAE variance explained against top-k PCA at k = 32; the retracted "novel to SVD" share | `out_ts3/svd/<M>_L<NN>_svd.json`, assembled into `atlas_full_notf.json` |
| `svd_projection_k.py` | how much of each SAE direction lies in the top-k PCA subspace, for k = 1 … 256 | `svd_projection_k.json` |
| `seed_variance.py`, `seed_variance_alllayer.py` | retrain the SAE with three seeds and repeat the counts | summarised in `controls.json` |
| `gsea_prerank.py` | a rank-based annotation over the full gene ranking, as a check on the cutoff-based one | `gsea_annot.json` |
| `depth_profile.py` | annotation rate, concept count, variance explained and dead fraction at every layer | `depth_alllayers.json` |
| `layer_explorer.py`, `build_explorer_slim.py`, `tsne_coords.py`, `coact_embed.py` | the feature maps of the site's Layer explorer | `explorer_slim_*.json` |
| `coactivation.py`, `modules_alllayers.py` | co-activation graphs and their communities per layer | `modules_alllayers.json` |
| `circuits_adjacent.py`, `circuits.py` | feature coupling between consecutive layers | `circuits_adjacent.json` |
| `flow_alllayers.py` | feature persistence between every adjacent layer | `flow_alllayers.json` |
| `nonlinearity.py` | linear versus MLP decodability of tissue and cell type at every layer (`--layers all`) | `nonlinearity_alllayers.json` |
| `tissue_from_emb.py`, `tissue_specificity.py` | tissue binding of embeddings at every layer; per-feature tissue specificity at the depth-matched layers | `tissue_alllayers.json`; `out_ts3/tissue/<M>_tissue.json`, assembled into `atlas_full_notf.json` |
| `celltype_difficulty.py`, `hardcell_agreement.py` | which cell types and which cells are hard to decode, and whether models agree | `celltype_difficulty.json`, `hardcell_agreement.json` |

## What was run

The published catalogues come from these commands, per model, on the pod:

```bash
python build_ts_corpus.py --per-tissue 2000 --seed 0 --out outputs/atlas/ts3.h5ad   # once; --seed 1 for the held-out draw
python run_model.py --model <M> --corpus outputs/atlas/ts3.h5ad --out out --all-layers --max-positions 500000
```

`--all-layers` extracts and trains at every block; `--max-positions` caps the saved gene-token
positions per layer at 500,000. `build_catalog()` writes `out/<M>/feature_catalog_L<NN>.json`
(top-20 genes per feature by mean activation, alive count, variance explained) and
`out/<M>/sae_L<NN>.pt` (the SAE weights, configuration and training statistics).

The catalogues were copied back and are published in the release archives (see
`../../docs/REPRODUCE.md`). The SAE weight files were not copied back; they remain on the pod's
`out/` directory and are not archived here. Retraining is deterministic in the SAE seed (seed 0)
but depends on the activations, which the release does not include either, so a re-extraction is
the way to regenerate them.

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
