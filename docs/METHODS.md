# Methods, in brief

The full methods are Section 2 of the paper and Supplementary Tables S1 to S6. This page is the
short version, with pointers into the code.

## Corpus

6,000 Tabula Sapiens cells (2,000 each: immune, kidney, lung), raw counts, HGNC symbols, from
cellxgene-census v1.18.0, seed 0. The held-out corpus is seed 1, same protocol, no overlap.
Every model sees the same cells. `pipeline/atlas_h100/build_ts_corpus.py`.

## Models

Ten gene-token foundation models, 10 M to 3 B parameters, 8 to 33 layers. How each reads a cell
(rank order, rank plus value, expression value, ESM protein tokens, gene names as text), its
objective (masked or autoregressive) and its prior (none, ESM, knowledge graph, text) are in
`pipeline/atlas_h100/configs/models.yaml` and Table S1. Adapters:
`pipeline/atlas_h100/adapters/`.

## SAE

One TopK SAE per layer: k = 32, dictionary 4 x d_model, MSE, Adam 3e-4, 4 epochs, at most
500,000 gene-token positions, unit-norm decoder, seed 0. A feature's genes are ranked by mean
activation per gene; the catalogue keeps the top 20. `pipeline/atlas_h100/common/sae.py`.
Dictionary health per layer (dead fraction, decoder cosine): `pipeline/scripts/sae_health.py`,
Table S3.

## Depth

All layers are extracted. The five layers at 0, 25, 50, 75 and 100 % of depth are the
depth-matched read-out where one comparable depth per model is needed. Feature counts are never
summed across layers. `pipeline/atlas_h100/common/layers.py`.

## Annotation

Permissive: top-5 genes, Fisher exact, BH < 0.05, against GO_BP, Reactome, KEGG (Enrichr via
gseapy, retrieved 2026-08-17), STRING v12 (score >= 700) and TRRUST; set size 5 to 500;
background 16,205 genes. `pipeline/atlas_h100/common/annotate.py`, `reannotate_string.py`.

Calibrated: top-10 genes, at least 3 in one curated GO_BP, Reactome or KEGG set of at most 200
genes, no interaction sets, BH < 0.05. Fixed on the discovery corpus before held-out
evaluation. `recalibrate_final.py`, `recalibrate_robust.py`. The two are compared in Table S4.

## Nulls

Random-gene null: replace each feature's top genes with random background genes and
re-annotate. Degree-matched null: the same, with each gene replaced by one of equal gene-set
membership degree. Cell-shuffle null for CKA. Configuration-model rewiring for the co-firing
graph. Publication-count-preserving shuffle for literature co-mention. Responsiveness-matched
gene for perturbation. Empirical p = (number of null draws at or above the observed value + 1)
divided by (N + 1); N per test in Table S5 and in `docs/DECISIONS.md`.

## Annotation-free comparisons

Linear CKA between per-cell embeddings (residual stream at 50 % depth, mean-pooled over
tokens), all 45 pairs, each against its own cell-shuffle null. `cell_cka.py`, `cka_svd_null.py`.

SAE variance explained at k = 32 against the best rank-32 PCA subspace and an untrained
random dictionary; projection of SAE directions onto the top-k PCA subspace for k = 1 to 256.
`svd_vs_sae.py`, `svd_projection_k.py`, Table S7, Fig S1.

Linear decodability of tissue (3 classes) and cell type (15 classes) with a logistic probe and
an MLP reference, cross-validated, at every layer. `nonlinearity.py`.

## Held-out prediction

Per model: pool features over all layers, top-10 genes per feature, every within-feature pair;
a pair is predicted when it recurs in at least five features. Score against TRRUST
(held out from the whole pipeline) with a configuration-model null, 500 rewirings.
Corroboration across models over all ten graphs, 100 rewirings. Equal-budget re-scoring at
8,631 pairs. Robustness variants: paralogues dropped, threshold 10, firing features only, both,
and the near-duplicate check of Table S12. `hypothesis_trrust*.py`, `hypothesis_robust.py`,
`hypothesis_sizematched.py`, `pair_redundancy.py`.

Literature: NCBI gene2pubmed bulk file of 2026-09-11, papers with more than 60 genes excluded.
`hypothesis_pubmed.py`. Perturbation: Replogle 2022 CRISPRi Perturb-seq, K562 and RPE1,
within-row percentile of the response, responsiveness-matched null. `hypothesis_perturb.py`.
Publication-stratum analysis: `hypothesis_studybias.py`, `study_bias_coverage.py`.

## Environments

GPU host (extraction, SAE training): Python 3.12, PyTorch 2.13 (CUDA 13), NumPy 1.26.4,
cellxgene-census 1.18.0, NVIDIA H100; Tahoe-x1 in its own environment (torch 2.6.0,
flash-attn 2.7.4). Local (everything downstream): Python 3.13.0, NumPy 2.3.4, SciPy 1.17.1,
h5py 3.16.0, anndata 0.13.2, matplotlib 3.10.7. `requirements.txt` pins the local set. Table S6.
