# Reproduce

Three stages. The first needs a GPU and the model checkpoints; the other two run on a laptop
from the files in this repository and its release archives.

## 0. Set up

```bash
git clone https://github.com/Biodyn-AI/atlas-comparison && cd atlas-comparison
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ATLAS_BASE=$(pwd)          # every script reads and writes under $ATLAS_BASE/outputs/atlas/
```

`requirements.txt` pins the versions the published nulls, figures and audits were computed
under (Python 3.13, NumPy 2.3.4). The GPU stage ran under a different NumPy major version; see
`pipeline/atlas_h100/README.md`.

## 1. Check the published numbers (no downloads)

The result files are in `outputs/atlas/comparative/`. The audit re-derives every number in the
paper, the supplement, the figures and the site from them. The manuscript is not in the
repository; drop `MANUSCRIPT.md` and `SUPPLEMENTARY.md` next to the results first.

```bash
cp /path/to/MANUSCRIPT.md /path/to/SUPPLEMENTARY.md outputs/atlas/comparative/
python pipeline/scripts/audit_all.py
```

Expected: six modules, all PASS, 312 checks. To rebuild the site from the same files:

```bash
python pipeline/scripts/inject_atlas.py      # writes index.html and data/explorer.json
python -m http.server 8765                   # then open http://localhost:8765/
```

To re-render the figures and the generated supplementary tables:

```bash
python pipeline/scripts/make_figures.py && python pipeline/scripts/make_fig14.py \
  && python pipeline/scripts/make_fig5.py && python pipeline/scripts/make_figS1.py \
  && python pipeline/scripts/make_figS2.py
python pipeline/scripts/audit_figures.py --update    # re-stamp the manifest
python pipeline/scripts/make_supp_tables.py          # Tables S2, S3, S5, S8, S9, S10, S12
```

## 2. Re-run the analysis from the catalogues

The per-run outputs and the external inputs are release assets, because they are 1.2 GB
unpacked. The fetch script downloads them, verifies the checksums and unpacks into
`outputs/atlas/`:

```bash
bash pipeline/scripts/fetch_runs.sh             # add --legacy for the superseded first run
```

The current release is `runs-v2`, the catalogues of the corrected extraction (every model reads
all 6,000 cells; `CHANGELOG.md`, 2026-10). `runs-v1` keeps the superseded catalogues and the
legacy cluster run.

| archive | MB | unpacks to | holds |
|---|---|---|---|
| `runs-ts3_out.tar.gz` | 70 | `outputs/atlas/ts3_out/` | feature catalogues on the shared corpus: five depth-matched layers for seven models, every layer for GeneCompass, scFoundation and Tahoe-x1; the permissive annotations of the seven models' five layers |
| `runs-alllayer_cat.tar.gz` | 57 | `outputs/atlas/alllayer_cat/` | every-layer catalogues for AIDO.Cell, C2S-Scale, Geneformer-V2, MaxToki, UCE, scGPT, tGPT |
| `runs-heldout_cat.tar.gz` | 23 | `outputs/atlas/heldout_cat/` | catalogues on the held-out draw (5,581 cells, none shared with the corpus), all ten models, five depth-matched layers |
| `runs-seeds.tar.gz` | 24 | `outputs/atlas/seeds/` | the SAE-seed control (three seeds, mid layer and every layer), read by `controls_import.py` |
| `inputs-genesets.tar.gz` | 101 | `outputs/atlas/genesets/` | GO_BP, KEGG, Reactome sets (Enrichr via gseapy, 2026-08-17), STRING v12 edges and aliases, TRRUST edges, the 16,205-gene background; unchanged from v1 |
| `legacy-cluster-run.tar.gz` (v1) | 76 | `outputs/atlas/cluster/` | the first cluster run (AIDO, UCE, tGPT, scGPT on an earlier corpus); superseded, read by no published number |

```
4dc9f0c1928fc8d0121cf01efbf1eba38dc53c675b835dfcd960324c405fd122  inputs-genesets.tar.gz
738b694e31f0502cd0e66dc72a83c8199730bde0b799e8168df4cd1c4147b4ee  runs-alllayer_cat.tar.gz
295df6e717d991984515cbc8c5ed82505dce67fcab1b1d7c3c44ec12b66bc55c  runs-heldout_cat.tar.gz
41382c8f762c3bae686f63029ec897d51061f1a26b10d87174e4542dd6e83a7c  runs-seeds.tar.gz
898c5e9471ba315e2da30792c7e226f4bc2f641b46f295d89b080d4fa933f0f1  runs-ts3_out.tar.gz
bdffaf0042bf4abc03d62dd23294cb898d5f197230281914dc783d69706e5866  legacy-cluster-run.tar.gz (v1)
```

Two inputs are not redistributed and must be fetched from their sources into
`outputs/atlas/pubmed/` and `outputs/atlas/perturb/`:

- NCBI `gene2pubmed.gz` and `Homo_sapiens.gene_info.gz` (ftp.ncbi.nlm.nih.gov/gene/DATA/),
  as downloaded on 2026-09-11. Used by `hypothesis_pubmed.py`, `hypothesis_studybias.py`,
  `study_bias_coverage.py`.
- Replogle et al. 2022 normalised pseudobulk Perturb-seq, `K562_gwps_normalized_bulk_01.h5ad`
  and `rpe1_normalized_bulk_01.h5ad` (plus.figshare.com, "Mapping information-rich
  genotype-phenotype landscapes with genome-scale Perturb-seq"). Used by
  `hypothesis_perturb.py`, `hypothesis_studybias.py`, `study_bias_coverage.py`.

Then the local stage, in this order (each script says what it reads and writes in its
docstring; `pipeline/scripts/README.md` groups them):

```
atlas_build_ts3 -> reannotate_string -> alllayer_matrix -> alllayer_concepts
  -> alllayer_concepts Tahoe,scFoundation,GeneCompass --merge -> module_themes -> genes_search_ts3
  -> findings -> flow_ts3
controls:    alllayer_null, recalibrate, recalibrate2, recalibrate_final,
             recalibrate_robust, stats_final, degree_null, depth_backbone, depth_backbone_fast,
             depth_calibrated, heldout_compare, heldout_calibrated, random_null, kegg_robust, sae_health, topn_sweep,
             controls, controls2, controls3, controls4, backbone_terms,
             novel_null, novel_calibrated, coexpr_core_baseline, controls_import
predictions: hypothesis_trrust, hypothesis_trrust2, hypothesis_trrust3, hypothesis_robust,
             hypothesis_sizematched, hypothesis_pubmed, hypothesis_perturb,
             hypothesis_studybias, hypothesis_singlelayer, study_bias_coverage, pair_redundancy,
             agreement_curve
assembly:    (atlas_assemble -> controls, controls2, controls3, controls4, controls_import) until controls.json
             stops changing -> atlas_assemble -> atlas_hypothesis_block -> inject_atlas
figures:     make_figures, make_fig14, make_fig5, make_figS1, make_figS2, audit_figures --update
tables:      make_supp_tables
check:       audit_all
```

Order matters in several places. `controls.py` reads the matrices that `alllayer_matrix.py` writes, so
the first line must run before the controls. `alllayer_concepts.py` covers seven models by default;
the `--merge` run adds the three whose every-layer catalogues sit in `ts3_out`. `depth_backbone_fast.py`
refuses to write unless it reproduces the observed counts of `depth_backbone.py`, so the slow script
runs first. `controls_import.py` copies into `controls.json` the blocks no other script writes (the
nulls, the held-out comparisons, the recalibration grid, the GPU-side seed control from
`outputs/atlas/seeds/`), so it runs after all of them. `hypothesis_trrust3.py` and
`hypothesis_sizematched.py` cache the co-firing graphs in `comparative/.cofire_cache.pkl`; delete it
whenever the catalogues change. `atlas_assemble.py` and the controls read each other's output (the
SVD block, the matrices summary), so they are iterated to a fixed point, two passes in practice;
run with `PYTHONHASHSEED=0` so that ties between equally ranked concepts resolve the same way
every time. The models carried into the cross-model analysis are read from
`hypothesis_trrust2.json` (significant at p < 0.05), and `hypothesis_trrust3.py` writes the full released
pair list that `hypothesis_pubmed.py` then annotates, so the predictions run in the order listed;
`pair_redundancy.py` reads `hypothesis_trrust2.json` and comes after it; `random_null.py` reads
`heldout.json` and comes after `heldout_compare.py`.

Permutation-based scripts take minutes to an hour on a laptop (`controls.py` about 3 minutes,
`depth_backbone_fast.py` and the `hypothesis_*` scripts longer). Seeds are fixed in each
script, so the numbers reproduce exactly.

## 3. Re-extract (GPU)

Needs the ten checkpoints, an H100-class GPU and the environments in
`pipeline/atlas_h100/environment.yml`. Build the corpus once, then run each model:

```bash
cd pipeline/atlas_h100
python build_ts_corpus.py --per-tissue 2000 --seed 0 --out external/perturb/tabula_3tissue_6k.h5ad
python build_ts_corpus.py --per-tissue 2000 --seed 1 --out external/perturb/tabula_3tissue_6k_heldout.h5ad
python make_heldout_disjoint.py                      # drop the 419 cells the two draws share
bash run_full_fixed.sh                               # every layer, every model, all 6,000 cells
bash run_heldout_fixed.sh                            # the held-out draw, five depth-matched layers
bash run_seeds_fixed.sh                              # the SAE-seed control
bash run_downstream_fixed.sh                         # cell embeddings, CKA, probes, SVD, explorer
# then svd_vs_sae.py (mid layer of each model) and gsea_prerank.py; copy their outputs to
# outputs/atlas/comparative/svd/ and outputs/atlas/comparative/gsea_annot.json
```

`run_model.py --per-cell 168` reads every cell and keeps at most 168 gene-token positions per
cell, about 1M positions per layer, so every model sees the same cells with equal weight. Without
`--per-cell` it falls back to the original corpus-order cap on `--max-positions`, which is how the
superseded v1 catalogues were made (AIDO.Cell read 32 cells, tGPT all 6,000). The scripts write
`feature_catalog_L<NN>.json` and `sae_L<NN>.pt` per layer. The SAE weights (about 17 GB) and
activations stay on the GPU host and are not archived; the catalogues, which every downstream
number uses, are.
