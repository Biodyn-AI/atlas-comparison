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

| archive | MB | unpacks to | holds |
|---|---|---|---|
| `runs-ts3_out.tar.gz` | 71 | `outputs/atlas/ts3_out/` | feature catalogues on the shared corpus: five depth-matched layers for seven models, every layer for GeneCompass, scFoundation and Tahoe-x1; the permissive annotations; per-layer analyses under `out_ts3/` |
| `runs-alllayer_cat.tar.gz` | 55 | `outputs/atlas/alllayer_cat/` | every-layer catalogues for AIDO.Cell, C2S-Scale, Geneformer-V2, MaxToki, UCE, scGPT, tGPT |
| `runs-heldout_cat.tar.gz` | 20 | `outputs/atlas/heldout_cat/` | catalogues on the held-out draw, nine models (no scGPT) |
| `inputs-genesets.tar.gz` | 101 | `outputs/atlas/genesets/` | GO_BP, KEGG, Reactome sets (Enrichr via gseapy, 2026-08-17), STRING v12 edges and aliases, TRRUST edges, the 16,205-gene background |
| `legacy-cluster-run.tar.gz` | 76 | `outputs/atlas/cluster/` | the first cluster run (AIDO, UCE, tGPT, scGPT on an earlier corpus); superseded, read by no published number |

```
4dc9f0c1928fc8d0121cf01efbf1eba38dc53c675b835dfcd960324c405fd122  inputs-genesets.tar.gz
bdffaf0042bf4abc03d62dd23294cb898d5f197230281914dc783d69706e5866  legacy-cluster-run.tar.gz
3e8be9f9275f3732bcd88c95841829a4f38c0dc5ca5ad0b6e74ccc5fabd06aae  runs-alllayer_cat.tar.gz
f764134647419f9510ad90a718ee12fbadeea786c8d568d51291ddaf70a6f88b  runs-heldout_cat.tar.gz
9df317cb3ce2bc21941629b529953b4f7638938c09403b8666b1ff8550ecb592  runs-ts3_out.tar.gz
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
atlas_build_ts3 -> reannotate_string -> alllayer_matrix -> alllayer_concepts -> module_themes
  -> genes_search_ts3 -> findings
controls:    random_null, alllayer_null, recalibrate, recalibrate2, recalibrate_final,
             recalibrate_robust, stats_final, degree_null, depth_backbone_fast, heldout_compare,
             heldout_calibrated, kegg_robust, sae_health, topn_sweep, gsea_annot,
             controls, controls2, controls3, controls4, backbone_terms, pair_redundancy,
             novel_null, novel_calibrated
predictions: hypothesis_trrust, hypothesis_trrust2, hypothesis_trrust3, hypothesis_robust,
             hypothesis_sizematched, hypothesis_pubmed, hypothesis_perturb,
             hypothesis_studybias, hypothesis_singlelayer, study_bias_coverage
assembly:    atlas_assemble -> atlas_hypothesis_block -> inject_atlas
figures:     make_figures, make_fig14, make_fig5, make_figS1, make_figS2, audit_figures --update
tables:      make_supp_tables
check:       audit_all
```

Permutation-based scripts take minutes to an hour on a laptop (`controls.py` about 3 minutes,
`depth_backbone_fast.py` and the `hypothesis_*` scripts longer). Seeds are fixed in each
script, so the numbers reproduce exactly.

## 3. Re-extract (GPU)

Needs the ten checkpoints, an H100-class GPU and the environments in
`pipeline/atlas_h100/environment.yml`. Build the corpus once, then run each model:

```bash
cd pipeline/atlas_h100
python build_ts_corpus.py --per-tissue 2000 --seed 0 --out $ATLAS_BASE/outputs/atlas/ts3.h5ad
python build_ts_corpus.py --per-tissue 2000 --seed 1 --out $ATLAS_BASE/outputs/atlas/ts3_heldout.h5ad
python run_model.py --model Tahoe --corpus $ATLAS_BASE/outputs/atlas/ts3.h5ad \
                    --out $ATLAS_BASE/outputs/atlas/ts3_out --all-layers --max-positions 500000
```

`run_model.py` writes `feature_catalog_L<NN>.json` and `sae_L<NN>.pt` per layer. The
per-model and aggregate analyses that need activations or weights are listed in
`pipeline/atlas_h100/README.md`. The SAE weights of the published run were not copied back from
the GPU host and are not archived; the catalogues, which every downstream number uses, are.
