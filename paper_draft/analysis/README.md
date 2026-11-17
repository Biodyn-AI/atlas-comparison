# Paper analyses on runs-v2

These analyses use the corrected 6,000-cell extraction. The checked-in JSON
and TSV files are the small outputs that support the paper.

| Directory | Analysis |
|---|---|
| `coexpression_baseline/` | TRRUST recovery for co-expression matched by graph size, genes or per-gene degree, within-cell-type and within-cell-type-and-assay comparisons, and the salivary and lung pair check |
| `trrust_robustness/` | Per-model recovered pairs, hub and family masks, alternative rewirings, layer pooling, multiple testing and the candidate-list filter |
| `input_encoding/` | Top-gene filler and name, family and tokenizer-order checks |
| `backbone_composition/` | Composition and depth of the 96 gene sets shared by at least eight models, with size sensitivity and nulls |
| `review_*/` | Independent checks, including Table 3's stricter rewiring and the family-masked comparison in Figure 5C |
| `untrained_controls/` | Controls using randomly initialized SAEs and models, exported results and generators |
| `adapter_check/` | Comparison with the official inference code for scGPT, scFoundation, UCE and Tahoe-x1 |
| `repro/` | Gene-popularity baseline, reconstructed from the stated method, and its historical validation |
| `figures/` | Generator for the four-panel co-expression Figure 5 |

## Inputs

Run from the repository root with the dependencies in `requirements.txt`.
The baseline scripts also use pandas and anndata, edge swapping uses numba, and
the figure uses Matplotlib. The corpus rebuild needs CELLxGENE Census and
TileDB-SOMA. The GPU checks use the model-specific environments from
`pipeline/atlas_h100/` and the authors' inference packages.

```sh
bash pipeline/scripts/fetch_runs.sh
export ATLAS_BASE="$PWD" PYTHONHASHSEED=0
analysis="$PWD/paper_draft/analysis"
export ATLAS_CACHE="$analysis/.cache"
export PYTHONPATH="$analysis${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ATLAS_CACHE"
```

The scripts read the release catalogues and comparative outputs under
`$ATLAS_BASE/outputs/atlas/`. Several read
`outputs/atlas/comparative/.cofire_cache.pkl`, generated from runs-v2 by
`pipeline/scripts/hypothesis_trrust3.py`. Generate this cache in a disposable
copy or back up the tracked outputs. The generator also rewrites the candidate
list and summary. Use a fresh cache when changing catalogues.

The rerun uses the same ordered Tabula Sapiens corpus as the extraction, with
6,000 cells, seed 0 and 2,000 cells per compartment. Use that corpus with
upper-case gene symbols as unique `var_names` and the sample metadata, including
`cell_type` and `assay`. For reconstruction, the original Census build was
`2025-11-08`:

```sh
corpus="$ATLAS_CACHE/corpus6k.h5ad"
python "$analysis/coexpression_baseline/build_corpus_lowmem.py" --census-version 2025-11-08 --seed 0 --per-tissue 2000 --out "$corpus"
python "$analysis/coexpression_baseline/same_cells_check.py" "$corpus"
```

For an exact reconstruction, compare cell identifiers and row order to the
extraction corpus. The bundled check compares class labels and counts.
`fix_corpus_obs.py` is a recovery utility for a partial historical download.

| Variable | Input |
|---|---|
| `TRRUST_RAW` | Human TRRUST v2 raw TSV, default `$ATLAS_CACHE/trrust_rawdata.human.tsv` |
| `HGNC_PATH` | HGNC complete set, default `$ATLAS_CACHE/hgnc_complete_set.txt` |
| `SCGPT_VOCAB` | Whole-human scGPT vocabulary, default `$ATLAS_CACHE/scgpt_vocab.json` |
| `AIDO_GENE_INDEX` | 19,264-gene panel in model order, default `$ATLAS_CACHE/OS_scRNA_gene_index.19264.tsv` |

The tokenizer revisions and UCE protein-embedding input are named in
`input_encoding/ie_common.py` and `input_encoding/uce_esm2_extract.py`.

## Reproduce the comparisons

```sh
python "$analysis/trrust_robustness/s01_build_cache.py"
python "$analysis/trrust_robustness/s02_name_edges.py"
python "$analysis/trrust_robustness/s02b_family_classes.py"
python "$analysis/review_12_trrust/r1_stratified_direction.py" 1000
python "$analysis/trrust_robustness/s04_null_choice.py" 200 100
python "$analysis/review_12_trrust/r2_curve_many_draws.py" 2000

python "$analysis/coexpression_baseline/coexpr_trrust.py" "$corpus" 200 pearson,spearman,withincomp
python "$analysis/coexpression_baseline/coexpr_degmatched.py" "$corpus" 200
python "$analysis/coexpression_baseline/coexpr_rmatched.py" "$corpus" 10000
python "$analysis/coexpression_baseline/residual_hits_prefix.py"
python "$analysis/coexpression_baseline/salivary_check.py" "$corpus"
python "$analysis/review_10_coexpr/prefix_estimate.py" "$corpus"
python "$analysis/review_10_coexpr/review_heavy.py" "$corpus" 200 P1
python "$analysis/review_10_coexpr/review_cheap.py"
python "$analysis/review_10_coexpr/lead_without_family.py"
python "$analysis/coexpression_baseline/coexpr_degmatched_strata.py" "$corpus" celltype 200
python "$analysis/coexpression_baseline/coexpr_degmatched_strata.py" "$corpus" celltype_assay 200
python "$analysis/repro/pop_baseline.py" 200
python "$analysis/figures/fig5.py"
python "$analysis/make_table3.py" > "$ATLAS_CACHE/Table_3.tex"
```

Table 3 uses the unmasked comparisons from
`hypothesis_trrust2.json`, `r1_stratified_direction.json`, `coexpr_trrust.json`,
`coexpr_degmatched.json` and `review_cheap.json`. Figure 5C also uses
`lead_without_family.json`. M1 excludes pairs sharing the first two symbol
characters. M2 excludes pairs within one curated DNA-binding-domain class.
For Tahoe-x1 the masked discordant hit counts are 53 versus 34 across all
cells (nominal p = 0.053), 49 versus 25 within cell types (p = 0.0071) and
47 versus 22 within cell type and assay (p = 0.0035). Figure 5 is generated in
`$ATLAS_CACHE/figures/`, overridable with `ATLAS_FIGURES`.

Run the numbered backbone and TRRUST scripts in order. For input encoding,
generate the UCE keys with `uce_esm2_extract.py` before `lexical.py`.
`filler_check_published.json` and `pop_baseline_OLDDATA_validation.json` describe
runs-v1. Use separate analysis and cache directories for those historical
checks. Filenames such as `backbone78_table.tsv` and JSON keys such as
`pooled_5_passing` are retained for compatibility. Their runs-v2 counts are
96 sets and four passing models.

The 4,882-pair reference for `trrust_robustness/s07_released_list.py` comes
from the pre-filter runs-v2 snapshot:

```sh
historical="$ATLAS_CACHE/pre-family-filter"
mkdir -p "$historical"
for f in hypothesis_pairs_full.json hypothesis_pubmed.json; do
  git show "04089f2:outputs/atlas/comparative/$f" > "$historical/$f"
done
python "$analysis/trrust_robustness/s07_released_list.py" "$historical"
```

The rebuilt list has 4,485 pairs and matches the current release. The
historical diagnostic scores the literature fields available at that point.
`input_encoding/released_flags.json` describes the 4,485-pair list.

## Controls requiring GPU inputs

`ATLAS_GPU_BUNDLE` points to the configured extraction bundle with checkpoints,
adapters and the corpus. Set `ATLAS_GPU_CORPUS` to override its corpus path.
The random-SAE generator reads the per-layer activations from `out_fixed/`
inside that bundle by default. Set `ATLAS_FIXED_ACTIVATIONS` to override it.
The adapter check uses the CKA embeddings under `out_fixed/cka/` by default
and supports `ATLAS_CKA_EMBEDDINGS` and `ATLAS_TAHOE_PYTHON`. Tahoe uses its own
model environment. The feature catalogues from the controls and the full
embeddings are external inputs.

```sh
python "$analysis/untrained_controls/random_sae_catalogs.py" AIDO UCE Geneformer MaxToki scGPT tGPT C2S scFoundation GeneCompass
# In the Tahoe model environment:
python "$analysis/untrained_controls/random_sae_catalogs.py" Tahoe
python "$analysis/untrained_controls/run_untrained.py" Geneformer
python "$analysis/untrained_controls/run_untrained.py" UCE
python "$analysis/adapter_check/adapter_check.py" scGPT
```

Run each model in its configured environment. Assemble one directory of
runs-v2 trained feature catalogues, with the all-layer release catalogues
for seven models and the all-layer catalogues from `ts3_out` for GeneCompass,
scFoundation and Tahoe. Set `ATLAS_TRAINED_CATALOGUES` to that directory.

```sh
controls="$analysis/untrained_controls"
python "$controls/controls.py" real "$ATLAS_TRAINED_CATALOGUES"
python "$controls/controls.py" random_sae "$ATLAS_CACHE/random_sae"
python "$controls/controls.py" untrained_model "$ATLAS_CACHE/out_untrained" Geneformer,UCE
python "$controls/core_overlap.py" "$ATLAS_TRAINED_CATALOGUES" "$ATLAS_CACHE/random_sae"
python "$controls/family_hits.py" "$ATLAS_TRAINED_CATALOGUES" "$ATLAS_CACHE/out_untrained" "$ATLAS_CACHE/random_sae"
```

The programme summary is produced by `pipeline/scripts/programme_table.py`.
It also uses the random-SAE catalogues and the original extraction corpus.
