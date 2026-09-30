# Analyses added for the paper

These scripts and result files were added when the atlas was written up as a paper. They test the
original results against stricter baselines and check the pipeline itself. They read only the atlas
release files (and, for the co-expression baseline, a rebuilt copy of the 6,000-cell dataset). The
paper cites every number it takes from here as `file.json:key`.

Nothing here changes the original results in `outputs/atlas/comparative/`; the scripts write their
output next to themselves.

## What each folder holds

| folder | question | used in the paper |
|---|---|---|
| `coexpression_baseline/` | Does plain gene–gene co-expression from the same cells recover known TF–target pairs (TRRUST) as well as the gene pairs that share SAE features? Same test and chance baseline as the original (`pipeline/scripts/hypothesis_trrust2.py`); co-expression networks matched for size, for genes, and for genes plus partner counts. Also rebuilds the 6,000-cell dataset and checks the salivary-gene pairs for ambient RNA. | Results 3.5, 3.6 |
| `trrust_robustness/` | How robust is the TRRUST result? The known pairs each model recovers, concentration on a few TFs, leaving out one TF at a time, a second rewiring method, the recurrence threshold, pooling of layers, a stricter baseline that keeps each gene's number of TF partners, multiple-testing correction, and the released list rebuilt with a working same-family rule. | Results 3.4 |
| `input_encoding/` | The sorting error in the feature top-gene lists (features active on fewer than 20 genes were filled with genes from one point in the alphabet), how often it happens per model, and whether features group genes by name or gene family rather than by biology. Effect of both on the TRRUST test. | Results 3.6, 3.8 |
| `backbone_composition/` | What the 78 gene sets shared by at least 8 of 10 models are made of: which models hold each one, how many distinct programmes they form, how much is protein synthesis, how the shared core changes with depth, and how it depends on the gene-set size limit. | Results 3.2 |
| `review_10_coexpr/`, `review_11_input_encoding/`, `review_12_trrust/`, `review_13_backbone/` | Independent re-checks of the four analyses above. Where a re-check changed a number, the paper uses the re-check. | as above |

Run logs and our internal working notes are not included.

## How to run

1. From the repository root, download the release archives (feature top-gene lists, gene sets,
   TRRUST): `bash pipeline/scripts/fetch_runs.sh`.
2. Python: the packages in `requirements.txt`, plus `anndata` and `pandas` (co-expression and
   salivary checks), `numba` (`trrust_robustness/s04_null_choice.py`), `cellxgene-census` and
   `tiledbsoma` (only to rebuild the dataset) and `torch` (only for
   `input_encoding/uce_esm2_extract.py`).
3. Rebuild the dataset for the co-expression analyses (CPU only; CELLxGENE Census release
   2025-11-08, the release that `census_version="stable"` pointed to when this was run):
   `python paper_draft/analysis/coexpression_baseline/build_corpus_lowmem.py --out <file>.h5ad`,
   then pass that file to the scripts that take a dataset path as their first argument.
4. Run a script with `python paper_draft/analysis/<folder>/<script>.py`. Folders number their
   scripts in the order they were run (for example `s01_…` to `s08_…` in `trrust_robustness/`).

Paths are set by environment variables; the defaults work from a checkout of this repository:

| variable | meaning | default |
|---|---|---|
| `ATLAS_BASE` | repository root, with the release archives unpacked | the root of this checkout |
| `ATLAS_CACHE` | folder for intermediate files | `paper_draft/analysis/.cache` |
| `TRRUST_RAW` | TRRUST v2 human raw file `trrust_rawdata.human.tsv` (https://www.grnpedia.org/trrust/), used only to annotate recovered pairs | `$ATLAS_CACHE/trrust_rawdata.human.tsv` |
| `HGNC_PATH` | HGNC complete set `hgnc_complete_set.txt` (https://www.genenames.org/download/) | `$ATLAS_CACHE/hgnc_complete_set.txt` |
| `SCGPT_VOCAB` | `vocab.json` of the scGPT whole-human checkpoint | `$ATLAS_CACHE/scgpt_vocab.json` |
| `AIDO_GENE_INDEX` | gene order of the 19,264-gene panel, `OS_scRNA_gene_index.19264.tsv` | `$ATLAS_CACHE/OS_scRNA_gene_index.19264.tsv` |

The tGPT and Geneformer tokenizers and the UCE protein embeddings are read from the Hugging Face
cache (`~/.cache/huggingface/hub`) at the revisions named in `input_encoding/ie_common.py` and
`input_encoding/uce_esm2_extract.py`.

Result files record where their inputs lived as `$ATLAS_CACHE/...` or `$HGNC_PATH`. Random steps
use fixed seeds, and the result files store their numbers of random draws.
