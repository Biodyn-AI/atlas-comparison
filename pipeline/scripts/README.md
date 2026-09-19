# Analysis scripts (local, CPU)

Everything here runs from the feature catalogues and the result files; nothing needs a GPU.
Each script reads its data root from `ATLAS_BASE` (default: the authors' path) and names what
it reads and writes in its docstring. `docs/RESULTS.md` has the same map from the file side.

## Build the matrices and indexes

| script | does |
|---|---|
| `atlas_build_ts3.py` | first pass over the catalogues: the depth-matched concept x model matrix and depth profiles |
| `reannotate_string.py` | re-annotates every catalogue with the permissive annotator (top-5, five databases) so all ten models share one vocabulary; writes `matrix_ts3_string.json` |
| `alllayer_matrix.py` | the same over every layer; writes `matrix_alllayer.json` and `perlayer_concepts.json` |
| `alllayer_concepts.py` | distinct concepts per model across the depth-matched layers |
| `module_themes.py` | maps co-activation communities to nine coarse themes |
| `genes_search_ts3.py` | the gene-search index the site uses |
| `findings.py` | emergence, convergence and curriculum descriptives (supplementary, underpowered) |
| `flow_ts3.py` | feature persistence between the depth-matched layers |

## Controls: the nulls behind the claims

| script | does |
|---|---|
| `random_null.py` | random-gene null at the mid layer; retracts the naive universal core |
| `alllayer_null.py` | the same null on the all-layer union, 20 permutations |
| `recalibrate.py`, `recalibrate2.py` | grids over annotator settings until random genes stop annotating |
| `recalibrate_final.py` | the chosen calibrated annotator and its backbone |
| `recalibrate_robust.py` | the backbone across five calibrated configurations and thresholds; the housekeeping keyword rule (`HK`) |
| `stats_final.py` | backbone against the uniform null (250 permutations) and the held-out replication test (150) |
| `degree_null.py` | the degree-matched random-gene null, 250 permutations |
| `depth_backbone.py`, `depth_backbone_fast.py` | the backbone at five read-out depths; the fast version does 250 permutations and checks it reproduces the slow one's counts |
| `heldout_compare.py`, `heldout_calibrated.py` | replication on the held-out draw, permissive (fails) and calibrated (passes) |
| `kegg_robust.py` | the backbone without KEGG |
| `sae_health.py` | dead features and decoder cosine for all 172 model-layers |
| `topn_sweep.py`, `gsea_annot.py` | annotation at other gene cutoffs; a rank-based annotation as a cross-check |
| `controls.py`, `controls2.py`, `controls3.py`, `controls4.py` | the reviewer-round controls, merged into `controls.json` (size-weighted null, equal-layer core, capacity confounds, ontology redundancy, vocabulary fairness, seed CIs, bootstrap CIs on the n = 10 correlations) |
| `backbone_terms.py` | the backbone concept names by sharing tier |
| `pair_redundancy.py` | near-duplicate feature blocks and whether the prediction result rests on them |
| `novel_null.py`, `novel_calibrated.py` | the two nulls that retired the "new biology" shortlist |
| `depth_calibrated.py` | annotation rate at every layer under the calibrated annotator |

## Held-out prediction and the two further kinds of evidence

| script | does |
|---|---|
| `hypothesis_trrust.py` | stage 1: co-firing graph per model against TRRUST, analytic degree-matched null |
| `hypothesis_trrust2.py` | stage 2: 500 rewirings per model; the corroboration curve over all ten models |
| `hypothesis_trrust3.py` | stage 3: the validated models and the released prediction list |
| `hypothesis_robust.py` | paralogues dropped, threshold 10, firing features only, both |
| `hypothesis_sizematched.py` | every model on an equal budget of 8,631 pairs |
| `hypothesis_singlelayer.py` | the test with a single-layer graph |
| `hypothesis_pubmed.py` | literature co-mention against a publication-count-preserving null |
| `hypothesis_perturb.py` | Perturb-seq in K562 and RPE1 against a responsiveness-matched null |
| `hypothesis_studybias.py`, `study_bias_coverage.py` | predictive power by publication stratum; what the evidence sources can check |

## Assemble the site

| script | does |
|---|---|
| `atlas_assemble.py` | builds `atlas_full_notf.json`, the one file the site embeds, from the component results |
| `atlas_hypothesis_block.py` | rebuilds its prediction block from the hypothesis files; `--check` reports drift |
| `inject_atlas.py` | embeds the data blocks into `../atlas_template.html` and writes `index.html` and `data/explorer.json` |

## Figures, tables, manuscript

| script | does |
|---|---|
| `make_figures.py` | Fig 2 and Fig 3 |
| `make_fig14.py` | Fig 1 (schematic) and Fig 4 |
| `make_fig5.py` | Fig 5 |
| `make_figS1.py`, `make_figS2.py` | Fig S1 and Fig S2 |
| `make_supp_tables.py` | regenerates the supplementary tables that depend on results (S2, S3, S5, S8, S9, S10, S12) |
| `manuscript_to_html.py`, `manuscript_to_docx.js` | render the manuscript with figures embedded (the manuscript itself is not in this repository) |

## Audit

| script | does |
|---|---|
| `audit_all.py` | runs the six modules below and says whether it is safe to submit |
| `audit_manuscript.py` | every number in Results 3.4 to 3.5, Discussion and Limitations, against the file that produced it |
| `audit_manuscript_core.py` | the same for Results 3.1 to 3.3 and Methods |
| `audit_manuscript_derived.py` | numbers the manuscript derives rather than reads off a file |
| `audit_supplement.py` | the hand-written supplementary tables against the data, the script constants and the environment |
| `audit_figures.py` | figures against the hashes of their inputs; `--update` re-stamps after re-rendering |

`fetch_runs.sh` downloads the release archives with the per-run outputs (`docs/REPRODUCE.md`).
`legacy/` holds earlier builders that no published number uses (`../legacy/README.md`).
