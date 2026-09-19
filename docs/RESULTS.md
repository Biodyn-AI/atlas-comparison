# Result files

Every file in `outputs/atlas/comparative/`, the script that writes it and the scripts that read it. Sizes in KB. The manuscript, the supplement and the site's data blocks are built from these files and nothing else; `pipeline/scripts/audit_all.py` checks that.

| file | KB | written by | read by | holds |
|---|---|---|---|---|
| `alllayer_concepts.json` | 1 | `alllayer_concepts.py` | `atlas_assemble.py` | distinct concepts per model across the depth-matched layers |
| `alllayer_null.json` | 0 | `alllayer_null.py` | `audit_manuscript_core.py`, `make_supp_tables.py`, `make_figures.py` | the same null on the all-layer union, 20 permutations |
| `atlas_full_notf.json` | 356 | `atlas_assemble.py` | `controls4.py`, `atlas_hypothesis_block.py`, `controls.py`, `make_fig14.py`, `inject_atlas.py` … | everything the site embeds: coverage, universality, similarity, depth, tissue, CKA, SVD, controls, hypothesis block |
| `backbone_terms.json` | 25 | `backbone_terms.py` | `atlas_hypothesis_block.py`, `make_figures.py` | the calibrated backbone concepts by sharing tier (>=7, >=8, >=9, >=10 of 10 models) |
| `celltype_difficulty.json` | 5 | `celltype_difficulty.py` | `atlas_assemble.py` | Per-cell-type decodability across models: which cell types are HARDEST to read out linearly, averaged |
| `circuits_adjacent.json` | 3 | `circuits_adjacent.py` | `atlas_assemble.py` | Adjacent-layer feature circuits — the methodologically-correct version (coupling between CONSECUTIVE |
| `cka_layers.json` | 26 | `cka_layers.py` | `atlas_assemble.py` | Within-model layer x layer CKA — how each model's representation evolves across its OWN depth. |
| `cka_svd_null.json` | 5 | `cka_svd_null.py` | `atlas_hypothesis_block.py`, `audit_manuscript_derived.py` | cell-shuffle null for CKA and random-direction null for the SVD comparison, all ten models |
| `cka_ts3.json` | 6 | `cell_cka.py` | `atlas_assemble.py` | Representational similarity (linear CKA) between the seven models. Each model's residual is |
| `controls.json` | 30 | `controls4.py`, `controls.py`, `controls2.py`, `controls3.py` | `audit_manuscript_core.py`, `audit_supplement.py`, `atlas_assemble.py`, `make_supp_tables.py`, `make_figures.py` … | every null, CI and control the paper reports (random-gene, degree-matched, held-out, seeds, capacity, KEGG, dedup) |
| `degree_null.json` | 0 | `degree_null.py` |  | the degree-matched random-gene null for the backbone, 250 permutations |
| `depth_alllayers.json` | 19 | `depth_profile.py` | `atlas_assemble.py` | Per-layer profile for ALL layers (annotation rate, semantic richness, SAE variance, dead fraction) — |
| `depth_backbone.json` | 25 | `depth_backbone.py`, `depth_backbone_fast.py` | `audit_manuscript_core.py`, `audit_supplement.py`, `make_supp_tables.py`, `make_figS2.py` | the backbone at five read-out depths against its null, 250 permutations |
| `depth_calibrated.json` | 6 | `depth_calibrated.py` | `atlas_assemble.py` | annotation rate at every layer under the calibrated annotator |
| `depth_ts3.json` | 2 | `atlas_build_ts3.py` |  | Build the updated comparative matrix from the TS-3-tissue re-run (review feedback applied: |
| `depth_ts3_string.json` | 7 | `reannotate_string.py` |  | Re-annotate the TS-3-tissue catalogs at top-5 with STRING added (5 databases, matching the prior atlases: |
| `explorer_slim_full.json` | 50,637 | `build_explorer_slim.py` | `inject_atlas.py` | the feature maps of the Layer explorer (served to the site as data/explorer.json) |
| `explorer_slim_light.json` | 5,353 | `build_explorer_slim.py` | `inject_atlas.py` | the same, capped per layer, for a lighter local build |
| `findings.json` | 3 | `findings.py` | `atlas_assemble.py` | emergence, convergence and curriculum descriptives (underpowered, supplementary) |
| `flow_alllayers.json` | 6 | `flow_alllayers.py` | `atlas_assemble.py` | Cross-layer feature flow over ALL ADJACENT layers (methodologically better than depth-matched, which |
| `flow_ts3.json` | 1 | `flow_ts3.py` |  | Cross-layer feature flow (the persistence view). For each consecutive pair of the 5 depth-matched |
| `freq_ts3.json` | 22,443 | `cell_cka.py` | `atlas_assemble.py`, `circuits.py` | feature firing frequencies |
| `genes_ts3.json` | 2,769 | `genes_search_ts3.py` | `atlas_assemble.py`, `inject_atlas.py` | the gene-search index: which models feature each gene and under which concept |
| `gsea_annot.json` | 1 | `gsea_annot.py` | `controls.py`, `atlas_assemble.py`, `gsea_prerank.py` | rank-based annotation as a check on the cutoff-based one |
| `hardcell_agreement.json` | 0 | `hardcell_agreement.py` | `atlas_assemble.py` | Do models agree on which INDIVIDUAL cells are hard? For each model, a stratified-CV linear cell-type |
| `heldout.json` | 1 | `heldout_compare.py` |  | held-out replication of the permissive core (fails) |
| `heldout_calibrated.json` | 2 | `heldout_calibrated.py` |  | held-out replication of the calibrated backbone (passes), nine models |
| `hypothesis_final.json` | 16 | `hypothesis_trrust3.py` | `atlas_hypothesis_block.py`, `make_supp_tables.py`, `audit_manuscript_derived.py`, `hypothesis_singlelayer.py`, `make_fig5.py` … | the validated models and the released prediction list |
| `hypothesis_pairs_full.json` | 550 | `hypothesis_pubmed.py` | `atlas_hypothesis_block.py`, `make_supp_tables.py`, `study_bias_coverage.py`, `hypothesis_perturb.py` | all 4,349 released pairs with models, weight, co-mentions, same-family flag |
| `hypothesis_perturb.json` | 1 | `hypothesis_perturb.py` | `atlas_hypothesis_block.py`, `make_supp_tables.py`, `audit_manuscript_derived.py`, `make_fig5.py`, `audit_manuscript.py` | the Perturb-seq test in K562 and RPE1 |
| `hypothesis_pubmed.json` | 29 | `hypothesis_pubmed.py` | `atlas_hypothesis_block.py`, `make_supp_tables.py`, `audit_manuscript_derived.py`, `make_fig5.py`, `audit_manuscript.py` | literature co-mention against a publication-count-preserving null; novelty counts |
| `hypothesis_robust.json` | 1 | `hypothesis_robust.py` | `atlas_hypothesis_block.py`, `make_supp_tables.py`, `audit_manuscript.py` | the five robustness variants of the held-out test |
| `hypothesis_singlelayer.json` | 2 | `hypothesis_singlelayer.py` | `audit_manuscript.py` | the held-out test with a single-layer co-firing graph |
| `hypothesis_sizematched.json` | 2 | `hypothesis_sizematched.py` | `atlas_hypothesis_block.py`, `make_supp_tables.py`, `audit_manuscript_derived.py`, `make_fig5.py`, `audit_manuscript.py` | every model re-scored on an equal budget of 8,631 pairs |
| `hypothesis_studybias.json` | 2 | `hypothesis_studybias.py` | `audit_manuscript.py` | predictive power by publication stratum; the recurrence-threshold trade |
| `hypothesis_trrust.json` | 8 | `hypothesis_trrust.py` |  | stage 1 of the held-out test: per-model co-firing graphs against TRRUST, analytic null |
| `hypothesis_trrust2.json` | 13 | `hypothesis_trrust2.py` | `atlas_hypothesis_block.py`, `hypothesis_sizematched.py`, `make_supp_tables.py`, `audit_manuscript_derived.py`, `make_fig5.py` … | per-model recovery with 500 rewirings; the corroboration curve over all ten models |
| `kegg_robust.json` | 1 | `kegg_robust.py` | `audit_manuscript_core.py`, `make_supp_tables.py` | the backbone with GO_BP and Reactome only |
| `matrix_alllayer.json` | 19,242 | `alllayer_matrix.py` | `controls.py`, `atlas_assemble.py`, `controls2.py`, `controls3.py` | concept x model matrix over every layer, permissive annotator |
| `matrix_ts3.json` | 7,522 | `atlas_build_ts3.py` |  | Build the updated comparative matrix from the TS-3-tissue re-run (review feedback applied: |
| `matrix_ts3_string.json` | 16,947 | `reannotate_string.py` | `findings.py`, `atlas_assemble.py`, `gsea_annot.py`, `genes_search_ts3.py`, `alllayer_matrix.py` … | concept x model matrix at the five depth-matched layers, permissive annotator (top-5, five databases) |
| `module_themes.json` | 0 | `module_themes.py` | `atlas_assemble.py` | communities mapped to nine coarse biological themes per model |
| `modules_alllayers.json` | 9,036 | `modules_alllayers.py` | `pair_redundancy.py`, `module_themes.py`, `inject_atlas.py`, `build_explorer_slim.py` | co-activation graphs and communities per layer |
| `nonlinearity_alllayers.json` | 26 | GPU side (atlas_h100), copied back | `atlas_assemble.py`, `make_fig14.py` | linear and MLP decodability of tissue and cell type at every layer |
| `novel_calibrated.json` | 0 | `novel_calibrated.py` |  | the degree-matched label permutation for the same shortlist |
| `novel_null.json` | 0 | `novel_null.py` |  | the random-feature-subset null for the retired new-biology shortlist |
| `pair_redundancy.json` | 4 | `pair_redundancy.py` | `atlas_hypothesis_block.py`, `make_supp_tables.py`, `audit_manuscript.py` | near-duplicate feature blocks per model and whether the prediction result rests on them (Table S12) |
| `perlayer_concepts.json` | 33,074 | `alllayer_matrix.py` | `controls.py` | the concept set of every model at every layer, permissive annotator |
| `random_null.json` | 0 | `random_null.py` |  | the mid-layer random-gene null that retracts the naive core |
| `recalibrate.json` | 0 | `recalibrate.py` |  | first grid over calibrated annotator settings against the random-gene null |
| `recalibrate2.json` | 1 | `recalibrate2.py` |  | second grid |
| `recalibrate_final.json` | 2 | `recalibrate_final.py` |  | the chosen calibrated annotator and its backbone |
| `recalibrate_robust.json` | 3 | `recalibrate_robust.py` | `make_supp_tables.py`, `audit_manuscript_derived.py` | the backbone across five calibrated configurations and thresholds; the housekeeping split |
| `sae_health.json` | 1 | `sae_health.py` | `audit_manuscript_core.py`, `audit_supplement.py`, `make_supp_tables.py`, `inject_atlas.py` | dead-feature fraction and decoder cosine for all 172 model-layers |
| `stats_final.json` | 0 | `stats_final.py` |  | the backbone against the uniform null and the held-out replication test |
| `study_bias_coverage.json` | 0 | `study_bias_coverage.py` | `audit_manuscript_derived.py`, `audit_manuscript.py` | how many featured genes any evidence source can check |
| `svd_projection_k.json` | 9 | `svd_projection_k.py` | `make_figS1.py` | projection of SAE directions onto top-k PCA subspaces, k = 1 to 256 |
| `tissue_alllayers.json` | 19 | `tissue_from_emb.py` | `atlas_assemble.py` | Per-layer tissue specificity from the all-layers cell embeddings (cell_cka --all-layers). Each sae_L is |
| `topn_sweep.json` | 2 | `topn_sweep.py` | `controls.py`, `atlas_assemble.py` | annotation at top-5, 10, 15 and 20 genes |

## Figures

`figures/` holds the seven manuscript figures (PNG at 300 dpi and PDF) and `.sources.json`, the manifest `audit_figures.py` stamps with the hashes of each plotting script's inputs.

## Not in the repository

Files no script reads were left out (a 97 MB all-layer seed-variance dump whose summary is in `controls.json`, and intermediates of earlier builds). The per-run catalogues, annotations and external inputs are in the release archives; see `REPRODUCE.md`.
