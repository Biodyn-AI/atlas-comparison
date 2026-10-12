# Decisions

What was decided, why, what it cost, and where it lives in the code. In the order the
work was done. Numbers are the ones the audit checks against the result files.

## 1. One corpus for all ten models

6,000 Tabula Sapiens cells, 2,000 each from immune, kidney and lung tissue, raw counts,
drawn from cellxgene-census v1.18.0 with seed 0. A second draw with seed 1 and the same
protocol is the held-out corpus.

Why: the earlier single-model atlases were built on different corpora (Geneformer on K562,
scGPT on Tabula Sapiens), so any difference between them could be the data. With one corpus,
a difference between two models is a difference between the models.

Cost: cross-species models (UCE, GeneCompass, C2S-Scale) are read on human cells only, and the
atlas says nothing about other tissues or platforms.

Code: `pipeline/atlas_h100/build_ts_corpus.py`.

## 2. Every layer of every model, with five depth-matched read-outs

An SAE is trained at every layer (8 to 33 per model, 172 in all). Where one comparable depth
per model is needed, the layers at 0, 25, 50, 75 and 100 % of depth are used.

Why: the five-point read-out was the earlier design, and it missed the per-layer structure
(mid-network peaks, late collapse of annotation rate). Every layer costs more compute but shows
the profile.

Rule that follows: feature counts are never summed across layers and never read as concepts
coexisting in one representation, because each layer is its own space with its own dictionary.
Section 3.1 of the paper explains why this matters.

Code: `run_model.py --all-layers`, `common/layers.py`.

## 3. The SAE

TopK, k = 32 active features per position, dictionary 4 x d_model, MSE loss, Adam 3e-4,
4 epochs, per-column mean subtraction, unit-norm decoder, seed 0 for NumPy and torch. Training
data: at most 168 gene-token positions from every one of the 6,000 cells, about 1M per layer
(decision 16 explains why it is not a position cap). A feature's genes are ranked by its mean
activation per gene, ties broken stably, zero-activation genes never listed; the catalogue keeps
the top 20.

Why: this is the configuration of the earlier atlases (bio-sae), so features are comparable
with them. It was not swept. Seed variance was measured (three seeds; the permissive mid-layer
core varies by 12 %, 435 to 580 concepts, and the all-layer core by 4 %) and hyperparameters were
not.

What it costs: with more features than a layer has distinct signals, the surplus fills with
near-copies of the strongest signal. Every model except Geneformer-V2 and GeneCompass shows a
block of such features. Section 2 of the paper measures it and Table S12 shows the prediction
result does not rest on it. The account of why some models have the block and others do not
is in Limitations and is explicitly untested.

Health: tGPT's dictionaries have the most dead features (21.3 % on average, 58.6 % at worst),
which is a confound for every tGPT-specific result. Said in Methods; not corrected for.

Code: `pipeline/atlas_h100/common/sae.py`, `pipeline/scripts/sae_health.py`.

## 4. Two annotators, and which one counts

Permissive: a feature's top-5 genes against GO_BP, Reactome, KEGG, STRING and TRRUST, Fisher
exact with BH < 0.05. This was the field default and the earlier atlases' choice.

Calibrated: top-10 genes, at least 3 of them in one curated GO_BP, Reactome or KEGG set of at most
200 genes, no interaction sets, BH < 0.05.

Why two: the permissive annotator fails a random-gene null. Replace every feature's top genes
with random genes and it still annotates almost as often, because the databases are large and
overlapping. Under it the ten models appear to share 1,977 concepts; random genes share 3,570
(z = -24.2). The calibrated annotator was built by grid search over database, set size, minimum
overlap and odds ratio until random genes annotate about 3 % of features against about 9 % for
real ones. Its configuration was fixed on the discovery corpus before the held-out corpus was
touched.

Consequence: every cross-model claim uses the calibrated annotator. The permissive one is kept
on the site for per-model description, labelled illustrative. The site's module labels are also
permissive and are labelled as hints (across all groups a label covers a median 20 % of a group's
features).

Code: `random_null.py`, `recalibrate.py`, `recalibrate2.py`, `recalibrate_final.py`,
`recalibrate_robust.py`; the permissive annotator is `atlas_h100/common/annotate.py`.

## 5. A null behind every claim

Each result is stated against a null built to produce the same result by chance:

| claim | null | permutations |
|---|---|---|
| shared backbone (concepts in >= 8 of 10 models) | uniform random genes; degree-matched random genes (same gene-set membership degree) | 250 each |
| backbone across depth | uniform random genes at each of five depths | 250 |
| held-out replication of the backbone | random-gene backbones on both draws, Jaccard between them | 150 |
| naive core (retracted) | uniform random genes, mid layer and all-layer union | 100; 20 |
| CKA between models | shuffle the cells of one embedding | per pair |
| SAE against PCA | top-k PCA at the same k; an untrained random dictionary | none needed |
| held-out regulatory recovery per model | configuration model: rewire the co-firing graph keeping every gene's degree | 500 |
| corroboration curve, equal-budget re-scoring | the same, all ten graphs per draw | 100 |
| robustness variants, literature, perturbation, publication strata | as above; publication-count-preserving shuffle; responsiveness-matched gene | 200 |

Permutation counts differ by cost and are stated wherever a number derived from them appears.
With N permutations the smallest p is 1 / (N + 1), and most significant results sit at that
floor. Two results sit near the threshold instead (MaxToki at equal budget, p = 0.04; the
corroboration gain in one robustness variant, p = 0.055) and are reported as marginal.

Code: `degree_null.py`, `stats_final.py`, `depth_backbone_fast.py`, `heldout_calibrated.py`,
`alllayer_null.py`, `cka_svd_null.py`, `hypothesis_trrust2.py`.

## 6. What was retracted, and kept as a negative result

- The universal core of about 2,000 concepts shared by all ten models. Fails the random-gene null
  at the mid layer (365 real against 582 random) and on the all-layer union (1,977 against
  3,570). Its held-out "replication" (Jaccard 0.50) is no better than two random draws (0.51).
- Annotation rate as an interpretability score. Random genes annotate almost as often under the
  permissive annotator. The calibrated rate (1.0 to 29.3 % at the mid layer) is the usable one.
- "About 100 % of SAE features are invisible to the top SVD axes." True of random directions at
  this dimensionality too, so it is a property of the dimension, not of the features. The test
  that carries weight is variance explained at matched sparsity: the SAE beats top-32 PCA in all
  ten models, by +0.02 (tGPT) to +0.55 (Geneformer-V2).
- A "new biology" shortlist: genes topping unannotated features in at least five models. Against
  a random feature subset (`novel_null.py`) and a degree-matched label permutation
  (`novel_calibrated.py`) the shortlist is a deficit, not an excess: 0 of 207 candidates survive.
  Consensus tracks database coverage; it does not fill its gaps.

The scripts and result files for all four are kept so the negatives are reproducible.

## 7. What survived

- A backbone of 96 concepts encoded by at least 8 of the 10 models under the calibrated
  annotator: 27.5x the uniform null and 49.2x the degree-matched null, p < 0.004; 66 % specific
  programme biology, 34 % housekeeping by the keyword rule of Table S11; present at every depth
  (55 to 133 concepts, 21.3 to 38.7x); reproduced on the held-out draw, which shares no cell with
  the corpus, on all ten models (Jaccard 0.52 against 0.035, p = 0.02). No concept is in all ten
  models; the all-ten intersection is brittle by construction.
- Most of the backbone is co-expression. A dictionary of seed genes plus their nine most
  correlated genes in the same cells, with no model, recovers 73 to 95 of the 96 concepts with
  random seeds (`coexpr_core_baseline.py`). The backbone is what the data's co-expression
  structure makes recoverable, not something the models add.
- Annotation-free geometry: linear CKA between every pair of models is above its cell-shuffle
  floor (worst pair 43x); the SAE beats top-32 PCA everywhere; tissue is linearly decodable at
  every depth with no gain from an MLP probe. PCA of the raw expression does as well on tissue
  (0.94) and cell type (0.83 to 0.85) as the models' middle layers, and agrees with them in CKA
  (0.18 to 0.63), so this is geometry the data already has (`raw_expression_baseline.py`).
- Features predict held-out regulatory edges (decision 8).

## 8. The held-out prediction test

For each model, pool features over all layers, take each feature's top-10 genes, form every
within-feature pair, and call a pair predicted when it recurs in at least five features. Score
the predicted pairs against TRRUST, which no step of the pipeline used: the co-firing statistic
uses no annotation, and the calibrated annotator uses pathway sets only. The null rewires the
graph while preserving every gene's degree, so neither abundance nor study bias can produce the
effect.

Result: four of ten models pass (UCE 10.2x, Tahoe-x1 9.4x, C2S-Scale 5.5x, Geneformer-V2
2.5x). scGPT passed on the superseded catalogues (11.9x, trained on about 420 cells) and does not
on the corrected ones (1.7x, p = 0.27). Pairs predicted by two models independently are 11.5x
over the null against 4.8x for one model, a precision of 0.23 %.

Two design points that took iteration:

- How many pairs a model can offer is set by its dictionary size, not by quality. The
  equal-budget re-scoring (50,707 pairs, the smallest passing graph) keeps all four passing
  models significant (Tahoe-x1 11.0x, UCE 11.8x, C2S-Scale 8.4x, Geneformer-V2 2.4x); the six
  smaller graphs cannot be scored at that budget. Both versions are reported; only the
  equal-budget one licenses a claim about quality.
- The corroboration curve is reported over all ten models, not over the four that passed,
  because the four were selected on TRRUST and would then be re-scored against it. The
  selected-four numbers (5.4x, 15.4x) are shown only to confirm the direction. The selected
  set is read from `hypothesis_trrust2.json` (p < 0.05), not typed in.

The Poisson whiskers on Fig 5A are there because 60 and 79 recovered edges do not support a
ranking between 10.2x and 9.4x.

Code: `hypothesis_trrust.py`, `hypothesis_trrust2.py`, `hypothesis_trrust3.py`,
`hypothesis_robust.py`, `hypothesis_sizematched.py`, `hypothesis_singlelayer.py`,
`pair_redundancy.py`.

## 9. Two more kinds of evidence for the same pairs

Literature: NCBI gene2pubmed as a dated bulk file (2026-09-11), not live queries. Papers
annotating more than 60 genes are dropped, because without that cap 96 % of all pairs are
"co-mentioned" and the measure says nothing. The null shuffles the gene-paper graph keeping each
gene's paper count. 1,281 of 4,621 pairs co-mentioned against 471 +- 19 (2.7x).

Perturbation: genome-wide CRISPRi Perturb-seq (Replogle 2022), K562 and RPE1. For a pair (A, B)
read the percentile of B's response within perturbation A, so a perturbation that moves
everything cannot score, and compare with a gene matched on its own responsiveness, so a gene
that reacts to everything cannot either. 1.25x in K562, 1.33x in RPE1, both at p = 0.005. The
effect is small on purpose: the screens are cancer and epithelial lines, the corpus is primary
immune, kidney and lung tissue.

Code: `hypothesis_pubmed.py`, `hypothesis_perturb.py`.

## 10. Why no new biology, and why that is about the evidence

Only 140 of 4,621 pairs were never co-mentioned, and only six of those join two well-studied
genes; five of the six involve a long non-coding RNA (GAS5, SNHG6, NEAT1, FENDRR). The analysis
of study bias (`hypothesis_studybias.py`, `study_bias_coverage.py`) shows the models do feature
5,953 genes with fewer than five papers, that the recurrence filter is 7.0x harsher on them, and
that 732 of the 774 protein-coding ones are absent from the Perturb-seq screen. Predictive power does not
decline with how studied a gene is. So the atlas ranks testable links; it cannot confirm or
refute a link about an unstudied gene with these resources, and neither could anything else.

## 11. Model facts corrected against the model papers

Late in the work three entries in the model table were found wrong and corrected everywhere:

- MaxToki reads a cell as rank value encoding (genes ranked by corpus-scaled expression) with
  an autoregressive Llama decoder, not as expression magnitude.
- GeneCompass ranks genes by expression and also embeds the absolute value, and carries four
  knowledge priors; it is not rank-only.
- tGPT has about 120 M parameters by its config (8 layers, d = 1024, vocabulary 21,150), not
  about 50 M.

Everything that reads model size or the rank/non-rank split was recomputed: concepts against
size r = 0.855 (CI 0.68 to 0.984), annotation rate against size 0.169, concentration against
rank tokenisation -0.091. No conclusion changed. The account of the near-duplicate feature
blocks was rewritten, because with the corrected inputs "magnitude input" no longer separates
the models that have a block from the two that do not.

## 12. What the site reads from data

Every number on the site is read from the result files at page-load time, including the
readings under charts. Model display names are applied in one pass. Three panels that were
removed to lighten the page had their code removed too. The feature maps (40 MB) are fetched
the first time the Layer explorer is opened, which brought the page from 61 MB to 12 MB.

Code: `pipeline/atlas_template.html`, `inject_atlas.py`, `atlas_hypothesis_block.py`.

## 13. Things that did not go to plan

- The depth sweep's own mid slice gives 97 concepts, not 96, because for models with an even
  number of layers its index lands one layer from the mid layer used elsewhere. Stated in
  Section 3.2.
- A stray display floor of 0.05 on the log-axis figures drew nulls that never occurred as if
  measured; a stored fold for the all-ten tier was computed with an undisclosed floor of 0.1 on
  the null. Both are now disclosed on the figures.
- `controls.py` used to overwrite the shared `controls.json` and once wiped fourteen blocks
  that other scripts had merged in. It merges now.
- The SAE weight files and activations stay on the GPU host and are not archived (see
  `pipeline/atlas_h100/README.md`).
- The extraction itself was wrong in four ways, found after the first release; decision 16.

## 14. KEGG

KEGG contributes 24 of the 96 backbone concepts. Without KEGG (GO_BP and Reactome only) the
backbone is 72 concepts at 25.7x over the null, so nothing rests on it. KEGG sets are used only as
membership lists via Enrichr; the copyright form is to be cleared before publication.

Code: `kegg_robust.py`.

## 15. The audit

Every number in the paper, the supplement, the figures and the site is re-derived from the
result file that produced it by `audit_all.py` (312 checks). Figures are stamped against the
hashes of their inputs, so a figure older than its data fails. The audit does not check prose
claims, architecture facts (decision 11 was found by reading the model papers) or whether a
figure is legible. Those were checked by hand, three passes, and the fixes are in the git log.

## 16. The extraction was wrong, and everything was re-run (2026-10)

An external read of the code found three faults in the first release; re-running exposed more.

- The position cap. `extract()` stopped after 500,000 gene-token positions taken in corpus order,
  so a model that reads every gene saw very few cells: AIDO.Cell 32, scFoundation about 160,
  Geneformer-V2 and MaxToki about 270, UCE 492, while tGPT (64 tokens per cell) saw all 6,000.
  Feature catalogues, and everything built on them, compared models trained on different cells.
  Now every model reads all 6,000 cells and keeps at most 168 positions from each (seed 0).
- Filler in the top-gene lists. A non-stable descending sort padded features that fire on few
  genes with arbitrary zero-activation genes. Now stable, and zero-activation genes are dropped.
- The SAE seeded NumPy but not torch, so weight initialisation and shuffling were unseeded. The
  published seed control was not affected (its scripts seeded torch themselves).
- UCE's preprocessing cache. UCE skips preprocessing when its output files exist, and the work
  directory had a fixed name, so every UCE run after the first silently reused 2,000 corpus cells;
  the held-out UCE catalogue was built on corpus cells. The cache is now keyed on the cells.
- The held-out draw shared 419 of its 6,000 cells with the corpus. They are removed.
- The site's CKA heatmap labelled blocks 0 to 4 as 0 to 100 % depth. It now reads each model's
  depth-matched blocks. The CKA numbers in the paper came from another file and were right.
- Five blocks of `controls.json` had been pasted in by hand from other result files, so the
  audit compared old text with old copies. `controls_import.py` now builds them.
- Three local steps were missing from the documented order (the `--merge` run of
  `alllayer_concepts.py`, the slow `depth_backbone.py`, `depth_calibrated.py`), the list of
  models carried into the cross-model test was typed in, and the full pair list was written by a
  one-off command. All are now in the scripts and in `REPRODUCE.md`.

Everything downstream of the catalogues was re-run from scratch in the documented order, and the
GPU-side analyses were re-run on the corrected SAEs. What changed: the backbone grew from 78 to
96 concepts and stayed far above its nulls; it still replicates on held-out cells, now on all ten
models; TRRUST prediction holds in four models instead of five (scGPT drops out); literature and
perturbation support got slightly stronger. Release `runs-v2` holds the corrected catalogues;
`runs-v1` keeps the superseded ones.

Code: `pipeline/atlas_h100/common/pipeline.py`, `common/sae.py`, `adapters/uce.py`,
`make_heldout_disjoint.py`, `run_*_fixed.sh`; `pipeline/scripts/controls_import.py`.
