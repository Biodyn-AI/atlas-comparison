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
4 epochs, at most 500,000 gene-token positions per layer, per-column mean subtraction, unit-norm
decoder, seed 0. A feature's genes are ranked by its mean activation per gene; the catalogue
keeps the top 20.

Why: this is the configuration of the earlier atlases (bio-sae), so features are comparable
with them. It was not swept. Seed variance was measured (three seeds; the mid-layer core moves
by 3 %) and hyperparameters were not.

What it costs: with more features than a layer has distinct signals, the surplus fills with
near-copies of the strongest signal. Every model except Geneformer-V2 and GeneCompass shows a
block of such features. Section 2 of the paper measures it and Table S12 shows the prediction
result does not rest on it. The account of why some models have the block and others do not
is in Limitations and is explicitly untested.

Health: tGPT's dictionaries have the most dead features (22.5 % on average, 58.3 % at worst),
which is a confound for every tGPT-specific result. Said in Methods; not corrected for.

Code: `pipeline/atlas_h100/common/sae.py`, `pipeline/scripts/sae_health.py`.

## 4. Two annotators, and which one counts

Permissive: a feature's top-5 genes against GO_BP, Reactome, KEGG, STRING and TRRUST, Fisher
exact with BH < 0.05. This was the field default and the earlier atlases' choice.

Calibrated: top-10 genes, at least 3 of them in one curated GO_BP, Reactome or KEGG set of at most
200 genes, no interaction sets, BH < 0.05.

Why two: the permissive annotator fails a random-gene null. Replace every feature's top genes
with random genes and it still annotates almost as often, because the databases are large and
overlapping. Under it the ten models appear to share 2,267 concepts; random genes share 3,563
(z = -18.4). The calibrated annotator was built by grid search over database, set size, minimum
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
  at the mid layer (353 real against 589 random) and on the all-layer union (2,267 against
  3,563). Its held-out "replication" (Jaccard 0.52) equals two random draws (0.50).
- Annotation rate as an interpretability score. Random genes annotate almost as often under the
  permissive annotator. The calibrated rate (1.3 to 32.2 % at the mid layer) is the usable one.
- "About 100 % of SAE features are invisible to the top SVD axes." True of random directions at
  this dimensionality too, so it is a property of the dimension, not of the features. The test
  that carries weight is variance explained at matched sparsity: the SAE beats top-32 PCA in all
  ten models, by +0.02 (tGPT) to +0.53 (Tahoe-x1).
- A "new biology" shortlist: genes topping unannotated features in at least five models. Against
  a random feature subset (`novel_null.py`) and a degree-matched label permutation
  (`novel_calibrated.py`) the shortlist is a deficit, not an excess: 0 of 207 candidates survive.
  Consensus tracks database coverage; it does not fill its gaps.

The scripts and result files for all four are kept so the negatives are reproducible.

## 7. What survived

- A backbone of 78 concepts encoded by at least 8 of the 10 models under the calibrated
  annotator: 22.5x the uniform null and 59.3x the degree-matched null, p < 0.004; 63 % specific
  programme biology, 37 % housekeeping by the keyword rule of Table S11; present at every depth
  (41 to 102 concepts, 17.6 to 30.4x); reproduced on the held-out draw (Jaccard 0.50 against 0.048,
  p < 0.007) on nine models, because scGPT could not be re-extracted there (decision 13).
  Only two concepts are in all ten models; the all-ten intersection is brittle by construction.
- Annotation-free geometry: linear CKA between every pair of models is above its cell-shuffle
  floor (worst pair 43x); the SAE beats top-32 PCA everywhere; tissue is linearly decodable at
  every depth with no gain from an MLP probe.
- Features predict held-out regulatory edges (decision 8).

## 8. The held-out prediction test

For each model, pool features over all layers, take each feature's top-10 genes, form every
within-feature pair, and call a pair predicted when it recurs in at least five features. Score
the predicted pairs against TRRUST, which no step of the pipeline used: the co-firing statistic
uses no annotation, and the calibrated annotator uses pathway sets only. The null rewires the
graph while preserving every gene's degree, so neither abundance nor study bias can produce the
effect.

Result: five of ten models pass (Tahoe-x1 13.3x, scGPT 11.9x, UCE 10.5x, C2S-Scale 4.1x,
Geneformer-V2 3.9x). Pairs predicted by two models independently are 20.8x over the null against
4.8x for one model, a precision of 0.21 %.

Two design points that took iteration:

- How many pairs a model can offer is set by its dictionary size, not by quality. The
  equal-budget re-scoring (8,631 pairs, the smallest passing graph) changes the order: UCE loses
  significance, MaxToki gains it, AIDO.Cell and scFoundation cannot be scored at all. Both
  versions are reported; only the equal-budget one licenses a claim about quality.
- The corroboration curve is reported over all ten models, not over the five that passed,
  because the five were selected on TRRUST and would then be re-scored against it. The
  selected-five numbers (5.3x, 28.4x) are shown only to confirm the direction.

The Poisson whiskers on Fig 5A are there because 59 and 10 recovered edges do not support a
ranking between 13.3x and 11.9x.

Code: `hypothesis_trrust.py`, `hypothesis_trrust2.py`, `hypothesis_trrust3.py`,
`hypothesis_robust.py`, `hypothesis_sizematched.py`, `hypothesis_singlelayer.py`,
`pair_redundancy.py`.

## 9. Two more kinds of evidence for the same pairs

Literature: NCBI gene2pubmed as a dated bulk file (2026-09-11), not live queries. Papers
annotating more than 60 genes are dropped, because without that cap 96 % of all pairs are
"co-mentioned" and the measure says nothing. The null shuffles the gene-paper graph keeping each
gene's paper count. 1,289 of 4,111 pairs co-mentioned against 591 +- 20 (2.2x).

Perturbation: genome-wide CRISPRi Perturb-seq (Replogle 2022), K562 and RPE1. For a pair (A, B)
read the percentile of B's response within perturbation A, so a perturbation that moves
everything cannot score, and compare with a gene matched on its own responsiveness, so a gene
that reacts to everything cannot either. 1.20x in K562, 1.28x in RPE1, both at p = 0.005. The
effect is small on purpose: the screens are cancer and epithelial lines, the corpus is primary
immune, kidney and lung tissue.

Code: `hypothesis_pubmed.py`, `hypothesis_perturb.py`.

## 10. Why no new biology, and why that is about the evidence

Only 171 of 4,111 pairs were never co-mentioned, and only four of those join two well-studied
genes; all four involve a long non-coding RNA. The analysis of study bias
(`hypothesis_studybias.py`, `study_bias_coverage.py`) shows the models do feature 4,003 genes
with fewer than five papers, that the recurrence filter is 4.8x harsher on them, and that 752 of
the 793 protein-coding ones are absent from the Perturb-seq screen. Predictive power does not
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

- scGPT could not be re-extracted on the held-out draw (a torchtext ABI failure in its
  dependency stack). The replication therefore runs on nine models and reproduces the
  >= 7-of-9 backbone (81 concepts), which is the same proportion but not literally the
  >= 8-of-10 set of 78. Stated in Section 3.2.
- The depth sweep's own mid slice gives 75 concepts, not 78, because for models with an even
  number of layers its index lands one layer from the mid layer used elsewhere. Stated in
  Section 3.2.
- A stray display floor of 0.05 on the log-axis figures drew nulls that never occurred as if
  measured; a stored fold for the all-ten tier was computed with an undisclosed floor of 0.1 on
  the null. Both are now disclosed on the figures.
- `controls.py` used to overwrite the shared `controls.json` and once wiped fourteen blocks
  that other scripts had merged in. It merges now.
- The SAE weight files were not copied back from the GPU host and are not archived (see
  `pipeline/atlas_h100/README.md`).

## 14. KEGG

KEGG contributes 22 of the 78 backbone concepts. Without KEGG (GO_BP and Reactome only) the
backbone is 56 concepts at 19x over the null, so nothing rests on it. KEGG sets are used only as
membership lists via Enrichr; the copyright form is to be cleared before publication.

Code: `kegg_robust.py`.

## 15. The audit

Every number in the paper, the supplement, the figures and the site is re-derived from the
result file that produced it by `audit_all.py` (312 checks). Figures are stamped against the
hashes of their inputs, so a figure older than its data fails. The audit does not check prose
claims, architecture facts (decision 11 was found by reading the model papers) or whether a
figure is legible. Those were checked by hand, three passes, and the fixes are in the git log.
