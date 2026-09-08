# Methods

## Corpus & control
One shared corpus for every model: 6,000 Tabula Sapiens cells (2,000 each immune / kidney / lung), raw
counts, HGNC symbols. Running all models on the *same* cells removes the corpus confound that separate
single-model atlases suffer from. Cross-species models (UCE, GeneCompass, C2S) receive human cells only.

## SAE
TopK SAE per layer: k=32, dictionary width d_sae = 4×d_model, MSE loss (no L1), decoder columns unit-normed
each step with per-column mean subtraction, 4 epochs, ~500k token positions/layer. Matches the bio-sae
config so results are comparable to Igor's single-model atlases.

## Depth matching
Models range 8–33 layers. For cross-model comparisons we sample 5 relative depths (0/25/50/75/100%); the
per-layer analyses (depth, tissue, linearity, modules, flow, layer-explorer) use *every* layer.

## Annotation (the join key)
Every feature's top-5 decoder genes → Fisher exact ('greater') + Benjamini-Hochberg < 0.05 against a shared
5-database vocabulary: GO_BP, Reactome, KEGG (gene sets) + STRING (PPI edges ≥700) + TRRUST (kept for the
annotator but **no TF-regulon analysis is surfaced** — the atlas is structural). Term size 5–500. The same
annotator is applied to *all* models (including re-annotating Igor's) to control DB-version confounds.

## Analyses
- **Universality** — concept × model membership; the core = concepts in all models.
- **CKA** — linear Centered Kernel Alignment on per-cell mean-pooled embeddings, at matched relative depth
  (cross-model, dimension-invariant) and within-model layer×layer (drift geometry).
- **SVD vs SAE** — for a layer, SVD/PCA of the residual; a feature is "novel" if its max |cos| to the top
  SVD axes < 0.7 (in superposition). Also matched-sparsity variance: top-k SVD vs the SAE's k-active recon.
- **Linearity** — decode a concept (tissue / cell type) from the residual with a linear (logreg) vs
  nonlinear (MLP) probe; the confound-free signal is the MLP−linear gap at equal dimensionality.
- **Modules** — Pearson co-activation of feature activations → strong-edge graph → greedy-modularity
  communities + spring layout, per layer.
- **Cross-layer flow** — persistence(L→L+1) = % of features in L with a top-5-gene Jaccard>0.3 match in L+1,
  between *every adjacent layer* (not depth-matched jumps that skip transformer blocks and understate it).
- **Emergence** — cumulative concept vocabulary as models are added smallest→largest; correlation of
  concept universality with the minimum model size that encodes it; category skew of scale-gated vs
  universal concepts.
- **Rosetta (convergence)** — % of a model's features with a near-identical top-gene twin in another model.
- **Curriculum** — mean relative depth at which each concept category first appears, across models.
- **Cell-type difficulty** — per-class balanced accuracy of a linear probe at the deepest layer, averaged
  across models — which cell types resist a linear readout everywhere.

## Inductive axis
Each model is a point in a 3-axis design space, which the atlas uses to explain the differences:
- **Tokenization** (rank / magnitude-expression / protein-token / cell-sentence) → how pathway-shaped the
  features are (rank models annotate highest).
- **Objective** (masked vs autoregressive) → representational drift and tissue-binding.
- **Prior** (none / ESM-protein / knowledge-GRN / text) → drift and feature organisation.

**Invariant across all axes**: superposition (~100% novel-to-SVD), linear-representation, the universal
core, smooth adjacent-layer drift, and the scaling of the concept vocabulary — i.e. these come from biology
+ sparse coding, not from any one architecture.

## Caveats
- Top-5 genes/feature (ceiling of Igor's published atlases) underpowers vs top-20; used for cross-model
  fairness.
- Depth profiles are SAE-training-sensitive; the robust findings (universality, superposition, linearity)
  are not.
- The Layer Explorer scatter is capped at 500 features/layer in the hosted (light) build for size; the
  full build is uncapped.
- Cell/set/spatial models (State, X-Cell, CellPLM, Novae) don't fit the per-gene-token contract and are
  excluded — they'd need a different SAE target.
