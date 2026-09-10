# Comparative SAE Feature Atlas

A cross-model interpretability atlas for **single-cell foundation models**. We train sparse
autoencoders (SAEs) on the residual stream of many single-cell FMs on **one shared, tissue-controlled
human corpus**, annotate every feature against the same biological vocabulary, and compare *what* each
model organises and *how*. It extends the single-model atlases of Biodyn-AI
([bio-sae](https://github.com/Biodyn-AI/bio-sae)) into one comparative frame.

**▶ Live atlas:** open `index.html` (self-contained, full-resolution ~61 MB — heavy but complete) — or host it on GitHub Pages (below). It embeds all data; no server or build needed to view.

---

## What's inside

The page is organised as four tabs — *Overview*, *① Choose a model*, *② Compare*, *③ Genes & biology* — plus a
*Findings* tab holding the full analysis set. Every panel pairs a chart with a plain-language reading.

| Section | Question | Status |
|---|---|---|
| **Controls** | every null behind every claim, including the ones that failed | **read this first** |
| Universality | how many models share each concept, calibrated, against its null | tested |
| CKA | representational similarity across all 45 model pairs, vs a cell-shuffle null | tested |
| Coverage | annotation rate, concept count, source mix per model | tested |
| Depth | how concept richness changes layer by layer | tested |
| SVD vs SAE | variance explained at matched sparsity k = 32, SAE vs top-k PCA | tested |
| Linearity | is cell identity linearly readable? (linear vs MLP probe gap), per layer | tested |
| Gene Search | which models encode a given gene, and under what concept | descriptive |
| Layer Explorer | UMAP / t-SNE map of features per layer, per model | descriptive |
| Modules | co-activation communities per layer (force-graph) | descriptive |
| Cross-layer flow | feature persistence between **every adjacent layer** | descriptive |
| Tissue | how tissue-specific features become with depth | descriptive |
| Scale · Emergence · Economy · Axis | concept-acquisition curve vs model size; architecture↔feature associations | **underpowered** (n = 10 non-independent models) |
| Models | roster with params, **training species**, inductive axis | reference |

"Descriptive" panels are exploratory read-outs without a null; "underpowered" ones have bootstrap CIs of
roughly ±1 and are not findings. Only the "tested" rows carry the claims in the manuscript.

## Model roster

Run on one shared corpus (6,000 Tabula Sapiens cells: immune + kidney + lung; cellxgene-census v1.18.0,
seed 0, held-out draw seed 1), depth-matched layers, TopK-SAE (k = 32, dictionary 4×d_model).

Two annotators are used, and the distinction matters:

| | Rule | Status |
|---|---|---|
| **Calibrated** | ≥3 of a feature's top-**10** genes in one curated GO_BP / Reactome / KEGG set of ≤200 genes, no PPI, hypergeometric + BH<0.05 | **the headline annotator** — every reported concept count uses it |
| Permissive | top-**5** genes vs GO_BP / Reactome / KEGG / STRING / TRRUST, BH<0.05 | illustrative only; it produces the database artefact documented above and must never be read as evidence of agreement |

| Model | Params | Species | Tokenization / objective / prior |
|---|---|---|---|
| AIDO.Cell | 10M | human | expression, MLM |
| scGPT | ~50M | human | expression + MLM |
| tGPT | ~50M | human | rank + autoregressive |
| scFoundation | 100M | human | read-depth MAE |
| GeneCompass | 104M | human+mouse | knowledge / GRN-prior BERT |
| MaxToki | 217M | human | temporal Llama (magnitude) |
| Geneformer-V2 | 316M | human | rank + MLM |
| UCE | 650M | multi-species | ESM protein-token prior |
| C2S-Scale | 2B | human+mouse | cell-sentence LLM (Gemma-2) |
| Tahoe-x1 | 3B | human | expression + MLM (MosaicX) |

Cross-species models (UCE, GeneCompass, C2S) are run on the **human** corpus and only their human-gene
features are read — the atlas stays a human atlas.

## Headline findings

Every number below is the one reported in the manuscript, with its control. Where a result did not survive
its control, that is stated rather than omitted.

**The main result is a caution.** Cross-model "universality" measured by naive gene-set annotation of SAE
features is mostly a property of the gene-set *databases*, not of the models. Any gene list hits the large,
overlapping databases, so random genes annotate at nearly the real rate.

- **The naive universal core is retracted.** Pooled across layers, the permissive annotator reports 2,267
  concepts shared by all ten models. Running the identical null on that exact construction gives a
  *random-gene* core of **3,563 ± 70** against the real 2,267 — z = −18.4, i.e. **0.64× the null**, with the
  random core larger in all 20 permutations. At a single mid layer the same inversion holds (real 353 vs null
  589, z = −6.1). The apparent held-out "replication" is no better: Jaccard 0.52 against 0.50 from two random
  draws.
- **A calibrated backbone is real but modest.** Requiring ≥3 of a feature's top-10 genes in one specific
  curated pathway (≤200 genes, no PPI) drops the random-gene annotation rate to ~3% versus ~9% for real
  features. Under it the strict all-ten core is **near-empty (2 concepts)**, and the defensible result is a
  backbone of **~78 concepts shared by ≥8 of 10 models** (50–126 across five calibrated configurations).
- **It clears both nulls.** 22× the uniform random-gene null (78 vs 3.5 ± 2.0, z = 36.6) and **59× a
  degree-matched null** (78 vs 1.3 ± 1.3, z = 61.6). No permutation of 250 reached the observed value, so
  p < 0.004 at the floor.
- **It is depth-robust and replicates.** 41–102 concepts at 16.7–27.6× over the null across the full depth
  sweep, richest early-to-middle. Membership does drift with depth (Jaccard 0.34–0.48 versus the mid-layer
  set), so the ~78 are one depth's slice of a depth-invariant phenomenon, not a fixed list. On independent
  held-out cells, backbone Jaccard 0.50 versus a 0.05 two-draw baseline (max 0.33, p < 0.007). About **63% is
  specific programme biology** (antigen presentation/MHC-II, cytokine signalling, defence response,
  muscle/cardiac signalling) and ~37% housekeeping.
- **Geometry agrees, without using the annotator at all.** Linear CKA with a cell-shuffle null over **all 45
  model pairs**: real 0.09–0.84 (median 0.49) against a floor of 0.001–0.009, **every pair above its own floor
  by more than tenfold**. Descriptively, AIDO.Cell is a consistent outlier and {MaxToki, UCE, scGPT, C2S}
  cluster tightly.
- **The SAE beats a linear summary — by a margin that varies a lot.** At matched per-sample sparsity k = 32
  the SAE explains 0.61–0.98 of residual variance versus 0.23–0.96 for the best fixed rank-32 subspace
  (top-k PCA). It is ahead in all ten models, but the gap runs from **+0.53 (Tahoe-x1) down to +0.02 (tGPT)**,
  whose residual stream is already close to rank-32. So "SAE ≫ PCA" holds for most, not all, of these models.
  An untrained random dictionary of the same shape explains *negative* variance, so the advantage comes from
  trained adaptive allocation, not from having more directions.
- **Cell identity is linearly decodable at every depth**, with no gain from a non-linear probe.

**What this means for model choice.** A downstream analysis inherits far more of one model's own vocabulary
than of the shared backbone — the models share basic machinery, and most of what each learns is
model-specific. Which model you pick shapes what your analysis can see.

**Not claimed here.** The "% of SAE features invisible to the top SVD axes" figure is ~100% for *random*
directions too at this dimensionality, so it is a dimensional artefact and is **not** evidence of
superposition; the load-bearing comparison is variance-explained at matched k, above. Architecture↔feature
associations (tokenization → pathway-shapedness, objective → tissue-binding) rest on n = 10 non-independent
models, are underpowered (bootstrap CIs ≈ ±1), and are reported as **descriptive supplementary material, not
findings** — the same applies to the scale/emergence, convergence and depth-curriculum panels. Validation on a
truly independent dataset (different platform or tissue) and a hyperparameter sweep beyond seed variance are
not yet done.

Live numbers are in the atlas (`Universality`, `Controls`, `CKA`, `SVD` sections).

**Tested and rejected.** An earlier version of the atlas carried a "new biology" panel: genes that top an
*unannotated* SAE feature in ≥5 of 10 models, offered as a shortlist of under-annotated biology. It does not
survive a null and has been removed. `pipeline/scripts/novel_null.py` compares the shortlist against an
equally-sized random subset of features (real 50 vs null 65.5 ± 6.4, z = −2.4); `pipeline/scripts/novel_calibrated.py`
redoes the selection with the calibrated annotator and a degree-matched null that permutes the
annotated/unannotated label within each model, holding every gene's degree exactly fixed (real 207 vs null
248.8 ± 6.2, z = −6.78; **0 of 207 candidates survive at BH q ≤ 0.05**, 2,000 permutations). Both directions show a
*deficit*, not an excess: cross-model agreement concentrates on genes the databases already cover well —
annotation coverage rises monotonically with the number of models agreeing on a gene (59% → 87% of genes carry
≥1 curated term; median 3 → 11 terms). Consensus tracks database coverage; it does not fill its gaps.

## Repository layout

```
index.html               the atlas (open directly, or serve via GitHub Pages)
data/atlas_full_notf.json the assembled cross-model data the page embeds
pipeline/
  atlas_h100/            extraction + SAE + per-model analyses
    adapters/            one adapter per model (base.py = the contract)
    run_model.py         residual extraction + TopK-SAE + catalog for a model
    common/              SAE, annotation, depth-matching, pipeline
    {cell_cka,cka_layers,layer_explorer,modules_alllayers,depth_profile,
     tissue_from_emb,nonlinearity,svd_vs_sae,build_explorer_slim,
     flow_alllayers,celltype_difficulty}.py   the per-model / aggregate analyses
  scripts/               assembly (run locally, CPU)
    reannotate_string.py     uniform 5-DB annotation → matrix + coverage
    alllayer_concepts.py     distinct concepts across depth-matched layers
    module_themes.py         theme×model matrix
    flow_ts3.py              depth-matched flow (all-layers = flow_alllayers.py)
    genes_search_ts3.py      cross-model gene index
    findings.py              emergence / rosetta / curriculum (descriptive)
    atlas_assemble.py        → data/atlas_full_notf.json (the source of truth)
    inject_atlas.py          embed the data blocks into index.html
    --- controls: the nulls behind the claims ---
    degree_null.py           degree-matched random-gene null for the backbone (the strongest test)
    alllayer_null.py         random-gene null on the all-layer construction (retracts the naive core)
    novel_calibrated.py      calibrated + degree-matched null for the "new biology" idea — negative
    novel_null.py            random-feature-subset null for the same — also negative
docs/METHODS.md          pipeline + inductive-axis writeup
```

## Reproduce

Per-model extraction runs on a GPU (H100-class); the assembly is CPU-only.

1. **Per model** (GPU): `python pipeline/atlas_h100/run_model.py --model <M> --corpus <ts3.h5ad> --out out_alllayers --all-layers`
   then the per-model analyses (`cell_cka --all-layers`, `layer_explorer`, `svd_vs_sae`).
2. **Aggregates** (GPU/CPU): `modules_alllayers --all`, `depth_profile --all`, `cka_layers --all`,
   `cell_cka --cka`, `nonlinearity --all --layers all`, `tissue_from_emb --all`, `build_explorer_slim`,
   `celltype_difficulty`, `flow_alllayers`.
3. **Assembly** (CPU, local): `reannotate_string` → `alllayer_concepts` → `module_themes` →
   `genes_search_ts3` → `findings` → `atlas_assemble` → `inject_atlas` (rebuilds `index.html`).

Adding a model = write `pipeline/atlas_h100/adapters/<m>.py` (implement `iter_activations`), register it in
`run_model.py`, add it to the `MODELS`/`PARAMS`/`AXIS` lists, run steps 1–3.

## Host on GitHub Pages

`index.html` is fully self-contained, so hosting is one setting:

1. Create a repo and push this folder.
2. GitHub → **Settings → Pages → Source: Deploy from a branch → `main` / root**.
3. Your atlas is live at `https://<user>.github.io/<repo>/`.

`index.html` is the full-resolution build (every feature, no per-layer cap, ~61 MB) — under GitHub's 100 MB
file limit, so it commits and serves directly. It's heavy to load in a browser; if you want a snappier page,
`inject_atlas.py` can emit a capped (~16–17 MB) build from the same data.

## Credit

Built on the method and single-model SAE atlases developed at Biodyn-AI
([bio-sae](https://github.com/Biodyn-AI/bio-sae): Geneformer / scGPT / Novae / MaxToki / C2S). This repo is the
cross-model extension: one shared corpus, one calibrated annotator, and an explicit null behind every claim.
