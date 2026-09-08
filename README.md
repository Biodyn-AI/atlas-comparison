# Comparative SAE Feature Atlas

A cross-model interpretability atlas for **single-cell foundation models (FMs)**. We train sparse
autoencoders (SAEs) on the residual stream of ten single-cell FMs on **one shared, tissue-controlled human
corpus**, annotate every feature against the same biological vocabulary, and compare *what* each model
organises and *how* — with an explicit null model so the cross-model agreement we report is real, not an
artefact of the gene-set databases. It extends the single-model atlases of Ihor Kendiukhov / Biodyn-AI
([bio-sae](https://github.com/Biodyn-AI/bio-sae)) into one calibrated comparative frame.

**▶ Live atlas:** open `index.html` (self-contained, full-resolution ~61 MB — heavy but complete; no server
or build needed), or host it on GitHub Pages (see below).

---

## The headline, honestly

The naive way to give SAE features biological meaning — test a feature's top genes for over-representation
in curated gene sets — is **not safe for cross-model universality claims**. Curated databases are large and
overlapping, so almost any gene list annotates, and a permissive annotator manufactures a large apparent
"universal core" shared by all ten models. That core **fails a random-gene null** (random genes reproduce it),
so it is a property of the databases, not the models.

With a **calibrated annotator + null**, a small but genuine signal survives:

- **A calibrated shared backbone of ~78 concepts** is annotated in **≥8 of 10** models — **22×** above a
  uniform random-gene null and **59×** above a degree-matched null (empirical *p* < 0.004), and it
  **replicates on a held-out cell sample** (backbone Jaccard 0.50 vs a 0.05 two-draw baseline, *p* < 0.007).
- **~63 % of the backbone is specific programme biology** (antigen presentation / MHC-II, cytokine
  signalling, defence response), not just housekeeping.
- It is **robust**: to the annotator configuration (fold 2.3–166× across five), to **read-out depth**
  (significant at every depth, 17–28× over null; exact membership drifts with depth), to dropping KEGG
  (56 concepts at 19×), and across ≥3 SAE seeds (core CV ≈ 3 %).
- **Annotation-free geometry agrees**: cross-model CKA runs 0.12–0.92 (far above a 0.002–0.009 cell-shuffle
  floor), and at matched per-sample sparsity the SAE explains 0.61–0.87 of residual variance vs 0.23–0.44
  for top-*k* PCA (an untrained random dictionary is negative); SAE directions are *reachable* by SVD only
  with many axes (projection-across-*k*), so the edge is sparse allocation, not directions SVD can't span.

The practical message is a **caution + a resource**: report cross-model SAE-feature agreement against an
explicit random- (ideally degree-matched) gene null, and here is a null-controlled comparative atlas of ten
FMs on a common corpus.

## What's inside the atlas

A single interactive page with, for each analysis, a chart + a plain-language reading:

| Section | Question |
|---|---|
| Universality | how many models share each concept — permissive core (artefact) vs the calibrated backbone vs null |
| Coverage | annotation rate, concept count, source mix per model |
| Depth | how annotation / concept richness changes layer by layer |
| Tissue | how tissue-specific features become with depth (linear vs MLP probe) |
| SVD vs SAE | superposition — SAE vs matched-capacity top-*k* PCA, and projection across *k* |
| Modules | co-activation communities per layer (force-graph) |
| Cross-layer flow | feature persistence between adjacent layers |
| Layer Explorer | UMAP / t-SNE map of features per layer, per model (colour = module / SVD / freq) |
| Gene Search | which models encode a given gene, at what depth, and under what concept |
| CKA | representational similarity across models (matched depth) + within-model layer×layer |
| Controls | the nulls and robustness checks behind every headline number |
| Models | roster with params, training species, inductive axis |

## Model roster

One shared corpus (6,000 Tabula Sapiens cells: immune + kidney + lung, raw counts), depth-matched layers,
TopK-SAE (k = 32, dictionary = 4× the residual dim). **Two annotators**: a *permissive* field-default one
(top-5 genes, 5 databases incl. STRING) used only to demonstrate the artefact, and the **calibrated** main
annotator (top-10 genes, ≥3 in a curated GO / Reactome / KEGG set ≤200, no PPI, Fisher 'greater' + BH<0.05).

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
| Tahoe-x1 | 3B | human | expression + MLM |

Cross-species models (UCE, GeneCompass, C2S) are run on the **human** corpus and only their human-gene
features are read — the atlas stays a human atlas.

## Repository layout

```
index.html                     the atlas (open directly, or serve via GitHub Pages)
data/atlas_full_notf.json      the assembled cross-model data the page embeds
data/controls.json             every reported null / CI / robustness result (source of truth)
docs/METHODS.md                pipeline + inductive-axis writeup
pipeline/
  atlas_template.html          editable markup for the page (data blocks are __DATA__ placeholders)
  atlas_h100/                  extraction + SAE + per-model / aggregate analyses (GPU)
    adapters/                  one adapter per model (base.py = the contract)
    common/                    SAE, annotation, depth-matching, per-model runners
    run_model.py               residual extraction + TopK-SAE + feature catalog for a model
    {modules_alllayers,cka_layers,cell_cka,nonlinearity,layer_explorer,svd_vs_sae,
     cka_svd_null,svd_projection_k,seed_variance,seed_variance_alllayer,gsea_prerank,
     flow_alllayers,depth_profile,tissue_from_emb,celltype_difficulty,
     build_explorer_slim}.py   the analyses
    configs/models.yaml · environment.yml · data/{build_genesets,prepare_corpus}.py
  scripts/                     assembly, controls & figures (CPU, run locally)
    reannotate_string · alllayer_concepts · alllayer_matrix · module_themes ·
    genes_search_ts3 · flow_ts3 · findings · atlas_assemble · inject_atlas   (build)
    controls[2-4] · random_null · degree_null · stats_final · recalibrate[_final,_robust] ·
    kegg_robust · depth_backbone · depth_calibrated · heldout_calibrated ·
    heldout_compare · topn_sweep · gsea_annot                                (controls → controls.json)
    make_figures · make_fig14 · make_figS1 · make_figS2                      (static figures)
    geometry/                single-model embedding-geometry probes (scPRINT / AIDO / ESM-prior
                             line — complexes, PPI, localization, clusters, orthologs, TF sign);
                             a related but separate analysis, not part of the 10-model comparison
```

> **Paths.** The CPU scripts assume the repo checked out under a working directory and use a `BASE`/`C`
> constant near the top of each file (currently an absolute path from the authors' machine) — set it to your
> checkout before running. `data/controls.json` (all nulls / CIs) is committed so the figures and number
> checks run without a GPU; the bulkier derived JSONs (per-model feature catalogs, `*_alllayers.json`,
> activations) are regenerated by the pipeline and are not committed here.

## Reproduce

Per-model extraction runs on a GPU (H100-class); assembly, controls and figures are CPU-only.

1. **Corpus** — `pipeline/atlas_h100/data/prepare_corpus.py` (Tabula Sapiens via cellxgene-census, seed 0;
   seed 1 for the held-out draw) and `data/build_genesets.py` (GO_BP / Reactome / KEGG / STRING / TRRUST).
2. **Per model (GPU)** — `run_model.py --model <M> --corpus <ts.h5ad> --out out_alllayers --all-layers`,
   then the per-model analyses (`cell_cka --all-layers`, `layer_explorer`, `svd_vs_sae`).
3. **Aggregates (GPU/CPU)** — `modules_alllayers --all`, `depth_profile --all`, `cka_layers --all`,
   `cell_cka --cka`, `nonlinearity --all --layers all`, `tissue_from_emb --all`, `build_explorer_slim`,
   `celltype_difficulty`, `flow_alllayers`; annotation-free nulls: `cka_svd_null`, `svd_projection_k`;
   stability: `seed_variance`, `seed_variance_alllayer`; `gsea_prerank`.
4. **Assembly (CPU)** — `reannotate_string` → `alllayer_concepts` → `alllayer_matrix` → `module_themes` →
   `genes_search_ts3` → `findings` → `atlas_assemble` → `inject_atlas` (rebuilds `index.html`).
5. **Controls (CPU)** — `controls*` , `random_null`, `degree_null`, `stats_final`, `recalibrate*`,
   `kegg_robust`, `depth_backbone`, `depth_calibrated`, `heldout_calibrated`, `heldout_compare`,
   `topn_sweep` → everything lands in `controls.json` (the null/CI source of truth).
6. **Figures (CPU)** — `make_figures` (artefact + backbone), `make_fig14` (overview + annotation-free),
   `make_figS1` (SVD projection across *k*), `make_figS2` (depth robustness) from `controls.json`.

Adding a model = write `pipeline/atlas_h100/adapters/<m>.py` (implement the extraction contract), register
it in `run_model.py` and the roster lists, then re-run steps 2–6.

## Controls & robustness

Every headline number is tested against a null and is reproducible from `data/controls.json`:

| Check | Result |
|---|---|
| Permissive "universal core" vs random-gene null | real < null → **artefact** (retracted as a result) |
| Calibrated backbone (≥8/10) vs uniform null | 78 vs 3.5 ± 2.0, *p* < 0.004, 22× |
| … vs **degree-matched** null | 78 vs 1.3 ± 1.3, 59× |
| Held-out replication | Jaccard 0.50 vs 0.05 two-draw baseline, *p* < 0.007 |
| Annotator-config robustness | fold 2.3–166× across five calibrated configs |
| Depth robustness | ≥8/10 significant at every depth (17–28× over null) |
| KEGG-drop | 56 concepts at 19× (GO+Reactome only) |
| SAE seeds | mid-layer core CV ≈ 3 % across 3 seeds |
| CKA reality | off-diag 0.12–0.92 vs 0.002–0.009 cell-shuffle floor |
| SAE vs top-*k* PCA (matched sparsity) | 0.61–0.87 vs 0.23–0.44; random dict < 0 |

## Host on GitHub Pages

`index.html` is fully self-contained, so hosting is one setting:

1. Push this repository (already done for `Biodyn-AI/atlas-comparison`).
2. GitHub → **Settings → Pages → Source: Deploy from a branch → `main` / root** (needs repo-admin).
3. Live at `https://<org>.github.io/<repo>/`.

`index.html` is the full build (~61 MB, under GitHub's 100 MB file limit, so it commits and serves directly).
It is heavy to load; `inject_atlas.py` can emit a capped (~17 MB) build from the same data.

## Credit & citation

Built on the method and single-model atlases of **Ihor Kendiukhov / Biodyn-AI**
([bio-sae](https://github.com/Biodyn-AI/bio-sae)). This repository is the calibrated cross-model extension.
A manuscript is in preparation; please cite the bio-sae work and this repository until it appears.
