# Comparative SAE Feature Atlas

A cross-model interpretability atlas for **single-cell foundation models**. We train sparse
autoencoders (SAEs) on the residual stream of many single-cell FMs on **one shared, tissue-controlled
human corpus**, annotate every feature against the same biological vocabulary, and compare *what* each
model organises and *how*. It extends the single-model atlases of Ihor Kendiukhov / Biodyn-AI
([bio-sae](https://github.com/Biodyn-AI/bio-sae)) into one comparative frame.

**▶ Live atlas:** open `index.html` (self-contained, full-resolution ~61 MB — heavy but complete) — or host it on GitHub Pages (below). It embeds all data; no server or build needed to view.

---

## What's inside

A single interactive page with, for each analysis, a chart + a plain-language reading:

| Section | Question |
|---|---|
| Universality | how many models share each biological concept (the universal core) |
| Coverage | annotation rate, concept count, source mix per model |
| Depth | how annotation/《concept richness》 changes layer by layer |
| Tissue | how tissue-specific features become with depth |
| SVD vs SAE | superposition — % of features invisible to a linear (SVD) basis |
| Linearity | is a concept linearly readable? (linear vs MLP probe gap), per layer |
| Modules | co-activation communities per layer (force-graph) |
| Cross-layer flow | feature persistence between **every adjacent layer** |
| Layer Explorer | UMAP / t-SNE map of features per layer, per model |
| Gene Search | which models encode a given gene, and under what concept |
| CKA | representational similarity across models (matched depth) + within-model layer×layer |
| Scale · Emergence · Convergence · Curriculum | concept-acquisition curve vs model size; cross-model feature convergence; depth curriculum; hardest cell types |
| Models | roster with params, **training species**, inductive axis |

## Model roster

Run on one shared corpus (6,000 Tabula Sapiens cells: immune + kidney + lung), depth-matched layers,
one annotator (top-5 genes → Fisher + BH<0.05 vs GO_BP / Reactome / KEGG / STRING), TopK-SAE (k=32, d=4×).

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

- **A universal biological backbone** of concepts is shared by *every* model on a matched corpus.
- **Superposition is universal**: ~100% of SAE features are invisible to the top SVD axes in every model —
  linear methods can't recover the dictionary (reproduces Igor's result across the whole roster).
- **Concepts are linearly represented** at every depth (MLP−linear probe gap ≈ 0), and more so deeper.
- **Emergence with scale**: the concept vocabulary grows smoothly from the smallest model to the largest;
  a third of all concepts are found only in the largest models, skewed to metabolism / membrane-transport,
  while the universal backbone is signaling / immune / cell-cycle.
- **Convergence on concepts, not features**: models agree on the same *biology* but almost never on the
  same *feature* (near-identical top-gene features recur <1% of the time across architectures).
- **Depth curriculum**: mitochondrial / RNA-processing / signaling features appear shallowest, metabolism
  / transport deepest — universal machinery early, specialised programs late.
- Architecture drives the differences (tokenization → how pathway-shaped features are; objective/prior →
  tissue-binding and representational drift); the invariants above hold across all architectures.

Live numbers are in the atlas (`Universality`, `Scale`, `SVD` sections).

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
    findings.py              emergence / rosetta / curriculum
    atlas_assemble.py        → data/atlas_full_notf.json (the source of truth)
    inject_atlas.py          embed the data blocks into index.html
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

## Host on GitHub Pages (Igor-style)

`index.html` is fully self-contained, so hosting is one setting:

1. Create a repo and push this folder.
2. GitHub → **Settings → Pages → Source: Deploy from a branch → `main` / root**.
3. Your atlas is live at `https://<user>.github.io/<repo>/`.

`index.html` is the full-resolution build (every feature, no per-layer cap, ~61 MB) — under GitHub's 100 MB
file limit, so it commits and serves directly. It's heavy to load in a browser; if you want a snappier page,
`inject_atlas.py` can emit a capped (~16–17 MB) build from the same data.

## Credit

Built on the method and single-model atlases of Ihor Kendiukhov / Biodyn-AI (bio-sae; Geneformer / scGPT /
Novae / MaxToki / C2S atlases). This repo is the cross-model extension.
