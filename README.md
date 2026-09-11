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
| **Prediction test** | do co-firing gene pairs recover held-out regulatory links the atlas never used? | tested |
| Prediction test · equal budget | is the per-model ranking about feature quality, or dictionary size? | tested |
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
- **It is depth-robust and replicates.** 41–102 concepts at 17.6–30.4× over the null across the full depth
  sweep (250 permutations, every depth at the empirical floor p < 0.004), richest early-to-middle. Membership does drift with depth (Jaccard 0.34–0.48 versus the mid-layer
  set), so the ~78 are one depth's slice of a depth-invariant phenomenon, not a fixed list. On independent
  held-out cells, backbone Jaccard 0.50 versus a 0.05 two-draw baseline (max 0.33, p < 0.007) — run on **nine of
  ten models**, since scGPT could not be re-extracted on the held-out corpus, so the replicated object is the ≥7-of-9
  backbone (81 concepts), not the headline ≥8-of-10 set of 78. About **63% is
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
- **The features predict, not just describe — in some models.** Gene pairs that repeatedly co-fire in the same SAE
  features recover held-out **TRRUST** transcription-factor→target edges. The co-firing statistic uses no annotation
  at all, and the calibrated annotator uses curated pathway sets only, so TRRUST is held out from both. (It does appear
  in the *permissive* annotator, which is retracted as illustrative and plays no part here.)
  Against a configuration-model null that preserves every gene's co-firing degree exactly (so abundance and study bias
  cannot produce it), **5 of 10 models pass** at p ≤ 0.005 over 200 rewirings: Tahoe-x1 13.3× (59 edges recovered),
  scGPT 11.9× (10), UCE 10.5× (43), C2S-Scale 4.1× (83), Geneformer-V2 3.9× (12). MaxToki reaches 3.5× but p = 0.11;
  tGPT 1.3×; AIDO.Cell, scFoundation and GeneCompass recover nothing. **Cross-model agreement sharpens it**: over all ten models, pairs
  predicted by one are enriched 4.8×, by two independently **20.8×** — the comparative design paying off directly,
  though in absolute terms that is a precision of 0.21%, about one real edge per 480 pairs proposed. (Pooling only
  the five models that passed gives 5.3× and 28.4×, but those were selected on TRRUST and re-scored against it, so
  the all-ten figure is the one we report.) Stable across every robustness variant (4.8–8.9× at one model, p ≤ 0.005 in all five),
  including dropping same-family paralogues, raising the evidence bar, and keeping only features that actually fire —
  the last *raises* enrichment (5.3× → 7.9×), ruling out a near-silent-gene artefact. It also holds within a single
  layer (6.1×, p = 0.005). 4,349 corroborated pairs absent from TRRUST, STRING and every curated pathway are released
  as **prioritised predictions, not findings** (`data/hypothesis_final.json`). Caveat: TRRUST records regulation
  someone has already published, so this scores recovery of *known* links; recovered counts are 10–83 per model, so we
  rank models rather than read small gaps.
- **The predictions hold against literature and against perturbation.** Two further, independent bodies of
  evidence on the same 4,349 pairs. *Literature* (NCBI gene2pubmed, a dated bulk file, not live queries):
  1,289 of 4,111 mappable pairs are co-mentioned versus **593 ± 17** under a null preserving each gene's
  publication count — **2.2×, z = 40.5, p = 0.005**. Papers annotating >60 genes are excluded; without that
  cap 96% of pairs are nominally co-mentioned and the measure is vacuous. *Perturbation* (Replogle et al.
  2022 genome-wide CRISPRi — no annotation, no literature): knocking down one gene moves its predicted
  partner into the top 5% of responders for **10.6% vs 8.8%** of pairs in K562 (**1.20×**, z = 3.8,
  p = 0.005) and **23.8% vs 18.7%** in the independent RPE1 line (**1.28×**, z = 5.4, p = 0.005), against a
  responsiveness-matched null with ranks taken within each perturbation. Magnitudes are small because the
  screens are cancer/epithelial lines while the corpus is immune/kidney/lung — this is transfer *across*
  biological context. Whether model agreement also raises the causal rate is **unresolved**: it does in RPE1
  (1.28× → 1.53× at ≥3 models, p = 0.005) but not in K562 (1.42× on 76 pairs, p = 0.13).
- **Why no new biology was found — and why that says little about the models.** The validation
  infrastructure has the same blind spot as the annotation infrastructure. The models place **4,003 genes
  with <5 publications** into some feature's top genes (793 protein-coding; the other 3,097 non-coding or
  pseudogenes). Our own confidence filter is **4.8× harsher** on those genes than on well-studied ones
  (3.5% of understudied genes survive into the tested pairs vs 16.8% of genes with ≥50 papers), because
  requiring recurrence across features selects for transcript abundance. And relaxing the filter does not
  help, because the evidence runs out: **752 of the 793** protein-coding understudied genes are absent from
  the genome-scale Perturb-seq screen entirely, whose perturbed set is **0.4% understudied against 9.9% of
  all protein-coding genes**. A prediction about an unstudied gene currently cannot be confirmed or refuted
  by any of these resources. Two results say the limit is the evidence, not the models: predictive power
  does **not** decline with how studied a gene is (1.02–1.21× across publication strata in K562,
  0.99–1.23× in RPE1, no trend, no agreement between lines on which stratum is strongest), and coverage of
  the unstudied genome can be bought but only with precision — dropping the recurrence threshold from 5
  features to 2 raises understudied genes in play from 141 to 703 while enrichment falls 1.20×→1.17×
  (K562) and 1.28×→1.07× (RPE1). See `pipeline/scripts/hypothesis_studybias.py`.
- **No evidence of *new* biology, from a third direction.** Only 171 of 4,111 pairs have never been
  co-mentioned, and just **4** pair two well-studied genes (≥20 papers each) that never appear together — all
  four long non-coding RNAs. Literature agrees with the databases and with TRRUST: cross-model consensus
  concentrates on documented biology. The atlas prioritises testable links; it does not detect the unexplored.

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
    --- held-out prediction test (the hypothesis generator that does work) ---
    hypothesis_trrust.py     co-firing graph per model vs held-out TRRUST, analytic degree-matched null
    hypothesis_trrust2.py    permutation validation + does cross-model agreement raise precision?
    hypothesis_trrust3.py    final: validated models only, the released prediction list
    hypothesis_robust.py     paralogues / firing-features-only / stricter-evidence variants
    depth_backbone_fast.py   depth robustness at 250 permutations (sparse annotator; verifies it
                             reproduces the 20-permutation observed counts before writing)
    hypothesis_pubmed.py     literature co-mention vs a publication-count-preserving null
    hypothesis_perturb.py    causal test on Replogle 2022 Perturb-seq (K562 + RPE1 replication)
    hypothesis_sizematched.py  re-scores every model on an equal prediction budget, separating
                             feature quality from dictionary size (the full-size ranking does not
                             survive: UCE loses significance, MaxToki gains it, two models are
                             untestable at any matched budget)
    hypothesis_studybias.py  does predictive power depend on how studied a gene is? (it does not)
    make_fig5.py             Fig 5 (per-model enrichment, cross-model precision, independent evidence)
    --- reproducibility guard ---
    audit_manuscript.py      re-checks every number in Results 3.4-3.5 / Discussion / Limitations
    audit_manuscript_core.py the same for Results 3.1-3.3 and Methods
                             (103 checks, each pulling the value from the JSON that produced it;
                              numeric comparison at the text's own rounding precision, so a
                              mismatch means a real disagreement, not a formatting difference)
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
