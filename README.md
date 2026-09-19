# Comparative SAE Atlas

Ten single-cell foundation models, run on the same 6,000 Tabula Sapiens cells, opened with one
sparse autoencoder per layer (172 in all), annotated with one calibrated method, and compared:
what each model encodes, what they share, which to use for a given task, and whether their
features predict biology the models were never shown.

Live atlas: https://avantipova.github.io/comparative-sae-atlas/ (also served from this
repository's `index.html`; open it locally with any static server).

## What is here

```
index.html, data/            the site: a 12 MB page plus the feature maps it fetches on demand
outputs/atlas/comparative/   the 59 result files and 7 figures every number is read from
pipeline/atlas_h100/         the GPU stage: adapters, SAE, extraction, per-model analyses
pipeline/scripts/            the local stage: matrices, controls, prediction tests, assembly, audit
pipeline/atlas_template.html the site's markup; index.html is built from it
pipeline/legacy/             earlier builders, kept for provenance, used by nothing
docs/DECISIONS.md            what was decided, why, what it cost, where it lives
docs/METHODS.md              methods in brief, with pointers into the code
docs/REPRODUCE.md            how to check the numbers, rebuild the site, re-run the analysis, re-extract
docs/RESULTS.md              every result file: written by, read by, what it holds
requirements.txt             the local analysis environment (Python 3.13, NumPy 2.3.4)
```

Not here: the manuscript and supplement (shared separately; the audit needs them dropped into
`outputs/atlas/comparative/`), the per-run feature catalogues and external inputs (1.2 GB,
published as release assets and fetched by `pipeline/scripts/fetch_runs.sh`), and the SAE
weight files, which were not copied back from the GPU host (`pipeline/atlas_h100/README.md`).

## Check it

```bash
pip install -r requirements.txt
export ATLAS_BASE=$(pwd)
cp /path/to/MANUSCRIPT.md /path/to/SUPPLEMENTARY.md outputs/atlas/comparative/
python pipeline/scripts/audit_all.py      # 312 checks: paper, supplement, figures, site
python pipeline/scripts/inject_atlas.py   # rebuilds index.html from the result files
```

## The models

| model | params | layers | reads a cell as | objective | prior | species |
|---|---|---|---|---|---|---|
| AIDO.Cell | 10M | 8 | expression, all genes | MLM | none | human |
| scGPT | ~50M | 12 | binned expression | MLM | none | human |
| tGPT | ~120M | 8 | rank order | autoregressive | none | human |
| scFoundation | 100M | 12 | expression, read-depth aware | MAE | none | human |
| GeneCompass | 104M | 12 | rank order plus value | MLM | knowledge graph | human, mouse |
| MaxToki | 217M | 11 | rank order | autoregressive (Llama) | none | human |
| Geneformer-V2 | 316M | 18 | rank order | MLM | none | human |
| UCE | 650M | 33 | ESM2 protein tokens, sampled by expression | masked | ESM | multi-species |
| C2S-Scale | 2B | 26 | rank-ordered gene names as text | autoregressive | text (Gemma-2) | human, mouse |
| Tahoe-x1 | 3B | 32 | binned expression | MLM | none | human |

Checkpoints and sizes are in `pipeline/atlas_h100/configs/models.yaml` and Table S1.

## What we found

Every number below is audited against the result file that produced it. The controls that
failed are reported alongside the ones that passed.

- **The naive "universal core" is a database artefact.** The permissive annotator (top-5 genes
  against five large databases) reports 2,267 concepts shared by all ten models. Random genes
  share 3,563 under the same construction (z = −18.4); at the mid layer 353 real against 589
  random (z = −6.1). Retracted.
- **A calibrated backbone is real and modest.** Requiring three of a feature's top-10 genes in
  one curated pathway of at most 200 genes, 78 concepts are shared by at least 8 of the 10
  models: 22.5× the uniform random-gene null and 59.3× a degree-matched one (p < 0.004, 250
  permutations); 41 to 102 concepts at every depth; reproduced on an independent draw of cells
  (Jaccard 0.50 against 0.048, p < 0.007, nine models). About 63 % is specific programme biology
  (antigen presentation, cytokine signalling, defence, muscle), 37 % housekeeping. Only two
  concepts are in all ten models.
- **Geometry agrees without the annotator.** Linear CKA between every pair of models is above
  its cell-shuffle floor (worst pair 43×). The SAE explains more variance than the best rank-32
  PCA subspace in all ten models, by +0.02 (tGPT) to +0.53 (Tahoe-x1). Tissue identity is
  linearly decodable at every depth.
- **Features predict held-out regulation, in half the models.** Gene pairs that co-fire in
  the same features recover TRRUST edges, which no step of the pipeline used, above a null that
  preserves every gene's degree: Tahoe-x1 13.3×, scGPT 11.9×, UCE 10.5×, C2S-Scale 4.1×,
  Geneformer-V2 3.9× (p ≤ 0.005); the other five do not. On an equal budget of 8,631 pairs UCE
  loses significance and MaxToki gains it. Pairs predicted by two models independently are 20.8×
  over the null against 4.8× for one, a precision of 0.21 %.
- **The same pairs hold against literature and perturbation.** Co-mention in NCBI gene2pubmed
  2.2× a publication-count-preserving null; in genome-wide CRISPRi Perturb-seq, knocking down
  one gene moves the other into the top 5 % of responders 1.20× more often than a
  responsiveness-matched null in K562 and 1.28× in RPE1.
- **No new biology, and the reason is the evidence.** Only 171 of 4,111 pairs were never
  co-mentioned, and four of those join two well-studied genes. The models feature 4,003 genes
  with fewer than five papers; the recurrence filter is 4.8× harsher on them, and 752 of the 793
  protein-coding ones are absent from the Perturb-seq screen. Predictive power does not decline
  with how studied a gene is. The atlas ranks testable links; it cannot confirm a link about an
  unstudied gene with these resources.

Also retracted: annotation rate as an interpretability score, "about 100 % of features invisible
to SVD" (true of random directions at this dimensionality), and a "new biology" shortlist that
two nulls showed to be a deficit. Architecture-to-feature associations rest on n = 10 models,
have bootstrap CIs of about ±1, and are reported as descriptive only.

## Credit

Built on the method and single-model atlases of Biodyn-AI
([bio-sae](https://github.com/Biodyn-AI/bio-sae)). This repository is the cross-model extension:
one corpus, one calibrated annotator, a null behind every claim.
