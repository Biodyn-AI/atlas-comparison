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
weights and activations, which stay on the GPU host (`pipeline/atlas_h100/README.md`).

The catalogues were re-extracted in October 2026 after three faults in the first release were
found: a position cap that let some models see 32 cells and others 6,000, filler genes in the
top-gene lists, and an unseeded SAE. Every number below is from the corrected run; what changed
and why is in `docs/DECISIONS.md` (decision 16).

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
  against five large databases) reports 1,977 concepts shared by all ten models. Random genes
  share 3,570 under the same construction (z = −24.2); at the mid layer 365 real against 582
  random (z = −4.9). It does not hold.
- **A calibrated backbone is real and modest.** Requiring three of a feature's top-10 genes in
  one curated pathway of at most 200 genes, 96 concepts are shared by at least 8 of the 10
  models: 27.5× the uniform random-gene null and 51.2× a degree-matched one (p < 0.004, 250
  permutations); 55 to 133 concepts at every depth; reproduced on an independent draw of cells
  that shares none with the corpus (Jaccard 0.52 against 0.035, p = 0.02, ten models). By the
  genes behind each match, 42 concepts are one protein-synthesis block, 12 other housekeeping
  and 42 specific programmes (antigen presentation, cytokine signalling, defence, muscle); the
  96 form 37 distinct programmes (`backbone_composition.json`). No concept is in all ten models.
- **Co-expression alone recovers most of it.** A dictionary built from gene-gene correlation in
  the same cells, with no model (a seed gene plus its nine most correlated genes), recovers 73
  to 95 of the 96 backbone concepts per dictionary with random seeds; the models' own middle
  layers recover 23 to 96. The backbone is mostly the co-expression structure of the data, not
  something the models add to it.
- **Geometry agrees without the annotator.** Linear CKA between every pair of models is above
  its cell-shuffle floor (worst pair 43×). The SAE explains more variance than the best rank-32
  PCA subspace in all ten models, by +0.02 (tGPT) to +0.55 (Geneformer-V2). Tissue identity is
  linearly decodable at every depth, with no gain from a non-linear probe. PCA of the raw
  expression decodes tissue as well (0.94 balanced accuracy against 0.85 to 0.96 for the models'
  middle layers).
- **Gene pairs recover known regulation in four of the ten models, but co-expression does about
  as well.** Gene pairs that co-fire in the same features recover TRRUST edges above a null that
  preserves every gene's degree: UCE 10.2×, Tahoe-x1 9.4×, C2S-Scale 5.5×, Geneformer-V2 2.5×
  (p ≤ 0.005); the other six do not. The loose annotation rule uses TRRUST, and the threshold of
  five shared features was chosen after scoring against it. Co-expression from the same cells
  passes the same test; matched for genes and partners per gene, only Tahoe-x1 stays ahead once
  same-family and look-alike pairs are set aside, also within cell types. SAEs that were never
  trained give weaker pair signal, and a Geneformer-V2 with random weights gives none. All four
  hold on an equal budget of 50,707 pairs.
  Pairs predicted by two models independently are 15.3× over the null against 5.0× for one
  (2,000 rewirings, `agreement_curve.py`), a precision of 0.23 %; agreement beyond two models adds
  nothing (none of the 918 pairs shared by five or more models is a known edge).
- **Literature and perturbation give small support.** Co-mention in NCBI gene2pubmed
  2.3× a publication-count-preserving null; in CRISPRi Perturb-seq (genome-wide K562 screen, RPE1 essential-gene screen), knocking down
  one gene moves the other into the top 5 % of responders 1.24× more often than a
  responsiveness-matched null in K562 and 1.33× in RPE1.
- **No new biology, and the reason is the evidence.** Only 98 of 4,272 pairs were never
  co-mentioned, and six of those join two well-studied genes. The models feature 5,953 genes
  with fewer than five papers; the recurrence filter is 7.0× harsher on them, and 732 of the 774
  protein-coding ones are absent from the Perturb-seq screen. Predictive power does not decline
  with how studied a gene is. The atlas ranks testable links; it cannot confirm a link about an
  unstudied gene with these resources.

Also not supported: annotation rate as an interpretability score, "about 100 % of features invisible
to SVD" (true of random directions at this dimensionality), and a "new biology" shortlist that
two nulls showed to be a deficit. Architecture-to-feature associations rest on n = 10 models,
have bootstrap CIs of about ±1, and are reported as descriptive only.

## Credit

Built on the method and single-model atlases of Biodyn-AI
([bio-sae](https://github.com/Biodyn-AI/bio-sae)). This repository is the cross-model extension:
one corpus, one calibrated annotator, a null behind every claim.
