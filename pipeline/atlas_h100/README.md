# Comparative SAE Feature Atlas — H100 pipeline

Unify many single-cell foundation models into **one** comparative atlas. Not more separate
atlases — a shared frame where features from every model are comparable, to extract
cross-model knowledge (universality, blind spots, the regulatory-logic gap, architecture →
feature-repertoire, feature-orthologs, depth grammar).

Join-compatible with Igor's atlases (`Biodyn-AI/bio-sae`, arXiv 2603.02952): same TopK SAE
config, same annotation vocabulary. Every model → `feature_catalog.json` (top-20 genes /
feature) → **one unified annotator** (Fisher + BH vs GO_BP / KEGG / Reactome / TRRUST) →
concept × model matrix.

## Layout
```
common/    sae.py (TopK, bio-sae config)  pipeline.py (extract→SAE→catalog)  annotate.py (unified)
adapters/  base.py (contract)  aido.py, scprint.py (VERIFIED)  _stubs.py (UCE/GeneCompass/scBERT/tGPT/CellPLM — recipes)
data/      build_genesets.py (GO/KEGG/Reactome via gseapy + TRRUST)  prepare_corpus.py (shared cells)
run_model.py  configs/models.yaml
```

## One-time setup
```bash
conda env create -f environment.yml && conda activate atlas
# shared annotation vocabulary (build once, used for ALL models):
python data/build_genesets.py --trrust /path/to/trrust_human.tsv --out data/genesets
# shared cell corpus (~2000 cells, broad genes) as data/corpus.h5ad — see data/prepare_corpus.py
```

## Run order (validate before scaling)
1. **End-to-end smoke on a trusted model** (AIDO — verified forward):
   ```bash
   python run_model.py --model AIDO --smoke      # 32 cells, all stages, ~minutes
   ```
   Check `out/AIDO/feature_catalog_L*.json` (alive count, VarExpl > 0.75, dead < 2%,
   decoder-cos < 0.05) and `out/annotations/AIDO_L*_annotations.json` (annotation rate ~30–60%).
2. **Full run, verified models:** `python run_model.py --model AIDO` then `--model scPRINT`.
3. **Add a scaffolded model:** implement its `iter_activations` in `adapters/_stubs.py`
   (each carries the exact load / forward-hook / tokenise / symbol recipe), run `--smoke`,
   verify the extraction-integrity check, then the full run. Priority: **UCE** (ESM-prior
   replication) and **GeneCompass** (knowledge prior → the headline TF-logic test).

## The headline test — regulatory logic (valid, non-circular)
`run_perturbation.py` runs Igor's Phase-8 test on any model: run Replogle K562 CRISPRi
knockdown + control cells through the model, find SAE features that DIFFERENTIALLY respond
to each TF knockdown (Wilcoxon KD-vs-control, BH<0.05), then SEPARATELY Fisher-test whether
responders detect that TF's TRRUST targets. No feature is selected on the targets → no
circularity (an in-model ablation test IS circular — see the memo). Reports detection rate
+ TF-specific rate vs Igor's 92% / 6.2%.
```bash
# get the data once (1.55 GB): Zenodo record 7041849, ReplogleWeissman2022_K562_essential.h5ad
python run_perturbation.py --model scPRINT --perturb data/replogle_k562_essential.h5ad \
    --trrust data/trrust_human.tsv --n-tfs 48 --n-kd 40 --device cuda
```
The cross-model question — is the null universal, or does UCE (ESM) / GeneCompass (knowledge)
/ scPRINT (GRN) break it? — is answered by running this on each model. First local scPRINT
pass (6 TFs, underpowered): detect 67%, TF-specific 17% (1/6) — low, consistent with Igor.

## Adapter contract (adapters/base.py)
Implement `iter_activations(adata, batch_size)` → yields, per batch:
`({layer: float32 [n_positions, d_model]}, hgnc_symbols[n_positions])`, taking only **gene-token**
positions (drop CLS/pad/special), symbols UPPER-cased HGNC. Everything else is model-agnostic.

## Outputs → comparative analysis (run locally, CPU)
`out/<model>/feature_catalog_L*.json` + `out/annotations/<model>_L*_annotations.json` feed the
concept × model matrix and the 8 analyses (see `project-biomech-comparative-atlas` memo).
Ingest Igor's models from `public/data/layer_XX_features.json` in his atlas repos, re-annotate
them with the SAME `data/build_genesets.py` vocabulary, and they drop into the same matrix.

## Fairness notes
- **Same corpus, same SAE config, same annotation vocabulary** across all newly-extracted models.
- Igor's ingested models were trained/extracted on his corpora (Geneformer=K562, scGPT=Tabula
  Sapiens) — re-annotated uniformly, but flag the data difference in cross-model claims.
- Different `d_model`/#layers handled by comparing in **annotation space** and by **relative depth**.
