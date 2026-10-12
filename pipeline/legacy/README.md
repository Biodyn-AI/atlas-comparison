# Legacy

Code from earlier stages of the project, kept for provenance. No published number, figure or
site panel is produced by anything here, and nothing in `../scripts/` imports it.

`../scripts/legacy/`

| script | what it was |
|---|---|
| `atlas_ingest.py` | read the earlier single-model atlases' per-layer feature files into one schema |
| `atlas_catalog.py`, `aido_catalog.py` | first feature catalogues, before the shared corpus and the H100 pipeline |
| `atlas_compare.py` | first concept x model matrix (`matrix.json`) |
| `atlas_build_v2.py`, `atlas_analyze_v2.py`, `atlas_genesearch.py` | the nine-model "v2" build on mixed corpora (`matrix_v2.json`), replaced by the shared-corpus build |
| `atlas_modules.py` | module views from the first cluster run (`modules_data.json`), replaced by `atlas_h100/modules_alllayers.py` |

`atlas_h100/`

| file | what it was |
|---|---|
| `_stubs.py`, `scbert.py`, `scprint.py` | adapter scaffolds for models that were tried or planned and are not in the atlas |
| `run_perturbation.py`, `perturbation.py` | an in-model perturbation-response test on CRISPRi data, replaced by the pseudobulk test in `scripts/hypothesis_perturb.py` |
| `prepare_corpus.py` | the first corpus builder, replaced by `build_ts_corpus.py` |
| `gsea_annot.py` | a local rank-based check on the stored top-20 genes only; too few ranks to reach significance (it returns no annotations). Superseded by `../atlas_h100/gsea_prerank.py`, which ranks all ~19,000 genes and writes the published `gsea_annot.json`. Running this would overwrite that file with zeros |

The outputs of the first cluster run are in the `legacy-cluster-run.tar.gz` release asset.
