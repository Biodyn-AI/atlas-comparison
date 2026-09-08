# Single-model embedding-geometry probes

A **related but separate** analysis line from the ten-model comparative atlas: instead of annotating SAE
features, these scripts ask whether a *known* biological relationship is encoded directly in a single model's
embedding geometry — do the relevant genes/proteins sit closer together than random? They come from the
single-model audit line (scPRINT / AIDO, ESM-prior story) and are kept here for reproducibility; they are not
part of the 10-model comparison and are not wired into `atlas_assemble` / the atlas page.

| Script | Probe |
|---|---|
| `complex_geometry.py` | subunits of the same physical protein complex (EBI Complex Portal) vs random |
| `ppi_geometry.py` | protein–protein interaction confidence (STRING) vs geometric distance |
| `localization_geometry.py` | shared subcellular localization (HPA) — the ESM-prior story |
| `cluster_geometry.py` | co-regulated genomic clusters (HLA / histones / HOX / protocadherins / keratins / IFN-α) |
| `sex_chrom_geometry.py` | sex-linked genes (XIST / Y) on the layer-0 input embedding (scPRINT vs AIDO) |
| `ortholog_alignment.py` | human gene vs its mouse ortholog once the species offset is removed (scPRINT) |
| `regulation_sign.py` | is the *sign* of TF regulation (activator vs repressor) encoded? (TRRUST) |
| `tissue_mlp.py` | tissue-agnostic MLP fact catalog for any Tabula-Sapiens-style h5ad |

Each script hard-codes a `BASE` path near the top and reads single-model embeddings/catalogs; adjust to your
checkout before running. See each file's docstring for inputs and outputs.
