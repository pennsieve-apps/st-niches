# sample-input

This app reads the output of `st-qc` (`spatial/<sample_id>.h5ad`), a `labels.csv` from
`st-celltype` (`cell_type`), `cell2location` (`dominant_type`) or `st-domains` (`domain`), and
`st-prep`'s `metadata/samples.csv`.

## Synthetic fixture (no download)

```sh
make fixture PYTHON=.venv/bin/python   # data/fixture/{linked,flat}/
make run PYTHON=.venv/bin/python       # Docker over data/fixture/linked/ -> data/output/cell-type/
```

Three samples with the same six labels planted in spatial patches: two Xenium-like samples
(3,000 cells in microns, Delaunay graph) with two tumour discs ringed by fibroblasts and T cells,
a B-cell follicle away from the tumour and stroma elsewhere, and one Visium-like sample (a
36 × 40 hex lattice in full-resolution pixels with a lowres image, grid graph). Tumor–Tumor and
B cell–B cell come out enriched, Tumor–B cell depleted, and the niches separate the tumour
cores, the follicle and the stroma.

## Real data: 10x Xenium FFPE Human Breast Cancer

10x Genomics, *FFPE Human Breast Cancer* (Xenium Onboard Analysis 1.0.1), replicates 1 and 2,
CC BY 4.0 (Janesick et al. 2023, https://doi.org/10.1038/s41467-023-43458-x):
https://www.10xgenomics.com/datasets/ffpe-human-breast-with-custom-add-on-panel-1-standard

1. Run `st-prep` on the two replicate folders (`cell_feature_matrix.h5` + `cells.parquet`) with
   a `samples.csv` (`sample_id, path`), optionally `MAX_OBS_PER_SAMPLE=20000` for a quick run.
2. Run `st-qc` on its output, then `st-celltype` on the qc'd files with a marker table.
3. Run this app with `spatial/` = st-qc's output, `labels/` = st-celltype's output and
   `metadata/` = st-prep's metadata:

```sh
docker run --rm --platform linux/amd64 --memory 3g \
  -e INPUT_DIR=/data/input -e OUTPUT_DIR=/data/output -e LABEL_COLUMN=cell_type \
  -v "$PWD/data/xenium:/data/input:ro" -v "$PWD/data/output:/data/output" st-niches
```

where `data/xenium/spatial/`, `data/xenium/labels/` and `data/xenium/metadata/` hold (or link
to) those outputs. For Visium data from workflow B, link `cell2location`'s output as `labels/`
and set `LABEL_COLUMN=dominant_type`; for `st-domains`, `LABEL_COLUMN=domain`.
