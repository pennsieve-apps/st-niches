# st-niches

How are the cell types (or spot types, or domains) of a tissue section arranged in space? Which
ones sit next to each other more often than chance, which avoid each other, and which recurring
**neighbourhoods** (niches) make up the tissue? This app answers those questions for one label
column of a spatial transcriptomics dataset: cell types from `st-celltype` (`cell_type`), the
dominant cell type per Visium spot from `cell2location` (`dominant_type`), or spatial domains
from `st-domains` (`domain`).

For a biologist: you get a label × label heat map of neighbourhood enrichment (red: these two
labels touch more than if labels were scattered at random over the same positions; blue: they
avoid each other), curves of how the co-occurrence of two labels changes with distance, and a
map of each section coloured by niche, where a niche is a kind of local mixture of labels (for
example "tumour core", "tumour–stroma boundary", "lymphoid follicle") found jointly over all
samples, so niche N03 means the same thing in every section.

For a computational analyst: per sample, squidpy's `nhood_enrichment` (permutation z-scores on
the spatial graph), `co_occurrence` and `interaction_matrix`, plus a pooled enrichment over
samples (one block-diagonal graph, labels shuffled within each sample); niches are KMeans
clusters of each observation's neighbourhood label composition (itself plus the graph neighbours
within `NICHE_HOPS` hops, or its `NICHE_K` nearest), as in Schürch et al. (2020) and in spirit
CellCharter (Varrone et al. 2024), over all samples at once.

It is a step of two workflows:

- **B. Cell-type Deconvolution & Niches** (Visium + scRNA-seq reference):
  `st-prep → st-qc → cell2location → st-niches (LABEL_COLUMN=dominant_type) → st-report`
- **C. Single-cell Spatial Profiling** (Xenium, MERSCOPE, CosMx):
  `st-prep → st-qc → st-celltype → {st-niches, st-ccc} → st-report`

It can run more than once in a workflow (on cell types and on domains) and several runs can link
into one report input, so **everything is written under `OUTPUT_DIR/<LABEL>/`**.

## Method

1. **Inputs.** For each sample listed in `samples.csv`, only `obs_names`, `obsm["spatial"]`,
   `uns["pennsieve"]` and the Visium lowres image are read from the `.h5ad` (the count matrix is
   never loaded). Labels are joined from `labels.csv` by `obs_id`; observations without a label
   (or with an empty one) are dropped from this analysis and counted. `Unassigned` / `Mixed` are
   kept as labels of their own.
2. **Spatial graph** (conventions, "Spatial neighbours"): Visium (`grid: hex`) with
   `coord_type="grid"`, 6 neighbours, `N_RINGS` rings; everything else `coord_type="generic"`
   with the Delaunay triangulation (`NEIGHBOURS=delaunay`, default for cells) or the `k` nearest
   (`NEIGHBOURS=6` for bins), dropping edges longer than `MAX_NEIGHBOUR_DISTANCE` ×
   `spot_diameter`. The graph is made binary and symmetric.
3. **Neighbourhood enrichment** (`squidpy.gr.nhood_enrichment`, `N_PERMS` permutations, seeded
   per sample from `SEED` and the sample id): for each label pair, the number of graph edges
   joining them, as a z-score against random relabellings of the same graph. **Pooled**: the
   per-sample graphs as one block-diagonal graph, labels shuffled only within their own sample,
   so a label that is common in one sample and rare in another cannot create enrichment.
4. **Co-occurrence** (`squidpy.gr.co_occurrence`): p(label b | within distance d of label a) /
   p(b), for `N_INTERVALS` equal distance steps from 0 to 50 × `spot_diameter` (at most half the
   sample's diagonal), on a seeded subsample of at most `MAX_OBS_FOR_COOCCURRENCE` observations
   (the cost grows with the square of the count; the subsample is shrunk further if squidpy's
   work array would exceed 1 GB, which only happens with very many labels).
5. **Interaction matrix** (`squidpy.gr.interaction_matrix`, row-normalized): of the graph
   neighbours of label a, the fraction carrying label b.
6. **Niches.** Every observation's neighbourhood composition (fraction of each label among
   itself and its neighbours within `NICHE_HOPS` hops, or its `NICHE_K` nearest by distance) is
   one row; KMeans (scikit-learn, k-means++, 10 starts, `SEED`) clusters the rows of all samples
   together. `N_NICHES=0` picks k in 3–10 by the best silhouette score on a seeded subsample of
   10,000 rows. Niches are named `N01`, `N02`, … from largest to smallest.

## Ports

| Port | Kind | Files | Notes |
|---|---|---|---|
| `spatial` | input (subfolder, required) | `<sample_id>.h5ad` | `st-qc` output (the AnnData contract). Uses `obsm["spatial"]` (or `obs` `x`/`y`), `uns["pennsieve"]` `grid` and `spot_diameter`, `uns["spatial"]` lowres image for the backdrop. |
| `labels` | input (subfolder, required) | `labels.csv` | `sample_id, obs_id, <label columns>`; `LABEL_COLUMN` must be one of them. If several `labels.csv` are present, the first with that column is used. |
| `metadata` | input (subfolder, required) | `samples.csv` | `st-prep`'s sample table; lists and orders the samples (an `.h5ad` not listed is skipped). Without it every `.h5ad` is a sample. |
| `niches` | output | `<LABEL>/…` (below) | The only output, so it is the whole of `OUTPUT_DIR`. |

Without links (files dropped together) the same files are found by name anywhere under
`INPUT_DIR` (symlinks followed).

## Parameters

| Parameter | Type | Default | What it does |
|---|---|---|---|
| `LABEL` | string | (empty) | Name of this run and its output folder, `[a-z0-9-]+`; anything else exits 2. Empty: `LABEL_COLUMN` with `_` as `-` (`cell_type` → `cell-type`, `dominant_type` → `dominant-type`). |
| `LABEL_COLUMN` | string | `cell_type` | Column of `labels.csv` to analyse (`cell_type`, `dominant_type`, `domain`, `cluster`, …). |
| `N_RINGS` | integer | `1` | Visium hex grid: rings of neighbours (1 = the 6 adjacent spots). |
| `NEIGHBOURS` | string | `delaunay` | Non-grid data: `delaunay`, or an integer k for the k nearest (use `6` for Visium HD / Slide-seq bins). |
| `MAX_NEIGHBOUR_DISTANCE` | number | `3.0` | Non-grid data: drop edges longer than this × `spot_diameter`; `0` = no cap. |
| `N_PERMS` | integer | `1000` | Permutations for the enrichment z-scores (per sample and pooled). |
| `N_INTERVALS` | integer | `20` | Distance intervals for co-occurrence. |
| `MAX_OBS_FOR_COOCCURRENCE` | integer | `20000` | Co-occurrence subsample per sample; `0` = all. |
| `NICHE_HOPS` | integer | `2` | Neighbourhood for niches: the observation plus graph neighbours within this many hops. |
| `NICHE_K` | integer | `0` | When > 0, the neighbourhood is the observation plus its K nearest observations (Schürch et al. use 10), ignoring `NICHE_HOPS`. |
| `N_NICHES` | integer | `0` | Number of niches; `0` = choose 3–10 by silhouette. `1` is rejected. |
| `MAX_PLOTS` | integer | `6` | Co-occurrence plots for at most this many labels (those whose short-range ratio with another label departs most from 1). |
| `SEED` | integer | `42` | Seed for permutations, subsamples and KMeans. |
| `THREADS` | integer | `0` | Threads for numba / BLAS; `0` = from the container's CPU quota. |

## Outputs (`OUTPUT_DIR/<LABEL>/`)

| File | Contents |
|---|---|
| `labels.csv` | `sample_id, obs_id, niche`: every labelled observation's niche (`N01`, …), the same niche names across samples. |
| `nhood_enrichment.csv` | `sample_id` (or `pooled`), `label_a, label_b, zscore, count` (count = graph edges between the two labels). Empty z-score where a label is absent from the sample. |
| `co_occurrence.csv` | `sample_id, label_a, label_b, distance, ratio`: distance is the upper bound of the interval, in coordinate units (µm for cells, full-resolution pixels for Visium). |
| `interactions.csv` | `sample_id, label_a, label_b, fraction` (rows of a label sum to 1). |
| `niche_composition.csv` | `niche, label, fraction` (of the niche's observations carrying the label), `enrichment` (fraction / the label's overall fraction), `neighbourhood_fraction` (mean neighbourhood composition, what KMeans clustered), `n_obs`. |
| `niche_summary.csv` | `sample_id, niche, n_obs, fraction`. |
| `summary.json` | `label`, `label_column`, `n_samples`, `n_obs`, `labels`, `label_fractions`, `per_sample` (graph, mean degree, co-occurrence subsample), `strongest_enriched` / `strongest_depleted` / `strongest_enriched_heterotypic` pooled pairs, `n_pairs_enriched` / `n_pairs_depleted` (abs z > 2), `n_niches`, `niche_selection` (silhouette per k), `niches` (size and top labels), `parameters`, `figures`. |
| `nhood_enrichment.png`, `<sample_id>_nhood_enrichment.png` | Pooled and per-sample z-score heat maps (one shared colour scale). |
| `niche_composition.png` | Niche × label composition heat map. |
| `<sample_id>_niches.png` | Spatial map coloured by niche (y inverted; on the lowres H&E for Visium). |
| `co_occurrence_<label>.png` | Co-occurrence ratio of the label with the others vs distance (mean over samples). |
| `provenance.json` | App and tool versions, parameters, method, inputs seen, per-sample status (observations, unlabelled, graph, timings), timings, peak RSS. |

A sample that fails (no labels for it, a single label, unreadable file) is logged and skipped;
the run exits 1 when every sample failed or there is no input, and 2 on bad parameters.

## Results on the demo data

**Xenium breast cancer, 20 k-cell subsample per replicate** (st-prep `MAX_OBS_PER_SAMPLE=20000`
→ st-qc → st-celltype `cell_type`, 7 types; defaults otherwise, output in the workflow repo
under `data/work/xenium-20k/niches/cell-type/`): 38,858 cells, Delaunay graph capped at 40 µm
(3 × the 13.4 µm typical cell diameter). Because the subsample keeps one cell in eight, the
graph is sparse (mean degree 3.0 and 3.6, 1,412 and 1,247 isolated cells); on the full slides
the mean degree is 5.8. Pooled z-scores: every type is most enriched with itself (Epithelial
142, T cell 118, Basal/myoepithelial 103); the strongest mixed pairs are Macrophage–T cell (20)
and Plasma cell–T cell (20), and Basal/myoepithelial–Epithelial (5.5, the duct wall); the
epithelium avoids everything else (with T cell −84, Fibroblast −73, Macrophage −39). The
silhouette chose 4 niches (0.44; 3–10 all 0.39–0.44): **N01** tumour epithelium (82 %
epithelial; 43 % of Rep1, 34 % of Rep2), **N02** mixed stroma (macrophages, fibroblasts,
endothelium; 27 % / 33 %), **N03** T-cell-rich immune niche (70 % T cells; 23 % / 22 %), **N04**
fibroblast-dense stroma (89 % fibroblasts; 7 % / 10 %). 47 s, peak 1.4 GB.
**Xenium breast cancer, full slides** (Rep1 160,674 + Rep2 116,801 cells after st-qc; st-celltype
`cell_type`, 8 types; defaults; `data/work/xenium/niches/cell-type/`): Delaunay graphs with mean
degree 5.8 and 5.7 (247 and 442 isolated cells). Pooled z-scores: homotypic contacts dominate
(Epithelial 630, Basal/myoepithelial 439, Endothelial 315, T cell 298, Fibroblast 215); the
strongest mixed pairs are B cell–T cell (141, lymphoid aggregates), Macrophage–T cell (78),
Plasma cell–T cell (59) and Fibroblast–Macrophage (52); the epithelium is segregated from T
cells (−361), fibroblasts (−308), macrophages (−200) and endothelium (−189). The silhouette chose
3 niches (0.42; k = 4 and 5 0.41): **N01** tumour epithelium (84 % epithelial; 45 % of Rep1,
35 % of Rep2), **N02** stroma (37 % fibroblasts, 15 % endothelial, 15 % macrophages; 30 % /
39 %), **N03** immune niche (54 % T cells, 15 % macrophages; 25 % / 26 %), which on the maps
rings the ducts and forms aggregates. 58 s, peak 1.36 GB (2 CPUs).
LYMPH_PLACEHOLDER

## Runtime

`computeTypes: [standard]`, 2 vCPU / 6 GiB, 1 h timeout. squidpy brings spatialdata, dask,
geopandas and numba, so the image is 2.1 GB, too heavy for a sensible Lambda. Samples are read
one at a time and only their coordinates and labels are kept (about 50 bytes per observation
plus the graph); the pooled enrichment and the niches hold every sample's graph and composition
at once, which is small next to squidpy's co-occurrence work array (subsample × intervals ×
labels², capped at 1 GB).

Measured in the linux/amd64 image under emulation on an M-series Mac (Docker, `--memory 3g
--cpus 2`, defaults: `N_PERMS=1000`, co-occurrence on 20,000 observations per sample):

| Input | Observations | Labels | Peak memory (cgroup) | Peak RSS | Wall time |
|---|---|---|---|---|---|
| Synthetic, 2 × 150 k cells (Delaunay) + 1 Visium-like | 301,437 | 6 | 1.64 GB | 1.22 GB | 56 s |
| Same, labels split into 24 | 301,437 | 24 | 2.54 GB | 1.99 GB | 79 s |
| Xenium breast Rep1 + Rep2 (full, st-celltype `cell_type`) | 277,475 | 8 | 1.36 GB | 1.04 GB | 58 s |
| Same cells, 18 stand-in labels (KMeans on PCA) | 277,475 | 18 | 2.11 GB | 1.59 GB | 70 s |
| Xenium 20 k-cell subsample per replicate (`cell_type`) | 38,858 | 7 | 1.40 GB | 1.06 GB | 47 s |
| Human Lymph Node (Visium, st-qc output, stand-in labels) | 3,928 | 8 | 1.23 GB | 1.01 GB | 31 s |

About 1 GB is the imported stack; the first enrichment and co-occurrence of a run spend ~15 s
compiling numba kernels. The 24-label run is the realistic worst case for a Xenium slide of the
demo's size (~285 k cells, 15–25 cell types); 6 GiB leaves room for a ~500 k-cell slide.
REAL_RUNTIME_PLACEHOLDER

## Running it locally

```sh
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
make test PYTHON=.venv/bin/python      # pytest on the synthetic fixture, no Docker
make run PYTHON=.venv/bin/python       # Docker over data/fixture/linked -> data/output/cell-type/
```

By hand:

```sh
docker build --platform linux/amd64 -t st-niches .
docker run --rm --platform linux/amd64 --memory 3g --network none \
  -e INPUT_DIR=/data/input -e OUTPUT_DIR=/data/output -e LABEL_COLUMN=cell_type \
  -v "$PWD/in:/data/input:ro" -v "$PWD/out:/data/output" st-niches
```

with `in/spatial/` (st-qc), `in/labels/` (st-celltype, cell2location or st-domains) and
`in/metadata/` (st-prep). See `sample-input/README.md`.

## Limitations

- Everything here is about the **labels**, not the expression: the result can only be as good as
  the cell typing or deconvolution upstream. Systematic mislabelling at boundaries (segmentation
  spill-over in Xenium, for example) shows up as enrichment between the neighbouring types.
- Visium spots hold several cells; `dominant_type` keeps only the most abundant one, so niches
  on Visium describe neighbourhoods of spot types, not of cells, and minority types are invisible.
- Enrichment z-scores grow with the number of observations: on a 150 k-cell section almost every
  pair is "significant". Compare z-scores between pairs and samples of similar size, and read
  the interaction fractions for effect size. The z-scores are not corrected for multiple tests.
- The pooled z-score treats samples as one tissue (shuffling within samples); it is not a test
  across replicates or conditions. Differences between conditions belong in `st-compare` /
  a statistical model over per-sample values.
- Niches are KMeans on label compositions with a silhouette-chosen k: a convenient partition,
  not a unique one. Silhouette often prefers few, coarse niches; set `N_NICHES` to look finer.
  Label compositions ignore expression within a type (CellCharter clusters expression
  embeddings, which can split niches that share a composition).
- Co-occurrence is on a subsample and its distance axis is in the coordinate unit of each sample
  (µm or pixels); samples of different platforms are not on one axis.
- On a subsample of cells (st-prep `MAX_OBS_PER_SAMPLE`) the cells are far apart, so the
  `MAX_NEIGHBOUR_DISTANCE` cap leaves many of them with few or no neighbours (isolated cells
  belong to a niche by their own label only). Use the full data for niches, or raise the cap.
- 2 µm Visium HD bins are out of scope (as for the whole workflow).

## Licences

Apache-2.0 for this app (see `LICENSE`). Bundled components and their licences: `THIRD_PARTY.md`.
The image contains no GPL code (squidpy and scanpy are installed without igraph / leidenalg).

## Citations

Neighbourhood enrichment, co-occurrence, interaction matrix and the spatial graph (squidpy):

> Palla G, Spitzer H, Klein M, Fischer D, Schaar AC, Kuemmerle LB, Rybakov S, Ibarra IL,
> Holmberg O, Virshup I, Lotfollahi M, Richter S, Theis FJ. **Squidpy: a scalable framework for
> spatial omics analysis.** Nat Methods 19(2):171-178 (2022).
> https://doi.org/10.1038/s41592-021-01358-2, PMID 35102346.

Cellular neighbourhoods as clusters of neighbourhood cell-type composition (the niche method):

> Schürch CM, Bhate SS, Barlow GL, Phillips DJ, Noti L, Zlobec I, Chu P, Black S, Demeter J,
> McIlwain DR, Kinoshita S, Samusik N, Goltsev Y, Nolan GP. **Coordinated Cellular Neighborhoods
> Orchestrate Antitumoral Immunity at the Colorectal Cancer Invasive Front.** Cell
> 182(5):1341-1359.e19 (2020). https://doi.org/10.1016/j.cell.2020.07.005, PMID 32763154.

Spatial niches from neighbourhood aggregation and clustering, over samples (CellCharter; the idea,
not its code or its expression-based features):

> Varrone M, Tavernari D, Santamaria-Martínez A, Walsh LA, Ciriello G. **CellCharter reveals
> spatial cell niches associated with tissue remodeling and cell plasticity.** Nat Genet
> 56(1):74-84 (2024). https://doi.org/10.1038/s41588-023-01588-4, PMID 38066188.

scanpy and anndata (data structures and the squidpy stack):

> Wolf FA, Angerer P, Theis FJ. **SCANPY: large-scale single-cell gene expression data
> analysis.** Genome Biol 19(1):15 (2018). https://doi.org/10.1186/s13059-017-1382-0,
> PMID 29409532.

> Virshup I, Rybakov S, Theis FJ, Angerer P, Wolf FA. **anndata: Access and store annotated data
> matrices.** JOSS 9(101):4371 (2024). https://doi.org/10.21105/joss.04371 (not in PubMed).

KMeans and the silhouette (via scikit-learn, https://scikit-learn.org):

> Lloyd S. **Least squares quantization in PCM.** IEEE Trans Inf Theory 28(2):129-137 (1982).
> https://doi.org/10.1109/TIT.1982.1056489 (not in PubMed).

> Rousseeuw PJ. **Silhouettes: A graphical aid to the interpretation and validation of cluster
> analysis.** J Comput Appl Math 20:53-65 (1987). https://doi.org/10.1016/0377-0427(87)90125-7
> (not in PubMed).

Demo data (Xenium): 10x Genomics, *FFPE Human Breast Cancer with Custom Add-on Panel*,
replicates 1 and 2 (CC BY 4.0,
https://www.10xgenomics.com/datasets/ffpe-human-breast-with-custom-add-on-panel-1-standard), from

> Janesick A, Shelansky R, Gottscho AD, Wagner F, Williams SR, Rouault M, Beliakoff G, Morrison CA,
> Oliveira MF, Sicherman JT, Kohlway A, Abousoud J, Drennon TY, Mohabbat SH, 10x Development
> Teams, Taylor SEB. **High resolution mapping of the tumor microenvironment using integrated
> single-cell, spatial and in situ analysis.** Nat Commun 14(1):8353 (2023).
> https://doi.org/10.1038/s41467-023-43458-x, PMID 38114474.

Demo data (Visium): 10x Genomics, *Human Lymph Node* (Space Ranger 1.1.0, CC BY 4.0), no DOI.

Every DOI above was checked on Crossref (title, full author list, journal, year, volume, pages)
and every PMID on PubMed (E-utilities `esearch` by DOI, then `esummary`: title, authors, year).
`CITATION.cff` lists the same works with their full author lists.
