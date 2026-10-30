# Third-party software

This app is Apache-2.0 (see `LICENSE`). The image it builds contains the following third-party
components, installed from PyPI at the versions pinned in `requirements.txt` (direct
dependencies first, then every transitive one as the image resolved it); their licences apply
to the image. Nothing in the image is GPL: squidpy and scanpy are installed without igraph or
leidenalg (this app does no clustering with them), so the image as a whole can be
redistributed under these permissive licences.

| Component | What it does here | Licence | Source |
|---|---|---|---|
| squidpy 1.8.4 | Spatial neighbour graph (`spatial_neighbors_*`), `nhood_enrichment`, `co_occurrence`, `interaction_matrix`. | BSD-3-Clause | https://github.com/scverse/squidpy |
| scanpy 1.12.5 | squidpy dependency (settings, logging). | BSD-3-Clause | https://github.com/scverse/scanpy |
| anndata 0.13.4 | Reads `obs`, `obsm["spatial"]`, `uns` of the `.h5ad` files (`anndata.io.read_elem`). | BSD-3-Clause | https://github.com/scverse/anndata |
| scikit-learn 1.9.1 | KMeans, silhouette score, nearest neighbours (`NICHE_K`). | BSD-3-Clause | https://scikit-learn.org |
| NumPy 2.5.4, SciPy 1.18.1, pandas 3.1.0 | Arrays, sparse graphs, tables. | BSD-3-Clause | https://numpy.org, https://scipy.org, https://pandas.pydata.org |
| numba 0.68.1 (+ llvmlite 0.50.0) | JIT kernels used by squidpy. | BSD-2-Clause (llvmlite: BSD-2-Clause, bundles LLVM under Apache-2.0 with LLVM exceptions) | https://numba.pydata.org |
| h5py 3.16.0 | HDF5 access (bundles HDF5, BSD-style licence). | BSD-3-Clause | https://www.h5py.org |
| Matplotlib 3.11.2 | PNG figures. | Matplotlib licence (PSF-based, BSD-compatible) | https://matplotlib.org |
| docrep 0.3.2 | squidpy dependency; installed from its source distribution (no wheel exists). | Apache-2.0 | https://github.com/Chilipp/docrep |
| spatialdata 0.8.0, spatialdata-plot, xarray, dask, distributed, zarr, ome-zarr, geopandas, shapely, pyogrio, pyproj, pyarrow, datashader, xarray-spatial, scikit-image, dask-image, tifffile, imagecodecs, statsmodels, seaborn, umap-learn, pynndescent, networkx, omnipath, centrosome, cp_measure, mahotas and the rest of `requirements.txt` | squidpy's dependencies (imported, mostly unused by this app). | BSD-3-Clause / BSD-2-Clause / MIT / Apache-2.0 / ISC / PSF / MPL-2.0 (certifi); colorcet's colour maps CC-BY-4.0. Checked from each package's metadata in the built image. | PyPI |
| python:3.12.11-slim-bookworm | Base image (Debian 12 + CPython 3.12). | PSF (Python); Debian packages under their own free licences | https://hub.docker.com/_/python |

## Not bundled

- **CellCharter** (https://github.com/CSOgroup/cellcharter, BSD-3-Clause) clusters neighbourhood
  embeddings of gene expression with Gaussian mixtures. This app's niches follow the same idea
  on label compositions with KMeans, implemented with scikit-learn; no CellCharter code is used.
- **Demo data** (not in the image or the repository): 10x Genomics *FFPE Human Breast Cancer*
  Xenium replicates 1 and 2 and *Human Lymph Node* Visium (CC BY 4.0), through st-prep, st-qc,
  st-celltype and cell2location.
