"""
The spatial neighbour graph of one sample, built with squidpy as the workflow
conventions say ("Spatial neighbours"):

  grid: hex (Visium)   coord_type="grid", n_neighs=6, n_rings=N_RINGS
  everything else      coord_type="generic", Delaunay triangulation or the K
                       nearest neighbours (NEIGHBOURS), capped at a radius of
                       MAX_NEIGHBOUR_DISTANCE x spot_diameter so sparse regions
                       don't connect across gaps.

squidpy 1.8 splits `spatial_neighbors` (deprecated, removed in 1.9) into one
builder per mode; the builders are used when present, with the same arguments.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

log = logging.getLogger("st-niches")


def _minimal_adata(coords: np.ndarray):
    import anndata as ad

    obs = pd.DataFrame(index=pd.Index([str(i) for i in range(coords.shape[0])]))
    return ad.AnnData(obs=obs, obsm={"spatial": np.asarray(coords, dtype=np.float64)})


def _binary_symmetric(adj: sp.spmatrix) -> sp.csr_matrix:
    a = sp.csr_matrix(adj, dtype=np.float32)
    a = a.maximum(a.T).tocsr()
    a.setdiag(0)
    a.eliminate_zeros()
    a.sum_duplicates()
    a.data[:] = 1.0
    return a


def build_graph(coords: np.ndarray, pennsieve: dict[str, Any], n_rings: int, neighbours: str | int,
                max_neighbour_distance: float) -> tuple[sp.csr_matrix, dict[str, Any]]:
    """Binary, symmetric CSR connectivities (no self loops) and a description of how they were built."""
    import scanpy as sc
    import squidpy as sq

    sc.settings.verbosity = 0   # squidpy logs through scanpy; our own log lines are enough
    logging.getLogger("spatialdata._logging").setLevel(logging.WARNING)   # squidpy logs graph building through it
    n = coords.shape[0]
    grid = str(pennsieve.get("grid") or "none").lower()
    spot = pennsieve.get("spot_diameter")
    try:
        spot = float(spot) if spot is not None else None
    except (TypeError, ValueError):
        spot = None
    cap = max_neighbour_distance * spot if (spot and spot > 0 and max_neighbour_distance > 0) else None
    adata = _minimal_adata(coords)
    info: dict[str, Any] = {"n_obs": n, "grid": grid, "spot_diameter": spot, "radius_cap": cap}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=DeprecationWarning)
        if grid == "hex":
            info.update(mode="grid", n_neighs=6, n_rings=n_rings)
            if hasattr(sq.gr, "spatial_neighbors_grid"):
                sq.gr.spatial_neighbors_grid(adata, n_neighs=6, n_rings=n_rings)
            else:
                sq.gr.spatial_neighbors(adata, coord_type="grid", n_neighs=6, n_rings=n_rings)
            adj = adata.obsp["spatial_connectivities"]
        elif neighbours == "delaunay":
            info.update(mode="delaunay")
            radius = (0.0, cap) if cap else None
            if hasattr(sq.gr, "spatial_neighbors_delaunay"):
                sq.gr.spatial_neighbors_delaunay(adata, radius=radius)
            else:
                sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True, radius=radius)
            adj = adata.obsp["spatial_connectivities"]
        else:
            k = int(neighbours)
            info.update(mode="knn", n_neighs=k)
            k_eff = max(1, min(k, n - 1))
            if hasattr(sq.gr, "spatial_neighbors_knn"):
                sq.gr.spatial_neighbors_knn(adata, n_neighs=k_eff)
            else:
                sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=k_eff)
            adj = sp.csr_matrix(adata.obsp["spatial_connectivities"])
            if cap:
                dist = sp.csr_matrix(adata.obsp["spatial_distances"])
                if dist.nnz:
                    dist.data = (dist.data <= cap).astype(np.float32)   # 1 where the edge is within the cap
                    adj = sp.csr_matrix(adj.multiply(dist))
    graph = _binary_symmetric(adj)
    degree = np.diff(graph.indptr)
    info.update(n_edges=int(graph.nnz // 2), mean_degree=round(float(degree.mean()), 3) if n else 0.0,
                n_isolated=int((degree == 0).sum()))
    return graph, info
