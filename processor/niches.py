"""
The method: squidpy's neighbourhood enrichment, co-occurrence and interaction
matrix on a label column, and cellular niches as KMeans clusters of each
observation's neighbourhood composition (Schürch et al. 2020; Varrone et al.
2024), implemented with scikit-learn.

Labels are handled as integer codes into one shared category list, so every
matrix has the same label x label layout in every sample and in the pooled
result; a label absent from a sample gives NaN there.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

log = logging.getLogger("st-niches")

# squidpy's co-occurrence kernel holds an (n, intervals x labels^2) int32 work array; the
# subsample is shrunk below MAX_OBS_FOR_COOCCURRENCE when that would exceed this budget.
COOCCURRENCE_BUDGET_BYTES = 1_000_000_000
# Co-occurrence distance thresholds run from 0 to COOCCURRENCE_RANGE x spot_diameter (capped at half the
# sample's extent; the half extent alone when spot_diameter is unknown) in N_INTERVALS equal steps.
COOCCURRENCE_RANGE = 50.0
SILHOUETTE_MAX_OBS = 10_000
SILHOUETTE_SAMPLE = 5_000
K_RANGE = range(3, 11)


def sample_seed(seed: int, sample_id: str) -> int:
    """A per-sample seed so a sample's draws don't depend on the rest of the run."""
    h = 1469598103934665603
    for b in sample_id.encode():
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return (seed * 1_000_003 + (h % 2_147_483_647)) % (2**31 - 1)


def _label_adata(codes: np.ndarray, categories: list[str], graph: sp.csr_matrix | None = None,
                 coords: np.ndarray | None = None, libraries: np.ndarray | None = None):
    import anndata as ad

    obs = pd.DataFrame(index=pd.Index([str(i) for i in range(len(codes))]))
    obs["label"] = pd.Categorical.from_codes(np.asarray(codes, dtype=np.int64), categories=categories)
    if libraries is not None:
        obs["library"] = pd.Categorical(libraries)
    a = ad.AnnData(obs=obs)
    if graph is not None:
        a.obsp["spatial_connectivities"] = graph
    if coords is not None:
        a.obsm["spatial"] = np.asarray(coords, dtype=np.float64)
    return a


def nhood_enrichment(graph: sp.csr_matrix, codes: np.ndarray, categories: list[str], n_perms: int, seed: int,
                     libraries: np.ndarray | None = None, n_jobs: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Label x label permutation z-scores and observed neighbour-pair counts (squidpy.gr.nhood_enrichment).

    With `libraries` (one id per observation) the labels are shuffled within each library, which
    is how the pooled result over samples is computed: one block-diagonal graph, permutations that
    never move a label from one sample to another.
    """
    import scanpy as sc
    import squidpy as sq

    sc.settings.verbosity = 0   # squidpy logs through scanpy; our own log lines are enough
    a = _label_adata(codes, categories, graph=graph, libraries=libraries)
    result = sq.gr.nhood_enrichment(a, cluster_key="label", library_key="library" if libraries is not None else None,
                                    n_perms=n_perms, rng=seed, copy=True, show_progress_bar=False, n_jobs=n_jobs)
    zscore = np.asarray(result.zscore, dtype=np.float64)
    count = np.asarray(result.counts)
    return zscore, count


def interaction_matrix(graph: sp.csr_matrix, codes: np.ndarray, categories: list[str]) -> np.ndarray:
    """Row-normalized label x label interaction matrix: of the neighbours of label a, the fraction with label b."""
    import squidpy as sq

    a = _label_adata(codes, categories, graph=graph)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = sq.gr.interaction_matrix(a, cluster_key="label", normalized=True, copy=True)
    return np.asarray(m, dtype=np.float64)


def co_occurrence_subsample(n: int, max_obs: int, n_intervals: int, n_labels: int, rng: np.random.Generator) -> np.ndarray:
    """Seeded row indices for the co-occurrence computation, within MAX_OBS and the memory budget."""
    per_obs = 4 * n_intervals * n_labels * n_labels
    budget = max(500, COOCCURRENCE_BUDGET_BYTES // max(per_obs, 1))
    keep = min(n, max_obs if max_obs > 0 else n, budget)
    if keep < n:
        return np.sort(rng.choice(n, keep, replace=False))
    return np.arange(n)


def co_occurrence(coords: np.ndarray, codes: np.ndarray, categories: list[str], n_intervals: int,
                  spot_diameter: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Co-occurrence ratio (squidpy.gr.co_occurrence) over n_intervals distance thresholds.

    Returns (ratio[label_a, label_b, interval] in the shared label order, NaN for labels absent from the
    subsample; the upper distance of each interval). Thresholds run from 0 to COOCCURRENCE_RANGE x
    spot_diameter (at most half the extent of the observations) in equal steps, so the first interval
    holds the closest pairs and the distances are the same for every sample of a platform.
    """
    import squidpy as sq

    present = np.unique(codes)
    local = {c: i for i, c in enumerate(present)}
    local_codes = np.array([local[c] for c in codes], dtype=np.int64)
    a = _label_adata(local_codes, [categories[c] for c in present], coords=coords)
    span = coords.max(axis=0) - coords.min(axis=0)
    d_max = float(np.sqrt((span ** 2).sum())) / 2.0
    if spot_diameter and spot_diameter > 0:
        d_max = min(d_max, COOCCURRENCE_RANGE * float(spot_diameter))
    if not d_max > 0:
        raise ValueError("all observations share one position")
    interval = np.linspace(0.0, d_max, n_intervals + 1)
    occ, thresholds = sq.gr.co_occurrence(a, cluster_key="label", interval=interval, copy=True)
    occ = np.asarray(occ, dtype=np.float64)
    full = np.full((len(categories), len(categories), occ.shape[2]), np.nan)
    full[np.ix_(present, present, np.arange(occ.shape[2]))] = occ
    return full, np.asarray(thresholds[1:], dtype=np.float64)


def neighbourhood_composition(graph: sp.csr_matrix, codes: np.ndarray, n_labels: int, hops: int,
                              coords: np.ndarray | None = None, k: int = 0) -> np.ndarray:
    """Per observation, the fraction of each label among the observation itself and its neighbours.

    Neighbours are the graph nodes within `hops` hops, or with k > 0 the k nearest observations by
    distance (kd-tree). Including the observation itself keeps isolated nodes from having an empty
    neighbourhood. Returns an (n, n_labels) float32 matrix whose rows sum to 1.
    """
    n = len(codes)
    onehot = sp.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), np.asarray(codes))), shape=(n, n_labels))
    if k > 0:
        from sklearn.neighbors import NearestNeighbors

        k_eff = max(1, min(k, n - 1))
        nn = NearestNeighbors(n_neighbors=k_eff + 1).fit(coords)
        idx = nn.kneighbors(coords, return_distance=False)   # includes the point itself
        rows = np.repeat(np.arange(n), idx.shape[1])
        reach = sp.csr_matrix((np.ones(idx.size, dtype=np.float32), (rows, idx.ravel())), shape=(n, n))
        reach.sum_duplicates()
        reach.data[:] = 1.0
    else:
        eye = sp.identity(n, dtype=np.float32, format="csr")
        reach = (graph + eye).tocsr()
        frontier = graph.tocsr()
        for _ in range(1, max(1, hops)):
            frontier = (frontier @ graph).tocsr()
            frontier.data[:] = 1.0
            reach = (reach + frontier).tocsr()
            reach.data[:] = 1.0
        reach.sum_duplicates()
        reach.data[:] = 1.0
    counts = np.asarray((reach @ onehot).todense(), dtype=np.float32)
    totals = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return counts / totals


def choose_n_niches(comp: np.ndarray, seed: int, k_range=K_RANGE) -> tuple[int, dict[int, float]]:
    """The k in k_range with the best silhouette on a seeded subsample of the composition rows."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    rng = np.random.default_rng(seed)
    n = comp.shape[0]
    rows = np.sort(rng.choice(n, min(n, SILHOUETTE_MAX_OBS), replace=False))
    sub = comp[rows]
    n_distinct = len(np.unique(sub.round(6), axis=0))
    scores: dict[int, float] = {}
    for k in k_range:
        if k >= n_distinct or k >= len(sub):
            break
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(sub)
        if len(np.unique(km.labels_)) < 2:
            continue
        scores[k] = float(silhouette_score(sub, km.labels_, sample_size=min(SILHOUETTE_SAMPLE, len(sub)),
                                           random_state=seed))
    if not scores:
        return min(max(2, n_distinct), k_range.start if hasattr(k_range, "start") else 3), scores
    best = max(scores, key=lambda k: (scores[k], -k))
    return best, scores


def cluster_niches(comp: np.ndarray, n_niches: int, seed: int) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """KMeans over all observations' neighbourhood compositions. Niches are named N01, N02, ... by size."""
    from sklearn.cluster import KMeans

    info: dict[str, Any] = {"requested": n_niches}
    k = n_niches
    if k <= 0:
        k, scores = choose_n_niches(comp, seed)
        info["silhouette"] = {str(kk): round(s, 4) for kk, s in scores.items()}
        info["chosen_by"] = "silhouette"
    n_distinct = len(np.unique(comp.round(6), axis=0))
    k = max(1, min(k, n_distinct, comp.shape[0]))
    info["n_niches"] = k
    if k == 1:
        return np.zeros(comp.shape[0], dtype=np.int64), ["N01"], info
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(comp)
    raw = km.labels_
    order = np.argsort(-np.bincount(raw, minlength=k), kind="stable")
    remap = np.empty(k, dtype=np.int64)
    remap[order] = np.arange(k)
    labels = remap[raw]
    names = [f"N{i + 1:02d}" for i in range(k)]
    info["inertia"] = float(km.inertia_)
    return labels, names, info


def pair_table(matrix: np.ndarray, categories: list[str], value: str, extra: dict[str, Any] | None = None,
               second: np.ndarray | None = None, second_name: str = "count") -> list[dict[str, Any]]:
    """A label x label matrix as long rows: label_a, label_b, value (+ a second matrix's value)."""
    rows = []
    for i, a in enumerate(categories):
        for j, b in enumerate(categories):
            r = dict(extra or {})
            r.update(label_a=a, label_b=b)
            r[value] = float(matrix[i, j])
            if second is not None:
                v = second[i, j]
                r[second_name] = int(v) if np.isfinite(v) and float(v).is_integer() else float(v)
            rows.append(r)
    return rows
