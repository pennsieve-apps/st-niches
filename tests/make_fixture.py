"""
A synthetic fixture that follows the workflow contract (conventions.md):
spatial/<sample_id>.h5ad as st-qc writes them, labels/labels.csv as st-celltype
writes it, metadata/samples.csv.

  python -m tests.make_fixture data/fixture [--layout both|linked] [--cells 3000]

Three samples with the same label vocabulary, planted in spatial patches:

  xa, xb  Xenium-like (cells, microns, no grid): two tumour discs with a boundary band of
          fibroblasts and T cells, one B-cell follicle away from the tumour, stroma elsewhere
          (fibroblasts, macrophages, a few endothelial cells). So Tumor-Tumor and B cell-B cell
          are enriched, Tumor-B cell is depleted, and there are four kinds of neighbourhood:
          tumour core, tumour boundary, follicle, stroma.
  vis     Visium-like (hex grid, full-resolution pixels, a lowres image): the same patches on a
          36 x 40 hex lattice.

30 genes with a little label-specific expression; the app never reads them.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

LABELS = ["B cell", "Endothelial", "Fibroblast", "Macrophage", "T cell", "Tumor"]
GENES = [f"G{i:02d}" for i in range(30)]


def _draw(rng, n, probs: dict[str, float]) -> np.ndarray:
    names = list(probs)
    p = np.array([probs[k] for k in names], dtype=float)
    return np.array(names)[rng.choice(len(names), n, p=p / p.sum())]


def plant_labels(rng: np.random.Generator, xy: np.ndarray, scale: float) -> np.ndarray:
    """Labels by region; `scale` = field width (microns or pixels), regions are fractions of it."""
    x, y = xy[:, 0] / scale, xy[:, 1] / scale
    labels = np.empty(len(xy), dtype=object)
    stroma = {"Fibroblast": 0.55, "Macrophage": 0.25, "T cell": 0.15, "Endothelial": 0.05}
    labels[:] = _draw(rng, len(xy), stroma)
    tumours = [((0.25, 0.33), 0.15), ((0.72, 0.68), 0.13)]
    for (cx, cy), r in tumours:
        d = np.hypot(x - cx, y - cy)
        band = (d >= r) & (d < r + 0.06)
        labels[band] = _draw(rng, int(band.sum()), {"Fibroblast": 0.5, "T cell": 0.3, "Tumor": 0.2})
        core = d < r
        labels[core] = _draw(rng, int(core.sum()), {"Tumor": 0.9, "T cell": 0.1})
    d = np.hypot(x - 0.78, y - 0.2)
    follicle = d < 0.08
    labels[follicle] = _draw(rng, int(follicle.sum()), {"B cell": 0.85, "T cell": 0.15})
    return labels.astype(str)


def _expression(rng: np.random.Generator, labels: np.ndarray):
    import scipy.sparse as sp

    n = len(labels)
    base = rng.gamma(0.5, 1.0, size=(n, len(GENES)))
    for i, lab in enumerate(LABELS):
        base[labels == lab, 5 * i:5 * i + 5] += 3.0   # five "markers" per label
    counts = rng.poisson(base).astype(np.float32)
    counts = sp.csr_matrix(counts)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    totals[totals == 0] = 1
    norm = sp.diags(1e4 / totals) @ counts
    norm.data = np.log1p(norm.data).astype(np.float32)
    return sp.csr_matrix(norm, dtype=np.float32), sp.csr_matrix(counts, dtype=np.float32)


def _write_h5ad(path: Path, sample_id: str, obs_ids: list[str], xy: np.ndarray, labels: np.ndarray,
                pennsieve: dict, rng: np.random.Generator, extra_obs: dict | None = None, spatial_uns: dict | None = None):
    import anndata as ad
    import pandas as pd

    x, counts = _expression(rng, labels)
    obs = pd.DataFrame(index=pd.Index([f"{sample_id}:{o}" for o in obs_ids]))
    obs["sample_id"] = pd.Categorical([sample_id] * len(obs_ids))
    obs["x"], obs["y"] = xy[:, 0], xy[:, 1]
    obs["in_tissue"] = 1
    obs["total_counts"] = np.asarray(counts.sum(axis=1)).ravel()
    obs["n_genes_by_counts"] = np.asarray((counts > 0).sum(axis=1)).ravel()
    obs["pct_counts_mt"] = 0.0
    for k, v in (extra_obs or {}).items():
        obs[k] = v
    var = pd.DataFrame(index=pd.Index(GENES))
    var["gene_ids"] = [f"ENSG{i:011d}" for i in range(len(GENES))]
    var["feature_types"] = "Gene Expression"
    var["mt"] = False
    a = ad.AnnData(X=x, obs=obs, var=var, layers={"counts": counts}, obsm={"spatial": xy.astype(np.float64)})
    a.uns["pennsieve"] = dict(pennsieve, normalized=True, target_sum=10000, prep_version="0.1.0", source=sample_id)
    if spatial_uns:
        a.uns["spatial"] = spatial_uns
    path.parent.mkdir(parents=True, exist_ok=True)
    a.write_h5ad(path)
    return obs.index.to_numpy()


def make_xenium_sample(root: Path, sample_id: str, n_cells: int, rng: np.random.Generator):
    width, height = 1200.0, 900.0   # microns
    xy = np.column_stack([rng.uniform(0, width, n_cells), rng.uniform(0, height, n_cells)])
    labels = plant_labels(rng, xy, width)
    obs_ids = [f"cell_{i + 1}" for i in range(n_cells)]
    pennsieve = {"platform": "xenium", "kind": "cell", "coordinate_unit": "micron", "spot_diameter": 12.0,
                 "grid": "none"}
    names = _write_h5ad(root / "spatial" / f"{sample_id}.h5ad", sample_id, obs_ids, xy, labels, pennsieve, rng,
                        extra_obs={"cell_area": rng.uniform(40, 120, n_cells), "control_counts": 0})
    return names, labels


def make_visium_sample(root: Path, sample_id: str, rng: np.random.Generator, n_rows: int = 36, n_cols: int = 40):
    pitch = 100.0   # full-resolution pixels between spot centres
    rows, cols = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    x = 200.0 + cols * pitch + (rows % 2) * pitch / 2
    y = 200.0 + rows * pitch * np.sqrt(3) / 2
    xy = np.column_stack([x, y])
    width = x.max() + 200.0
    labels = plant_labels(rng, xy, width)
    obs_ids = [f"BC{i:04d}-1" for i in range(len(xy))]
    scale = 0.05
    image = rng.integers(150, 230, size=(int((y.max() + 200) * scale) + 1, int(width * scale) + 1, 3), dtype=np.uint8)
    pennsieve = {"platform": "visium", "kind": "spot", "coordinate_unit": "pixel", "spot_diameter": 65.0, "grid": "hex"}
    spatial_uns = {sample_id: {"images": {"lowres": image, "hires": image},
                               "scalefactors": {"tissue_lowres_scalef": scale, "tissue_hires_scalef": scale,
                                                "spot_diameter_fullres": 65.0, "fiducial_diameter_fullres": 105.0}}}
    names = _write_h5ad(root / "spatial" / f"{sample_id}.h5ad", sample_id, obs_ids, xy, labels, pennsieve, rng,
                        extra_obs={"array_row": rows, "array_col": cols * 2 + rows % 2}, spatial_uns=spatial_uns)
    return names, labels


def make_synthetic(root: Path, cells: int = 3000, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    rows, sample_rows = [], []
    specs = [("xa", "xenium"), ("xb", "xenium"), ("vis", "visium")]
    for sid, platform in specs:
        if platform == "xenium":
            names, labels = make_xenium_sample(root, sid, cells, rng)
            path, kind = f"{sid}/", "cell"
        else:
            names, labels = make_visium_sample(root, sid, rng)
            path, kind = f"{sid}/outs/", "spot"
        clusters = {lab: f"C{i + 1:02d}" for i, lab in enumerate(LABELS)}
        for name, lab in zip(names, labels):
            rows.append({"sample_id": sid, "obs_id": name, "cluster": clusters[lab], "cell_type": lab})
        sample_rows.append({"sample_id": sid, "path": path, "platform": platform, "kind": kind, "n_obs": len(names),
                            "n_genes": len(GENES), "n_obs_in_tissue": len(names), "condition": "ctrl" if sid != "xb" else "treated"})
    # A handful of unlabelled observations (as st-celltype's Unassigned-free rows would be absent), to be dropped.
    rows = rows[:-3]
    (root / "labels").mkdir(parents=True, exist_ok=True)
    with open(root / "labels" / "labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample_id", "obs_id", "cluster", "cell_type"])
        w.writeheader()
        w.writerows(rows)
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    with open(root / "metadata" / "samples.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sample_rows[0]))
        w.writeheader()
        w.writerows(sample_rows)
    return root


def flatten(linked: Path, flat: Path) -> Path:
    """The same files with no port folders, nested a little, as an unlinked run would see them."""
    for p in linked.rglob("*"):
        if p.is_file():
            dest = flat / ("nested" if p.suffix == ".h5ad" else "") / p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, dest)
    return flat


def make_fixture(root: Path, layout: str = "both", cells: int = 3000) -> dict[str, Path]:
    root = Path(root)
    out = {"linked": make_synthetic(root / "linked", cells)}
    if layout == "both":
        out["flat"] = flatten(out["linked"], root / "flat")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--layout", default="both", choices=["both", "linked"])
    ap.add_argument("--cells", type=int, default=3000)
    a = ap.parse_args()
    print(make_fixture(Path(a.root), a.layout, a.cells))
