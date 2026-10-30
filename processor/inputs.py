"""
Finding and reading the inputs.

Linked layout (placement: subfolder): INPUT_DIR/spatial/<sample_id>.h5ad,
INPUT_DIR/labels/labels.csv and INPUT_DIR/metadata/samples.csv. Without links
the same files are found by name anywhere under INPUT_DIR (symlinks followed).

Only what this app needs is read from each .h5ad (obs, obsm["spatial"],
uns["pennsieve"], the low-resolution image when there is one); the count matrix
is never loaded.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("st-niches")


@dataclass
class Sample:
    sample_id: str
    path: Path
    row: dict[str, Any] = field(default_factory=dict)


@dataclass
class Inputs:
    samples: list[Sample]
    labels_csv: Path | None
    samples_csv: Path | None
    files_seen: list[Path]
    skipped: dict[str, str]


@dataclass
class SampleData:
    sample_id: str
    obs_names: np.ndarray            # str, in file order
    coords: np.ndarray               # (n, 2) float64, obsm["spatial"]
    pennsieve: dict[str, Any]        # uns["pennsieve"] (may be empty)
    image: np.ndarray | None = None  # lowres H&E (Visium) as uint8 (h, w, 3)
    image_scale: float = 1.0         # tissue_lowres_scalef


def walk(root: Path) -> list[Path]:
    """Every file under root, following symlinks (INPUT_DIR entries are symlinks), sorted."""
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        found.extend(Path(dirpath) / f for f in sorted(filenames) if not f.startswith("."))
    return found


def _find(root: Path, port: str, predicate) -> list[Path]:
    """Files of a port: its own folder when it exists, else anywhere under INPUT_DIR."""
    folder = root / port
    base = folder if folder.is_dir() else root
    return [p for p in walk(base) if predicate(p)]


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def discover(input_dir: Path, label_column: str) -> Inputs:
    h5ads = _find(input_dir, "spatial", lambda p: p.suffix.lower() == ".h5ad")
    labels_files = _find(input_dir, "labels", lambda p: p.name.lower() == "labels.csv")
    samples_files = _find(input_dir, "metadata", lambda p: p.name.lower() == "samples.csv")
    skipped: dict[str, str] = {}

    labels_csv = None
    if len(labels_files) == 1:
        labels_csv = labels_files[0]
    elif labels_files:
        # Several labels.csv (several label apps linked into one port): the first that has the column.
        with_column = [p for p in labels_files if label_column in pd.read_csv(p, nrows=0).columns]
        if not with_column:
            raise ValueError(f"{len(labels_files)} labels.csv found, none with a {label_column!r} column")
        if len(with_column) > 1:
            log.warning("%d labels.csv files have a %r column; using %s", len(with_column), label_column,
                        with_column[0])
        labels_csv = with_column[0]
    samples_csv = samples_files[0] if samples_files else None
    if len(samples_files) > 1:
        log.warning("%d files named samples.csv; using %s", len(samples_files), samples_csv)

    by_stem: dict[str, Path] = {}
    for p in h5ads:
        if p.stem in by_stem:
            log.warning("two .h5ad files for sample %s; using %s", p.stem, by_stem[p.stem])
            continue
        by_stem[p.stem] = p

    samples: list[Sample] = []
    if samples_csv is not None:
        table = pd.read_csv(samples_csv, dtype=str, keep_default_na=False)
        if "sample_id" not in table.columns:
            raise ValueError(f"{samples_csv.name}: no sample_id column")
        listed = set()
        for _, r in table.iterrows():
            row = {k: _text(v) for k, v in r.items()}
            sid = row["sample_id"]
            listed.add(sid)
            if sid not in by_stem:
                skipped[sid] = "listed in samples.csv but no .h5ad found"
                continue
            samples.append(Sample(sid, by_stem[sid], row))
        for stem, path in by_stem.items():
            if stem not in listed:
                skipped[stem] = ".h5ad not listed in samples.csv"
    else:
        log.warning("no samples.csv: every .h5ad found is a sample")
        samples = [Sample(stem, path, {"sample_id": stem}) for stem, path in sorted(by_stem.items())]
    return Inputs(samples, labels_csv, samples_csv, h5ads, skipped)


def read_labels(path: Path, label_column: str) -> pd.DataFrame:
    """labels.csv as a frame indexed by obs_id with the label column as strings (empty = unlabelled)."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ("sample_id", "obs_id", label_column):
        if col not in table.columns:
            raise ValueError(f"{path.name}: no {col!r} column (columns: {', '.join(table.columns)})")
    table = table[["sample_id", "obs_id", label_column]].rename(columns={label_column: "label"})
    table["label"] = table["label"].str.strip()
    if table["obs_id"].duplicated().any():
        n = int(table["obs_id"].duplicated().sum())
        log.warning("%s: %d duplicated obs_id values; keeping the first of each", path.name, n)
        table = table[~table["obs_id"].duplicated()]
    return table.set_index("obs_id")


def read_sample(sample: Sample) -> SampleData:
    """obs_names, coordinates and metadata of one .h5ad, without its count matrix."""
    import h5py
    from anndata.io import read_elem

    with h5py.File(sample.path, "r") as h:
        obs = read_elem(h["obs"])
        obs_names = np.asarray(obs.index.astype(str))
        if "obsm" in h and "spatial" in h["obsm"]:
            coords = np.asarray(read_elem(h["obsm"]["spatial"]), dtype=np.float64)
        elif "x" in obs.columns and "y" in obs.columns:
            coords = np.column_stack([obs["x"].to_numpy(np.float64), obs["y"].to_numpy(np.float64)])
        else:
            raise ValueError("no obsm['spatial'] and no obs x/y columns")
        if coords.ndim != 2 or coords.shape[0] != len(obs_names) or coords.shape[1] < 2:
            raise ValueError(f"obsm['spatial'] has shape {coords.shape} for {len(obs_names)} observations")
        coords = np.ascontiguousarray(coords[:, :2])
        if not np.isfinite(coords).all():
            raise ValueError("non-finite coordinates in obsm['spatial']")
        pennsieve: dict[str, Any] = {}
        if "uns" in h and "pennsieve" in h["uns"]:
            raw = read_elem(h["uns"]["pennsieve"])
            pennsieve = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in dict(raw).items()}
        image, scale = None, 1.0
        if "uns" in h and "spatial" in h["uns"]:
            lib = h["uns"]["spatial"]
            key = sample.sample_id if sample.sample_id in lib else (next(iter(lib.keys()), None))
            if key is not None and "images" in lib[key] and "lowres" in lib[key]["images"]:
                try:
                    image = np.asarray(read_elem(lib[key]["images"]["lowres"]))
                    sf = read_elem(lib[key]["scalefactors"]) if "scalefactors" in lib[key] else {}
                    scale = float(sf.get("tissue_lowres_scalef", 1.0))
                except Exception as exc:   # the image is only a backdrop
                    log.warning("%s: cannot read the lowres image: %s", sample.sample_id, exc)
                    image = None
    return SampleData(sample.sample_id, obs_names, coords, pennsieve, image, scale)
