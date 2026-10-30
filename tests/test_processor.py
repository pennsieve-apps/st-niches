"""The synthetic fixture through the processor (both input layouts), the method's pieces, parameter checks."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from processor import main as app
from processor import niches as N
from processor.graph import build_graph
from tests.make_fixture import make_fixture

OUTPUTS = {"labels.csv", "nhood_enrichment.csv", "co_occurrence.csv", "interactions.csv", "niche_composition.csv",
           "niche_summary.csv", "summary.json", "provenance.json", "nhood_enrichment.png", "niche_composition.png",
           "xa_niches.png", "xb_niches.png", "vis_niches.png", "xa_nhood_enrichment.png"}
FAST = {"N_PERMS": "200"}


@pytest.fixture(scope="session")
def fixture_root(tmp_path_factory) -> dict[str, Path]:
    return make_fixture(tmp_path_factory.mktemp("fixture"), "both", cells=2500)


@pytest.fixture(scope="session")
def default_run(fixture_root, tmp_path_factory):
    """One default run (linked layout) shared by the tests that read its outputs."""
    out = tmp_path_factory.mktemp("out")
    env = {"INPUT_DIR": str(fixture_root["linked"]), "OUTPUT_DIR": str(out), "MPLCONFIGDIR": "/tmp/matplotlib", **FAST}
    import os

    saved = {k: os.environ.get(k) for k in list(app.PARAMETERS) + ["INPUT_DIR", "OUTPUT_DIR"]}
    for k in app.PARAMETERS:
        os.environ.pop(k, None)
    os.environ.update(env)
    try:
        code = app.main()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert code == 0
    return out / "cell-type"


def run(input_dir: Path, output_dir: Path, monkeypatch, **params) -> int:
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("MPLCONFIGDIR", "/tmp/matplotlib")
    for name in app.PARAMETERS:
        monkeypatch.delenv(name, raising=False)
    for name, value in {**FAST, **params}.items():
        monkeypatch.setenv(name, value)
    return app.main()


def rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def zscores(path: Path, sample: str) -> dict[tuple[str, str], float]:
    return {(r["label_a"], r["label_b"]): float(r["zscore"]) for r in rows(path)
            if r["sample_id"] == sample and r["zscore"] != ""}


def test_outputs_and_planted_pairs(default_run):
    out = default_run
    assert out.name == "cell-type" and [p.name for p in out.parent.iterdir()] == ["cell-type"]
    files = {p.name for p in out.iterdir()}
    assert OUTPUTS <= files
    assert any(f.startswith("co_occurrence_") and f.endswith(".png") for f in files)

    enrich = rows(out / "nhood_enrichment.csv")
    assert list(enrich[0]) == ["sample_id", "label_a", "label_b", "zscore", "count"]
    assert {r["sample_id"] for r in enrich} == {"xa", "xb", "vis", "pooled"}
    for sample in ("xa", "xb", "vis", "pooled"):
        z = zscores(out / "nhood_enrichment.csv", sample)
        assert z[("Tumor", "Tumor")] > 3, sample
        assert z[("B cell", "B cell")] > 3, sample
        assert z[("Tumor", "B cell")] < -2, sample
        assert z[("Tumor", "B cell")] == pytest.approx(z[("B cell", "Tumor")], abs=1e-6)
    # pooled counts are the sum of the per-sample counts
    counts = {}
    for r in enrich:
        counts.setdefault(r["sample_id"], {})[(r["label_a"], r["label_b"])] = int(r["count"])
    for pair, c in counts["pooled"].items():
        assert c == counts["xa"][pair] + counts["xb"][pair] + counts["vis"][pair]

    inter = rows(out / "interactions.csv")
    tumor = [float(r["fraction"]) for r in inter if r["sample_id"] == "xa" and r["label_a"] == "Tumor"]
    assert sum(tumor) == pytest.approx(1.0, abs=1e-4)

    cooc = rows(out / "co_occurrence.csv")
    assert list(cooc[0]) == ["sample_id", "label_a", "label_b", "distance", "ratio"]
    xa = [r for r in cooc if r["sample_id"] == "xa" and r["label_a"] == "Tumor" and r["label_b"] == "Tumor"]
    assert len(xa) == 20
    assert float(xa[0]["ratio"]) > 1.5          # tumour cells cluster at short range
    assert float(xa[0]["distance"]) < float(xa[-1]["distance"])


def test_niches_recover_planted_patches(default_run):
    out = default_run
    labels = rows(out / "labels.csv")
    assert list(labels[0]) == ["sample_id", "obs_id", "niche"]
    assert len(labels) == 2 * 2500 + 36 * 40 - 3           # three fixture rows are unlabelled
    niches = sorted({r["niche"] for r in labels})
    assert niches[0] == "N01" and 3 <= len(niches) <= 10
    comp = rows(out / "niche_composition.csv")
    assert list(comp[0])[:4] == ["niche", "label", "fraction", "enrichment"]
    frac = {(r["niche"], r["label"]): float(r["fraction"]) for r in comp}
    # a tumour-core niche and a follicle niche
    assert max(frac[(n, "Tumor")] for n in niches) > 0.7
    assert max(frac[(n, "B cell")] for n in niches) > 0.5
    for n in niches:
        assert sum(frac[(n, lab)] for lab in ("B cell", "Endothelial", "Fibroblast", "Macrophage", "T cell", "Tumor")) == pytest.approx(1.0, abs=1e-4)
    summary = rows(out / "niche_summary.csv")
    assert list(summary[0]) == ["sample_id", "niche", "n_obs", "fraction"]
    assert sum(float(r["fraction"]) for r in summary if r["sample_id"] == "xa") == pytest.approx(1.0, abs=1e-4)
    s = json.loads((out / "summary.json").read_text())
    assert s["label"] == "cell-type" and s["label_column"] == "cell_type" and s["n_samples"] == 3
    assert s["n_niches"] == len(niches) and s["niche_selection"]["chosen_by"] == "silhouette"
    assert s["strongest_enriched"][0]["zscore"] > 0 and s["strongest_depleted"][0]["zscore"] < 0
    assert {"Tumor", "B cell"} == {s["strongest_depleted"][0]["label_a"], s["strongest_depleted"][0]["label_b"]} or \
        any({p["label_a"], p["label_b"]} == {"Tumor", "B cell"} for p in s["strongest_depleted"])
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["app"] == "st-niches" and prov["version"] == "0.1.0"
    assert prov["n_ok"] == 3 and prov["n_failed"] == 0
    assert {"squidpy", "scikit-learn", "numpy"} <= set(prov["tools"])
    by_sample = {st["sample_id"]: st for st in prov["samples"]}
    assert by_sample["xa"]["graph"]["mode"] == "delaunay" and by_sample["xa"]["n_unlabelled"] == 0
    assert by_sample["vis"]["n_unlabelled"] == 3


def test_hex_grid_path(default_run):
    prov = json.loads((default_run / "provenance.json").read_text())
    vis = next(st for st in prov["samples"] if st["sample_id"] == "vis")
    assert vis["graph"]["mode"] == "grid" and vis["graph"]["n_rings"] == 1
    assert 5.0 < vis["graph"]["mean_degree"] <= 6.0      # 6 on a hex lattice, fewer at the edge
    assert vis["graph"]["n_isolated"] == 0
    assert (default_run / "vis_niches.png").stat().st_size > 1000


def test_flat_layout(fixture_root, tmp_path, monkeypatch):
    out = tmp_path / "out"
    assert run(fixture_root["flat"], out, monkeypatch, N_NICHES="4", MAX_PLOTS="1") == 0
    s = json.loads((out / "cell-type" / "summary.json").read_text())
    assert s["n_samples"] == 3 and s["n_niches"] == 4
    assert zscores(out / "cell-type" / "nhood_enrichment.csv", "pooled")[("Tumor", "B cell")] < -2
    pngs = [p.name for p in (out / "cell-type").iterdir() if p.name.startswith("co_occurrence_")]
    assert len(pngs) == 1


def test_knn_neighbours_and_niche_k(fixture_root, tmp_path, monkeypatch):
    out = tmp_path / "out"
    assert run(fixture_root["linked"], out, monkeypatch, NEIGHBOURS="8", NICHE_K="10", N_NICHES="3",
               LABEL="knn", N_INTERVALS="5") == 0
    prov = json.loads((out / "knn" / "provenance.json").read_text())
    xa = next(st for st in prov["samples"] if st["sample_id"] == "xa")
    assert xa["graph"]["mode"] == "knn" and xa["graph"]["n_neighs"] == 8
    assert 7.5 <= xa["graph"]["mean_degree"] <= 12   # symmetrised, so at least k minus the radius-capped edges
    assert zscores(out / "knn" / "nhood_enrichment.csv", "xa")[("Tumor", "Tumor")] > 3
    cooc = [r for r in rows(out / "knn" / "co_occurrence.csv") if r["sample_id"] == "xa"]
    assert len({r["distance"] for r in cooc}) == 5


def test_labels_share_an_output_folder(fixture_root, tmp_path, monkeypatch):
    out = tmp_path / "out"
    assert run(fixture_root["linked"], out, monkeypatch, LABEL_COLUMN="cluster", N_NICHES="3", MAX_PLOTS="0") == 0
    assert run(fixture_root["linked"], out, monkeypatch, LABEL_COLUMN="cluster", LABEL="domains", N_NICHES="3",
               MAX_PLOTS="0") == 0
    assert sorted(p.name for p in out.iterdir()) == ["cluster", "domains"]
    s = json.loads((out / "cluster" / "summary.json").read_text())
    assert s["labels"] == ["C01", "C02", "C03", "C04", "C05", "C06"]


@pytest.mark.parametrize("label", ["Cell_Type", "cell type", "a/b", "raw.1"])
def test_bad_label(fixture_root, tmp_path, monkeypatch, label):
    assert run(fixture_root["linked"], tmp_path / "out", monkeypatch, LABEL=label) == 2


@pytest.mark.parametrize("params", [{"NEIGHBOURS": "six"}, {"N_NICHES": "1"}, {"N_PERMS": "0"}, {"NICHE_HOPS": "0"},
                                    {"MAX_NEIGHBOUR_DISTANCE": "-1"}])
def test_bad_parameters(fixture_root, tmp_path, monkeypatch, params):
    assert run(fixture_root["linked"], tmp_path / "out", monkeypatch, **params) == 2


def test_default_label_is_the_column():
    assert app.default_label("cell_type") == "cell-type"
    assert app.default_label("dominant_type") == "dominant-type"
    assert app.default_label("domain") == "domain"


def test_missing_label_column_fails(fixture_root, tmp_path, monkeypatch):
    assert run(fixture_root["linked"], tmp_path / "out", monkeypatch, LABEL_COLUMN="dominant_type") == 1


def test_no_input_fails(tmp_path, monkeypatch):
    (tmp_path / "in").mkdir()
    assert run(tmp_path / "in", tmp_path / "out", monkeypatch) == 1


def test_neighbourhood_composition_hops():
    # a path 0-1-2-3: with 1 hop node 0 sees {0,1}; with 2 hops {0,1,2}
    g = sp.csr_matrix(np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], dtype=np.float32))
    codes = np.array([0, 1, 1, 0])
    one = N.neighbourhood_composition(g, codes, 2, hops=1)
    two = N.neighbourhood_composition(g, codes, 2, hops=2)
    assert np.allclose(one[0], [0.5, 0.5]) and np.allclose(two[0], [1 / 3, 2 / 3])
    assert np.allclose(one.sum(axis=1), 1) and np.allclose(two.sum(axis=1), 1)
    knn = N.neighbourhood_composition(g, codes, 2, hops=1, coords=np.array([[0, 0], [1, 0], [2, 0], [3, 0.0]]), k=1)
    assert np.allclose(knn[3], [0.5, 0.5])


def test_choose_n_niches_finds_three_blobs():
    rng = np.random.default_rng(0)
    centres = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    comp = np.vstack([c + rng.normal(0, 0.04, (300, 3)) for c in centres]).astype(np.float32)
    k, scores = N.choose_n_niches(comp, seed=1)
    assert k == 3 and scores[3] > scores[5]
    labels, names, info = N.cluster_niches(comp, 0, seed=1)
    assert names == ["N01", "N02", "N03"] and info["n_niches"] == 3
    assert np.bincount(labels).tolist() == [300, 300, 300]


def test_co_occurrence_subsample_budget():
    rng = np.random.default_rng(0)
    assert len(N.co_occurrence_subsample(100, 0, 20, 5, rng)) == 100
    assert len(N.co_occurrence_subsample(100, 40, 20, 5, rng)) == 40
    # 4 B x 20 intervals x 50^2 labels = 200 kB per observation -> the 1 GB budget allows 5000
    assert len(N.co_occurrence_subsample(100000, 20000, 20, 50, rng)) == 5000


def test_delaunay_radius_cap():
    rng = np.random.default_rng(3)
    xy = np.vstack([rng.uniform(0, 100, (200, 2)), rng.uniform(500, 600, (200, 2))])   # two islands
    capped, info = build_graph(xy, {"grid": "none", "spot_diameter": 10.0}, 1, "delaunay", 3.0)
    uncapped, _ = build_graph(xy, {"grid": "none", "spot_diameter": 10.0}, 1, "delaunay", 0.0)
    assert info["radius_cap"] == 30.0 and capped.nnz < uncapped.nnz
    # no edge between the islands once capped
    assert capped[:200, 200:].nnz == 0 and uncapped[:200, 200:].nnz > 0
    assert (capped != capped.T).nnz == 0 and capped.diagonal().sum() == 0


def test_seed_is_per_sample():
    assert N.sample_seed(42, "a") != N.sample_seed(42, "b") and N.sample_seed(42, "a") == N.sample_seed(42, "a")
