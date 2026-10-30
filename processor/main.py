"""
st-niches: how are the labelled observations of a spatial
transcriptomics dataset organized in space?

For one label column of labels.csv (cell types from st-celltype, cell2location's
dominant_type, or st-domains' domain) it computes per sample, on the spatial
neighbour graph of the workflow conventions:

  neighbourhood enrichment   squidpy permutation z-scores, label x label, plus a pooled table
  co-occurrence              squidpy co-occurrence ratio over distance intervals
  interaction matrix         row-normalized neighbour fractions
  cellular niches            KMeans clusters of each observation's neighbourhood composition,
                             jointly over all samples (Schürch et al. 2020; Varrone et al. 2024)

Everything is written under OUTPUT_DIR/<LABEL>/ so several runs (cell types,
domains) can be linked into one folder of the report step.
"""
from __future__ import annotations

import json
import logging
import math
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
log = logging.getLogger("st-niches")
log.setLevel(logging.INFO)
for noisy in ("matplotlib", "PIL", "fontTools", "numba", "h5py"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

APP_NAME = "st-niches"
REPO_ROOT = Path(__file__).resolve().parent.parent
LABEL_RE = re.compile(r"^[a-z0-9-]+$")

DEFAULTS = {
    "LABEL": "",
    "LABEL_COLUMN": "cell_type",
    "N_RINGS": "1",
    "NEIGHBOURS": "delaunay",
    "MAX_NEIGHBOUR_DISTANCE": "3.0",
    "N_PERMS": "1000",
    "N_INTERVALS": "20",
    "MAX_OBS_FOR_COOCCURRENCE": "20000",
    "NICHE_HOPS": "2",
    "NICHE_K": "0",
    "N_NICHES": "0",
    "MAX_PLOTS": "6",
    "SEED": "42",
    "THREADS": "0",
}
PARAMETERS = tuple(DEFAULTS)


class ParameterError(ValueError):
    pass


def default_label(label_column: str) -> str:
    """The conventions' default LABEL: the label column with `_` as `-` (and lower case)."""
    return label_column.strip().replace("_", "-").lower()


def read_parameters(env=os.environ) -> dict:
    raw = {k: (env.get(k) if env.get(k) not in (None, "") else v) for k, v in DEFAULTS.items()}
    p: dict = {}
    p["LABEL_COLUMN"] = raw["LABEL_COLUMN"].strip()
    if not p["LABEL_COLUMN"]:
        raise ParameterError("LABEL_COLUMN must name a column of labels.csv")
    p["LABEL"] = raw["LABEL"].strip() or default_label(p["LABEL_COLUMN"])
    if not LABEL_RE.match(p["LABEL"]):
        raise ParameterError(f"LABEL must match [a-z0-9-]+, got {p['LABEL']!r} (set LABEL explicitly)")
    nb = raw["NEIGHBOURS"].strip().lower()
    if nb == "delaunay":
        p["NEIGHBOURS"] = "delaunay"
    else:
        try:
            p["NEIGHBOURS"] = int(nb)
        except ValueError as exc:
            raise ParameterError(f"NEIGHBOURS must be 'delaunay' or an integer k, got {raw['NEIGHBOURS']!r}") from exc
        if p["NEIGHBOURS"] < 1:
            raise ParameterError("NEIGHBOURS k must be >= 1")
    try:
        for name in ("N_RINGS", "N_PERMS", "N_INTERVALS", "MAX_OBS_FOR_COOCCURRENCE", "NICHE_HOPS", "NICHE_K",
                     "N_NICHES", "MAX_PLOTS", "SEED", "THREADS"):
            p[name] = int(raw[name])
        p["MAX_NEIGHBOUR_DISTANCE"] = float(raw["MAX_NEIGHBOUR_DISTANCE"])
    except ValueError as exc:
        raise ParameterError(f"bad numeric parameter: {exc}") from exc
    if p["N_RINGS"] < 1 or p["N_PERMS"] < 1 or p["N_INTERVALS"] < 1 or p["NICHE_HOPS"] < 1:
        raise ParameterError("N_RINGS, N_PERMS, N_INTERVALS and NICHE_HOPS must be >= 1")
    if min(p["MAX_OBS_FOR_COOCCURRENCE"], p["NICHE_K"], p["N_NICHES"], p["MAX_PLOTS"], p["THREADS"]) < 0:
        raise ParameterError("MAX_OBS_FOR_COOCCURRENCE, NICHE_K, N_NICHES, MAX_PLOTS and THREADS must be >= 0")
    if p["MAX_NEIGHBOUR_DISTANCE"] < 0:
        raise ParameterError("MAX_NEIGHBOUR_DISTANCE must be >= 0 (0 = no cap)")
    if p["N_NICHES"] == 1:
        raise ParameterError("N_NICHES must be 0 (choose) or >= 2")
    return p


def cgroup_cpus() -> int | None:
    """CPUs from the cgroup quota (v2 cpu.max, v1 cfs quota), None without a quota."""
    try:
        text = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if text and text[0] != "max":
            return max(1, math.ceil(int(text[0]) / int(text[1])))
    except (OSError, ValueError, IndexError):
        pass
    try:
        quota = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if quota > 0 and period > 0:
            return max(1, math.ceil(quota / period))
    except (OSError, ValueError):
        pass
    return None


def set_threads(threads: int) -> int:
    """Pin the BLAS/numba thread count before numpy is imported (THREADS=0: the cgroup quota)."""
    n = threads if threads > 0 else cgroup_cpus()
    if n is None:
        try:
            n = len(os.sched_getaffinity(0))
        except AttributeError:
            n = min(4, os.cpu_count() or 1)
    n = max(1, int(n))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = str(n)
    return n


def _rel(path, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return str(path)


def _clean_json(obj):
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {str(k): _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v) for v in obj]
    if hasattr(obj, "item"):   # numpy scalars
        return _clean_json(obj.item())
    return obj


def write_json(path: Path, data) -> None:
    with open(path, "w") as fh:
        json.dump(_clean_json(data), fh, indent=2)
        fh.write("\n")


def write_table(path: Path, rows, columns: list[str]) -> None:
    import pandas as pd

    frame = pd.DataFrame(list(rows), columns=columns) if not hasattr(rows, "columns") else rows[columns]
    frame.to_csv(path, index=False, float_format="%.6g")


def tool_versions() -> dict:
    from importlib.metadata import version

    out = {"python": platform.python_version()}
    for name in ("squidpy", "scanpy", "anndata", "scikit-learn", "numpy", "scipy", "pandas", "matplotlib", "numba"):
        try:
            out[name] = version(name)
        except Exception:   # not installed
            pass
    return out


def peak_rss_mb() -> float | None:
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)   # bytes on macOS, KiB on Linux
    except (ImportError, OSError):
        return None


def app_version() -> str:
    try:
        return (REPO_ROOT / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


def main() -> int:
    started = datetime.now(timezone.utc)
    t0 = time.monotonic()
    input_dir, output_dir = os.environ.get("INPUT_DIR"), os.environ.get("OUTPUT_DIR")
    if not input_dir or not output_dir:
        log.error("INPUT_DIR and OUTPUT_DIR must be set")
        return 2
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.is_dir():
        log.error("INPUT_DIR %s is not a folder", input_dir)
        return 2
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
    try:
        params = read_parameters()
    except ParameterError as exc:
        log.error("%s", exc)
        return 2
    n_threads = set_threads(params["THREADS"])

    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    from processor import figures as F
    from processor import niches as N
    from processor.graph import build_graph
    from processor.inputs import discover, read_labels, read_sample

    try:
        found = discover(input_dir, params["LABEL_COLUMN"])
    except (OSError, ValueError) as exc:
        log.error("cannot read the inputs: %s", exc)
        return 1
    for name, reason in found.skipped.items():
        log.warning("%s: skipped, %s", name, reason)
    if not found.samples:
        log.error("no .h5ad samples found under %s", input_dir)
        return 1
    if found.labels_csv is None:
        log.error("no labels.csv found under %s", input_dir)
        return 1
    try:
        labels = read_labels(found.labels_csv, params["LABEL_COLUMN"])
    except (OSError, ValueError) as exc:
        log.error("cannot read %s: %s", found.labels_csv, exc)
        return 1
    categories = sorted({v for v in labels["label"].unique() if v})
    if len(categories) < 2:
        log.error("%s: fewer than 2 distinct %s labels (%s)", found.labels_csv.name, params["LABEL_COLUMN"], categories)
        return 1
    cat_index = {c: i for i, c in enumerate(categories)}
    n_cls = len(categories)
    log.info("%d samples, %d labels in %s (%s), label %s, %d threads", len(found.samples), n_cls,
             params["LABEL_COLUMN"], ", ".join(categories), params["LABEL"], n_threads)

    # --- per sample ------------------------------------------------------------------------
    kept = []          # dicts with sample_id, obs_names, coords, codes, graph, comp, image, ...
    statuses = []
    enrich_rows, cooc_rows, inter_rows, graph_infos = [], [], [], {}
    t_samples = time.monotonic()
    for s in found.samples:
        t = time.monotonic()
        status = {"sample_id": s.sample_id, "file": _rel(s.path, input_dir)}
        try:
            data = read_sample(s)
            n_file = len(data.obs_names)
            lab = labels["label"].reindex(data.obs_names)
            sid_of = labels["sample_id"].reindex(data.obs_names)
            if lab.notna().any() and (sid_of.dropna() != s.sample_id).all() and sid_of.notna().any():
                log.warning("%s: labels.csv lists these obs_ids under sample_id %s", s.sample_id,
                            sid_of.dropna().iloc[0])
            has = lab.notna().to_numpy() & (lab.fillna("").to_numpy() != "")
            n_unlabelled = int((~has).sum())
            if has.sum() < 2:
                raise ValueError(f"{int(has.sum())} of {n_file} observations have a {params['LABEL_COLUMN']} label")
            obs_names = data.obs_names[has]
            coords = data.coords[has]
            codes = np.array([cat_index[v] for v in lab.to_numpy()[has]], dtype=np.int64)
            graph, ginfo = build_graph(coords, data.pennsieve, params["N_RINGS"], params["NEIGHBOURS"],
                                       params["MAX_NEIGHBOUR_DISTANCE"])
            graph_infos[s.sample_id] = ginfo
            present = np.unique(codes)
            if len(present) < 2:
                raise ValueError(f"only one label ({categories[present[0]]}) in this sample")
            seed = N.sample_seed(params["SEED"], s.sample_id)
            rng = np.random.default_rng(seed)
            t1 = time.monotonic()
            z, cnt = N.nhood_enrichment(graph, codes, categories, params["N_PERMS"], seed, n_jobs=n_threads)
            t_enrich = time.monotonic() - t1
            enrich_rows += N.pair_table(z, categories, "zscore", {"sample_id": s.sample_id}, cnt, "count")
            inter = N.interaction_matrix(graph, codes, categories)
            inter_rows += N.pair_table(inter, categories, "fraction", {"sample_id": s.sample_id})
            t1 = time.monotonic()
            rows = N.co_occurrence_subsample(len(codes), params["MAX_OBS_FOR_COOCCURRENCE"], params["N_INTERVALS"],
                                             len(present), rng)
            occ, dist = N.co_occurrence(coords[rows], codes[rows], categories, params["N_INTERVALS"],
                                        ginfo.get("spot_diameter"))
            t_cooc = time.monotonic() - t1
            for i, a in enumerate(categories):
                for j, b in enumerate(categories):
                    for d_i, d in enumerate(dist):
                        cooc_rows.append({"sample_id": s.sample_id, "label_a": a, "label_b": b, "distance": float(d),
                                          "ratio": float(occ[i, j, d_i])})
            comp = N.neighbourhood_composition(graph, codes, n_cls, params["NICHE_HOPS"], coords, params["NICHE_K"])
            kept.append({"sample_id": s.sample_id, "obs_names": obs_names, "coords": coords, "codes": codes,
                         "graph": graph, "comp": comp, "zscore": z, "count": cnt, "occ": occ, "dist": dist,
                         "image": data.image, "image_scale": data.image_scale, "pennsieve": data.pennsieve,
                         "n_cooc": int(len(rows))})
            status.update(status="ok", n_obs_file=n_file, n_obs=int(has.sum()), n_unlabelled=n_unlabelled,
                          n_labels=int(len(present)), graph=ginfo, n_obs_co_occurrence=int(len(rows)),
                          seconds=round(time.monotonic() - t, 2),
                          seconds_enrichment=round(t_enrich, 2), seconds_co_occurrence=round(t_cooc, 2))
            log.info("%s: %d obs (%d unlabelled dropped), %d labels, %s graph (%d edges, mean degree %.1f, %d isolated),"
                     " enrichment %.1fs, co-occurrence on %d obs %.1fs, %.1fs total", s.sample_id, int(has.sum()),
                     n_unlabelled, len(present), ginfo["mode"], ginfo["n_edges"], ginfo["mean_degree"],
                     ginfo["n_isolated"], t_enrich, len(rows), t_cooc, time.monotonic() - t)
        except Exception as exc:   # one bad sample is skipped, not fatal
            status.update(status="failed", error=f"{type(exc).__name__}: {exc}", seconds=round(time.monotonic() - t, 2))
            log.error("%s: failed: %s", s.sample_id, exc, exc_info=True)
        statuses.append(status)
    t_samples = time.monotonic() - t_samples
    rss_samples = peak_rss_mb()
    if not kept:
        log.error("every sample failed")
        return 1

    # --- pooled enrichment: one block-diagonal graph, labels shuffled within each sample ------
    t_pool = time.monotonic()
    all_codes = np.concatenate([k["codes"] for k in kept])
    libraries = np.concatenate([np.repeat(k["sample_id"], len(k["codes"])) for k in kept])
    pooled_method = "pooled permutations (labels shuffled within each sample)"
    try:
        big = sp.block_diag([k["graph"] for k in kept], format="csr")
        z_pool, cnt_pool = N.nhood_enrichment(big, all_codes, categories, params["N_PERMS"], params["SEED"],
                                              libraries=libraries, n_jobs=n_threads)
        del big
    except Exception as exc:
        log.warning("pooled enrichment failed (%s); using the mean of the per-sample z-scores", exc)
        pooled_method = "mean of per-sample z-scores"
        z_pool = np.nanmean(np.stack([k["zscore"] for k in kept]), axis=0)
        cnt_pool = np.sum(np.stack([k["count"] for k in kept]), axis=0)
    enrich_rows += N.pair_table(z_pool, categories, "zscore", {"sample_id": "pooled"}, cnt_pool, "count")
    t_pool = time.monotonic() - t_pool
    log.info("pooled enrichment over %d samples, %d obs: %.1fs", len(kept), len(all_codes), t_pool)

    # --- niches: KMeans over every observation's neighbourhood composition --------------------
    t_niche = time.monotonic()
    comp_all = np.concatenate([k["comp"] for k in kept])
    niche_codes, niche_names, niche_info = N.cluster_niches(comp_all, params["N_NICHES"], params["SEED"])
    n_niches = len(niche_names)
    offsets = np.cumsum([0] + [len(k["codes"]) for k in kept])
    for i, k in enumerate(kept):
        k["niche"] = niche_codes[offsets[i]:offsets[i + 1]]
    t_niche = time.monotonic() - t_niche
    log.info("niches: %d (%s%s), %.1fs", n_niches, niche_info.get("chosen_by", "N_NICHES"),
             "; silhouette " + ", ".join(f"k={k}: {v:.3f}" for k, v in niche_info.get("silhouette", {}).items())
             if niche_info.get("silhouette") else "", t_niche)

    # niche composition: observed label fractions per niche, and the mean neighbourhood composition
    overall = np.bincount(all_codes, minlength=n_cls) / len(all_codes)
    comp_rows = []
    niche_fraction = np.zeros((n_niches, n_cls))
    niche_nhood = np.zeros((n_niches, n_cls))
    for q in range(n_niches):
        members = niche_codes == q
        niche_fraction[q] = np.bincount(all_codes[members], minlength=n_cls) / max(1, members.sum())
        niche_nhood[q] = comp_all[members].mean(axis=0) if members.any() else 0.0
        for j, c in enumerate(categories):
            comp_rows.append({"niche": niche_names[q], "label": c, "fraction": float(niche_fraction[q, j]),
                              "enrichment": float(niche_fraction[q, j] / overall[j]) if overall[j] > 0 else float("nan"),
                              "neighbourhood_fraction": float(niche_nhood[q, j]), "n_obs": int(members.sum())})
    summary_rows = []
    for k in kept:
        counts = np.bincount(k["niche"], minlength=n_niches)
        for q in range(n_niches):
            summary_rows.append({"sample_id": k["sample_id"], "niche": niche_names[q], "n_obs": int(counts[q]),
                                 "fraction": float(counts[q] / len(k["niche"]))})

    # --- tables ------------------------------------------------------------------------------
    out = output_dir / params["LABEL"]
    out.mkdir(parents=True, exist_ok=True)
    label_frame = pd.DataFrame({
        "sample_id": np.concatenate([np.repeat(k["sample_id"], len(k["codes"])) for k in kept]),
        "obs_id": np.concatenate([k["obs_names"] for k in kept]),
        "niche": np.asarray(niche_names)[niche_codes],
    })
    label_frame.to_csv(out / "labels.csv", index=False)
    write_table(out / "nhood_enrichment.csv", enrich_rows, ["sample_id", "label_a", "label_b", "zscore", "count"])
    write_table(out / "co_occurrence.csv", cooc_rows, ["sample_id", "label_a", "label_b", "distance", "ratio"])
    write_table(out / "interactions.csv", inter_rows, ["sample_id", "label_a", "label_b", "fraction"])
    write_table(out / "niche_composition.csv", comp_rows,
                ["niche", "label", "fraction", "enrichment", "neighbourhood_fraction", "n_obs"])
    write_table(out / "niche_summary.csv", summary_rows, ["sample_id", "niche", "n_obs", "fraction"])

    # --- summary -----------------------------------------------------------------------------
    pairs = []
    for i, a in enumerate(categories):
        for j, b in enumerate(categories):
            if j < i or not np.isfinite(z_pool[i, j]):
                continue
            pairs.append({"label_a": a, "label_b": b, "zscore": round(float(z_pool[i, j]), 3),
                          "count": int(cnt_pool[i, j]), "homotypic": a == b})
    enriched = sorted([p for p in pairs if p["zscore"] > 0], key=lambda p: -p["zscore"])
    depleted = sorted([p for p in pairs if p["zscore"] < 0], key=lambda p: p["zscore"])
    niche_top = []
    for q in range(n_niches):
        order = np.argsort(-niche_fraction[q])[:3]
        niche_top.append({"niche": niche_names[q], "n_obs": int((niche_codes == q).sum()),
                          "fraction": round(float((niche_codes == q).mean()), 4),
                          "top_labels": [{"label": categories[j], "fraction": round(float(niche_fraction[q, j]), 4),
                                          "enrichment": round(float(niche_fraction[q, j] / overall[j]), 3) if overall[j] else None}
                                         for j in order if niche_fraction[q, j] > 0]})
    summary = {
        "label": params["LABEL"], "label_column": params["LABEL_COLUMN"],
        "n_samples": len(kept), "n_samples_failed": len(statuses) - len(kept),
        "n_obs": int(len(all_codes)), "n_labels": n_cls, "labels": categories,
        "label_fractions": {c: round(float(overall[j]), 5) for j, c in enumerate(categories)},
        "per_sample": {k["sample_id"]: {"n_obs": int(len(k["codes"])), "n_labels": int(len(np.unique(k["codes"]))),
                                        "graph": graph_infos[k["sample_id"]]["mode"],
                                        "mean_degree": graph_infos[k["sample_id"]]["mean_degree"],
                                        "n_obs_co_occurrence": k["n_cooc"]} for k in kept},
        "pooled_zscore_method": pooled_method,
        "strongest_enriched": enriched[:5], "strongest_depleted": depleted[:5],
        "strongest_enriched_heterotypic": [p for p in enriched if not p["homotypic"]][:5],
        "n_pairs_enriched": sum(1 for p in pairs if p["zscore"] > 2), "n_pairs_depleted": sum(1 for p in pairs if p["zscore"] < -2),
        "n_niches": n_niches, "niche_selection": niche_info, "niches": niche_top,
        "parameters": params, "n_threads": n_threads,
    }

    # --- figures -----------------------------------------------------------------------------
    t_fig = time.monotonic()
    figure_paths = []
    try:
        figure_paths = make_figures(out, params, categories, kept, z_pool, niche_names, niche_fraction, F)
    except Exception as exc:   # figures are a convenience; the tables stand without them
        log.exception("figures failed: %s", exc)
    t_fig = time.monotonic() - t_fig
    summary["figures"] = [p.name for p in figure_paths]
    write_json(out / "summary.json", summary)

    n_failed = len(statuses) - len(kept)
    write_json(out / "provenance.json", {
        "app": APP_NAME, "version": app_version(), "tools": tool_versions(),
        "parameters": params, "n_threads": n_threads,
        "method": {
            "graph": "squidpy spatial neighbours per the workflow conventions (grid for hex, else Delaunay or kNN "
                     "with a radius cap of MAX_NEIGHBOUR_DISTANCE x spot_diameter)",
            "nhood_enrichment": f"squidpy.gr.nhood_enrichment, {params['N_PERMS']} permutations, seeded per sample; "
                                f"pooled: {pooled_method}",
            "co_occurrence": f"squidpy.gr.co_occurrence on a seeded subsample, thresholds from 0 to "
                             f"{N.COOCCURRENCE_RANGE:g} x spot_diameter (at most half the sample's extent) in "
                             f"N_INTERVALS equal steps",
            "interaction_matrix": "squidpy.gr.interaction_matrix, row-normalized",
            "niches": "KMeans (scikit-learn, k-means++, 10 inits) over the fraction of each label among the "
                      "observation and its neighbours within NICHE_HOPS hops (or the NICHE_K nearest), all samples "
                      "jointly; N_NICHES=0 picks k in 3..10 by the silhouette on a subsample; niches numbered by size",
        },
        "inputs": {"spatial": [_rel(p, input_dir) for p in found.files_seen],
                   "labels_csv": _rel(found.labels_csv, input_dir), "samples_csv": _rel(found.samples_csv, input_dir),
                   "skipped": found.skipped},
        "started": started.isoformat(), "finished": datetime.now(timezone.utc).isoformat(),
        "timings_seconds": {"samples": round(t_samples, 2), "pooled_enrichment": round(t_pool, 2),
                            "niches": round(t_niche, 2), "figures": round(t_fig, 2),
                            "total": round(time.monotonic() - t0, 2)},
        "peak_rss_mb": {"after_samples": rss_samples, "total": peak_rss_mb()},
        "samples": statuses, "n_ok": len(kept), "n_failed": n_failed,
        "outputs": sorted(p.name for p in out.iterdir()) + ["provenance.json"],
    })
    log.info("done: %d samples (%d failed), %d obs, %d labels, %d niches, %d enriched / %d depleted pairs (|z| > 2), "
             "%.1fs, peak RSS %s MB -> %s", len(kept), n_failed, len(all_codes), n_cls, n_niches,
             summary["n_pairs_enriched"], summary["n_pairs_depleted"], time.monotonic() - t0, peak_rss_mb(), out)
    return 0


def make_figures(out: Path, params: dict, categories: list[str], kept: list[dict], z_pool, niche_names: list[str],
                 niche_fraction, F) -> list[Path]:
    import numpy as np

    paths = []
    zs = [z_pool] + [k["zscore"] for k in kept]
    finite = np.concatenate([np.abs(z[np.isfinite(z)]).ravel() for z in zs])
    vmax = float(np.percentile(finite, 98)) if finite.size else 1.0
    vmax = max(2.0, float(np.ceil(vmax)))
    paths.append(F.zscore_heatmap([(f"pooled over {len(kept)} samples", z_pool)], categories,
                                  f"Neighbourhood enrichment · {params['LABEL_COLUMN']} · {params['LABEL']}", vmax,
                                  out / "nhood_enrichment.png"))
    for k in kept:
        paths.append(F.zscore_heatmap([(k["sample_id"], k["zscore"])], categories,
                                      f"Neighbourhood enrichment · {k['sample_id']} · {params['LABEL']}", vmax,
                                      out / f"{k['sample_id']}_nhood_enrichment.png"))
    paths.append(F.composition_heatmap(niche_names, categories, niche_fraction,
                                       f"Niche composition (fraction of observations) · {params['LABEL']}",
                                       out / "niche_composition.png"))
    for k in kept:
        paths.append(F.niche_scatter(k["sample_id"], k["coords"], k["niche"], niche_names, k["image"], k["image_scale"],
                                     k["pennsieve"].get("spot_diameter"), f"Niches · {k['sample_id']} · {params['LABEL']}",
                                     out / f"{k['sample_id']}_niches.png"))
    # Co-occurrence: the labels whose short-range ratio with some other label departs most from 1 (pooled = mean
    # over samples), MAX_PLOTS of them.
    occ = np.stack([k["occ"] for k in kept])   # samples x a x b x intervals
    with np.errstate(invalid="ignore"):
        mean_occ = np.nanmean(occ, axis=0)
    interest = []
    for i, a in enumerate(categories):
        short = mean_occ[i, :, 0]
        others = [abs(np.log2(short[j])) for j in range(len(categories)) if j != i and np.isfinite(short[j]) and short[j] > 0]
        interest.append((max(others) if others else -1.0, a))
    ranked = [a for score, a in sorted(interest, key=lambda t: -t[0]) if score >= 0]
    dist = kept[0]["dist"] if len({len(k["dist"]) for k in kept}) == 1 else None
    if dist is not None:
        dist = np.mean(np.stack([k["dist"] for k in kept]), axis=0)
        for a in ranked[: max(0, params["MAX_PLOTS"])]:
            i = categories.index(a)
            paths.append(F.co_occurrence_plot(a, categories, dist, mean_occ[i], len(kept),
                                              f"Co-occurrence with {a} · {params['LABEL']}",
                                              out / f"co_occurrence_{F.safe_name(a)}.png"))
    return paths


if __name__ == "__main__":
    sys.exit(main())
