"""
PNG figures (150 dpi): label x label z-score heatmaps (diverging blue-grey-red,
symmetric about 0), the niche x label composition heatmap (one-hue sequential),
spatial scatters coloured by niche (fixed categorical slots, y inverted, on the
lowres H&E when there is one), and co-occurrence ratio vs distance lines.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
          "#0d366b", "#8a4b1c", "#0a5a3f", "#7a5200", "#8c2f55", "#003f00", "#2d2170", "#7a1d1c"]
SEQUENTIAL = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#f0efec", "#f5b9b8", "#ec7e7d", "#d03b3b", "#7a1d1c"]
DPI = 150


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
    })
    return plt


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or "label"


def _style_matrix_axes(ax, rows, cols):
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=7.5, color=INK_2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7.5, color=INK_2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.5)
    ax.tick_params(which="minor", length=0)


def zscore_heatmap(panels: list[tuple[str, np.ndarray]], labels: list[str], title: str, vmax: float, path: Path) -> Path:
    """One panel per (subtitle, label x label z-score matrix); NaN = no value (label absent)."""
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("z", DIVERGING).with_extremes(bad=SURFACE)   # absent = blank, not ~0
    n = len(labels)
    side = max(4.0, 0.3 * n + 1.8)
    fig, axes = plt.subplots(1, len(panels), figsize=(side * len(panels) + 1.2, side + 0.9), squeeze=False)
    image = None
    for ax, (subtitle, z) in zip(axes[0], panels):
        image = ax.imshow(np.ma.masked_invalid(z), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="equal",
                          interpolation="nearest")
        _style_matrix_axes(ax, labels, labels)
        ax.set_title(subtitle, loc="left", fontsize=9, fontweight="normal", color=INK_2)
        if n <= 16:
            for i in range(n):
                for j in range(n):
                    v = z[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                                color=SURFACE if abs(v) > 0.6 * vmax else INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 0.87, 0.95))
    cax = fig.add_axes((0.89, 0.3, 0.015, 0.45))
    cb = fig.colorbar(image, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=MUTED, length=0)
    cb.set_label("z-score", fontsize=7.5, color=INK_2)
    fig.text(0.01, 0.008, "Rows: the label; columns: its neighbours. Red: more neighbour pairs than in permuted labels,\n"
             "blue: fewer; blank: label absent from the sample. With up to 16 labels the z-scores are printed in the cells.",
             fontsize=7, color=MUTED)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def composition_heatmap(niches: list[str], labels: list[str], fraction: np.ndarray, title: str, path: Path) -> Path:
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("frac", SEQUENTIAL)
    fig, ax = plt.subplots(figsize=(max(5.0, 0.32 * len(labels) + 2.2), max(2.6, 0.32 * len(niches) + 1.8)))
    vmax = max(0.05, float(np.nanmax(fraction))) if fraction.size else 1.0
    image = ax.imshow(fraction, cmap=cmap, vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
    _style_matrix_axes(ax, niches, labels)
    if len(labels) <= 20:
        for i in range(len(niches)):
            for j in range(len(labels)):
                v = fraction[i, j]
                if v >= 0.005:
                    ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=6,
                            color=SURFACE if v > 0.6 * vmax else INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 0.86, 0.95))
    cax = fig.add_axes((0.88, 0.3, 0.015, 0.45))
    cb = fig.colorbar(image, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=MUTED, length=0)
    cb.set_label("fraction", fontsize=7.5, color=INK_2)
    fig.text(0.01, 0.008, "Numbers: percent of the niche's observations carrying the label.", fontsize=7, color=MUTED)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def niche_scatter(sample_id: str, coords: np.ndarray, niche: np.ndarray, niche_names: list[str], image, image_scale: float,
                  spot_diameter, title: str, path: Path) -> Path:
    plt = _plt()

    xy = coords * (image_scale if image is not None else 1.0)
    span = xy.max(axis=0) - xy.min(axis=0)
    w = 7.0
    h = max(3.5, min(9.0, w * span[1] / span[0])) if span[0] > 0 else 6.0
    fig, ax = plt.subplots(figsize=(w + 1.8, h + 0.9))
    if image is not None:
        ax.imshow(image, origin="upper", interpolation="nearest")
    n = len(niche)
    size = max(0.3, min(12.0, 40000.0 / max(n, 1)))
    rng = np.random.default_rng(0)
    order = rng.permutation(n)   # no niche hides under another
    colours = np.asarray([SERIES[i % len(SERIES)] for i in range(len(niche_names))])
    ax.scatter(xy[order, 0], xy[order, 1], s=size, c=colours[niche[order]], linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    if image is None:
        ax.invert_yaxis()   # image convention: y grows downwards
        ax.set_xlabel("x", color=INK_2)
        ax.set_ylabel("y", color=INK_2)
    else:
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.set_xticks([])
        ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)
    counts = np.bincount(niche, minlength=len(niche_names))
    handles = [plt.Line2D([], [], marker="o", linestyle="none", markersize=6, color=colours[i],
                          label=f"{nm}  {100 * counts[i] / max(n, 1):.0f}%") for i, nm in enumerate(niche_names)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.5, frameon=False,
              title="niche", title_fontsize=8)
    ax.set_title(f"{title}  ({n:,} observations)", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def co_occurrence_plot(label: str, labels: list[str], distances: np.ndarray, ratio: np.ndarray, n_samples: int,
                       title: str, path: Path, max_lines: int = 8) -> Path:
    """ratio: (labels x intervals) co-occurrence ratio of `label` with each label, mean over samples."""
    plt = _plt()

    i0 = labels.index(label)
    partners = [j for j in range(len(labels)) if np.isfinite(ratio[j]).any()]
    # the partners whose short-range ratio departs most from 1 are drawn; the rest fold into a grey band
    def departure(j):
        v = ratio[j, 0]
        return abs(np.log2(v)) if np.isfinite(v) and v > 0 else 0.0
    shown = sorted(partners, key=lambda j: (-departure(j), labels[j]))[:max_lines]
    shown = sorted(shown, key=lambda j: (j != i0, labels[j]))
    rest = [j for j in partners if j not in shown]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axhline(1.0, color=GRID, linewidth=1.0, zorder=1)
    if rest:
        block = ratio[rest]
        with np.errstate(invalid="ignore"):
            lo, hi = np.nanmin(block, axis=0), np.nanmax(block, axis=0)
        ax.fill_between(distances, lo, hi, color=GRID, alpha=0.6, linewidth=0, zorder=1,
                        label=f"{len(rest)} other labels (range)")
    for slot, j in enumerate(shown):
        style = dict(linewidth=2.2, linestyle="-") if j == i0 else dict(linewidth=1.8)
        ax.plot(distances, ratio[j], color=SERIES[slot % len(SERIES)], zorder=3,
                label=f"{labels[j]}" + (" (itself)" if j == i0 else ""), **style)
    ax.set_xlabel("distance (coordinate units)", color=INK_2)
    ax.set_ylabel(f"p({label} | neighbour within d) / p({label})", color=INK_2)
    ax.set_xlim(0, float(distances[-1]) * 1.02)
    ax.set_ylim(bottom=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
    ax.set_title(f"{title}  (mean over {n_samples} sample{'s' if n_samples != 1 else ''})", loc="left")
    fig.text(0.01, 0.008, "Ratio > 1: the label is found near the line's label more often than its overall share; "
             "the first interval holds the closest pairs.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path
