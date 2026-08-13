#!/usr/bin/env python3
"""Bar chart of % of bound metal for a 29-metal ICP-MS panel (wide triplicate).

Input format: header row lists each metal symbol followed by N-1 empty
("Unnamed") columns (one block per metal); the next row(s) hold the
replicate values in the same order. Auto-detects the number of replicates
per metal from the block width.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_percent_metal_bar import (  # noqa: E402
    _draw_diagonal_stripes,
    configure_nature_style,
    save,
)

# Full DSF palette from dsf_analysis.py, keyed by plain element symbol.
METAL_COLORS: dict[str, str] = {
    "Mn": "#FF1493", "Co": "#8B008B", "Ni": "#00C853", "Cu": "#1E90FF",
    "Nd": "#8A2BE2", "Dy": "#FFD700", "Pr": "#80FF20", "Er": "#FF4D80",
    "Ho": "#FF9D00", "Fe": "#ffffff",
    "Li": "#DCE8F0", "K":  "#E8E8D8", "Rb": "#9090A8", "Cs": "#707060",
    "Mg": "#D8D8D0", "Ca": "#C8D0D8", "Sr": "#A8B0A8", "Ba": "#585858",
    "Sc": "#989898", "Y":  "#686868", "Zn": "#C0C0C0",
    "La": "#484858", "Ce": "#A8C0B0", "Sm": "#C8C0B0", "Eu": "#C0B0C0",
    "Gd": "#686060", "Tb": "#606868", "Tm": "#A0A898", "Yb": "#787070",
    "Lu": "#404040",
}
ATOMIC_NUMBER: dict[str, int] = {
    "H": 1, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "Na": 11,
    "Mg": 12, "Al": 13, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23,
    "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "Cs": 55, "Ba": 56,
    "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Sm": 62, "Eu": 63, "Gd": 64,
    "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80, "Pb": 82, "Bi": 83, "U": 92,
}

PANEL_HEIGHT_MM = 45.0
PANEL_HEIGHT_IN = PANEL_HEIGHT_MM / 25.4


def main() -> None:
    args = parse_args()
    configure_nature_style()
    long_df = load_wide_triplicate(args.input)
    metals = list(args.metals) if args.metals else _sort_by_atomic_number(long_df.columns.tolist())
    missing = [m for m in metals if m not in long_df.columns]
    if missing:
        raise ValueError(f"Input is missing metal columns: {missing}")
    stats = _summarize(long_df, metals)
    not_spiked = frozenset(args.not_spiked or ())
    if args.heatmap:
        fig, _ = plot_bars_with_heatmap(stats, metals, title=args.title,
                                        ylabel=args.ylabel, ymax=args.ymax,
                                        panel_width_mm=args.panel_mm,
                                        not_spiked=not_spiked,
                                        cmax=args.cmax)
    else:
        fig, _ = plot(stats, metals, title=args.title, ylabel=args.ylabel,
                      ymax=args.ymax, panel_width_mm=args.panel_mm,
                      not_spiked=not_spiked)
    save(fig, args.output, args.formats)
    if args.show or not args.output:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", type=Path, required=True,
                   help="Path to input .xlsx or .csv (wide-triplicate format)")
    p.add_argument("-o", "--output", type=Path,
                   default=Path(__file__).resolve().parent.parent
                   / "outputs" / "lanm_icp_29metal_percent",
                   help="Output path WITHOUT extension.")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"],
                   help="Output formats (default: pdf png).")
    p.add_argument("--metals", nargs="+", default=None,
                   help="Override metal columns and left-to-right order.")
    p.add_argument("--not-spiked", nargs="+", default=(),
                   help="Metals to draw hatched (detected but not added).")
    p.add_argument("--title", default=None, help="Optional plot title")
    p.add_argument("--ylabel", default="% of bound metal",
                   help="Y-axis label (default: '%% of bound metal')")
    p.add_argument("--ymax", type=float, default=10.0,
                   help="Y-axis upper limit (default: 10)")
    p.add_argument("--panel-mm", type=float, default=90.0,
                   help="Figure width in mm (default: 90; height is 45 mm)")
    p.add_argument("--heatmap", action="store_true",
                   help="Bars rising out of a single-row heatmap (Nd-purple).")
    p.add_argument("--cmax", type=float, default=None,
                   help="Reference max for colorbar physical height "
                        "(default: --ymax). Heatmap color scale itself always "
                        "spans 0..data_max.")
    p.add_argument("--show", action="store_true", help="Show plot window")
    return p.parse_args()


def load_wide_triplicate(path: Path) -> pd.DataFrame:
    """Reshape a wide-triplicate sheet into a long DataFrame (rows=reps, cols=metals).

    Header row: metal symbol in the first column of each replicate block,
    empty/NaN for the remaining columns of the block. All blocks must have
    the same width (auto-detected as the run length between named columns).
    """
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        raw = pd.read_excel(path, header=None)
    elif suffix == ".csv":
        raw = pd.read_csv(path, header=None)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    if raw.shape[0] < 2:
        raise ValueError("Input must have at least one header row and one data row")

    header = raw.iloc[0].tolist()
    # Positions of named (non-NaN, non-empty) columns.
    named_positions = [
        i for i, v in enumerate(header)
        if v is not None and not (isinstance(v, float) and np.isnan(v)) and str(v).strip() != ""
    ]
    if not named_positions:
        raise ValueError("Header row has no metal names")

    # Block width = distance between consecutive named columns (must be uniform).
    starts = named_positions + [raw.shape[1]]
    widths = [starts[i + 1] - starts[i] for i in range(len(starts) - 1)]
    if len(set(widths)) != 1:
        raise ValueError(
            f"Replicate block widths are not uniform: {widths}. "
            "Each metal must have the same number of replicate columns."
        )
    n_reps = widths[0]

    metals = [str(header[i]).strip() for i in named_positions]
    data_rows = raw.iloc[1:].reset_index(drop=True)

    # Emit one long-format row per (data_row, replicate_slot).
    long_rows: list[dict] = []
    for _, row in data_rows.iterrows():
        for rep_idx in range(n_reps):
            long_rows.append({
                m: row.iloc[start + rep_idx]
                for m, start in zip(metals, named_positions)
            })
    return pd.DataFrame(long_rows)[metals]


def _sort_by_atomic_number(cols: list[str]) -> list[str]:
    return sorted(cols, key=lambda c: (ATOMIC_NUMBER.get(c, 10_000), c))


def _summarize(df: pd.DataFrame, metals: list[str]) -> pd.DataFrame:
    """Per-replicate metal fractions, then mean and SEM per metal (percent)."""
    vals = df[metals].apply(pd.to_numeric, errors="coerce").clip(lower=0)
    totals = vals.sum(axis=1)
    keep = totals > 0
    frac = vals.loc[keep].div(totals.loc[keep], axis=0) * 100.0
    n = frac.shape[0]
    mean = frac.mean(axis=0)
    sem = frac.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else pd.Series(0.0, index=metals)
    return pd.DataFrame({"metal": metals, "mean": mean.values, "sem": sem.values})


def plot(stats: pd.DataFrame, metals: list[str], title: str | None,
         ylabel: str, ymax: float | None, panel_width_mm: float,
         not_spiked: frozenset[str]) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(panel_width_mm / 25.4, PANEL_HEIGHT_IN))
    x = np.arange(len(metals))
    means = stats["mean"].to_numpy()
    sems = stats["sem"].to_numpy()

    bar_w = 0.75
    for xi, m, mean, sem in zip(x, metals, means, sems):
        color = METAL_COLORS.get(m, "#888888")
        ax.bar(xi, mean, width=bar_w, color=color, edgecolor="0.15",
               linewidth=0.5, zorder=2)
        if sem > 0:
            ax.errorbar(xi, mean, yerr=sem, fmt="none", ecolor="0.15",
                        elinewidth=0.5, capsize=1.2, capthick=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(metals, rotation=90 if len(metals) > 12 else 0)
    ax.set_xlim(-0.6, len(metals) - 0.4)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    else:
        top = float(np.nanmax(means + sems)) if len(means) else 1.0
        ax.set_ylim(0, top * 1.10 if top > 0 else 1.0)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", pad=1.5)
    ax.tick_params(axis="y", pad=1.5)
    fig.tight_layout(pad=0.2)

    fig.canvas.draw()
    for xi, m, mean in zip(x, metals, means):
        if m in not_spiked and mean > 0:
            _draw_diagonal_stripes(ax, xi - bar_w / 2, 0.0, bar_w, mean,
                                   spacing_pt=2.5, linewidth=0.5, color="0.15")
    return fig, ax


# Nd-purple: white -> #8A2BE2 (Nd fill color from the DSF palette).
_ND_PURPLE_CMAP = LinearSegmentedColormap.from_list(
    "nd_purple", ["#ffffff", "#8A2BE2"]
)


def plot_bars_with_heatmap(stats: pd.DataFrame, metals: list[str],
                           title: str | None, ylabel: str,
                           ymax: float | None, panel_width_mm: float,
                           not_spiked: frozenset[str],
                           cmax: float | None = None) -> tuple[plt.Figure, plt.Axes]:
    """Bars rising out of a single-row heatmap, both keyed to the same metric.

    Heatmap color scale runs 0 -> data_max (this dataset). The colorbar's
    physical height is shrunk to (data_max / cmax) so multiple panels drawn
    against a shared reference `cmax` (e.g. the bar y-axis limit) can be
    compared at a glance.
    """
    means = stats["mean"].to_numpy()
    sems = stats["sem"].to_numpy()
    if ymax is None:
        top = float(np.nanmax(means + sems)) if len(means) else 1.0
        ymax = top * 1.10 if top > 0 else 1.0
    if cmax is None:
        cmax = ymax
    data_max = float(np.nanmax(means)) if len(means) else 1.0
    data_max = max(data_max, cmax * 1e-6)

    fig = plt.figure(figsize=(panel_width_mm / 25.4, PANEL_HEIGHT_IN))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[1.0, 0.14], width_ratios=[1.0, 0.03],
        hspace=0.05, wspace=0.04,
    )
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_hm = fig.add_subplot(gs[1, 0], sharex=ax_bar)
    ax_cb = fig.add_subplot(gs[:, 1])

    x = np.arange(len(metals))
    bar_w = 0.75
    for xi, m, mean, sem in zip(x, metals, means, sems):
        color = METAL_COLORS.get(m, "#888888")
        ax_bar.bar(xi, mean, width=bar_w, color=color, edgecolor="0.15",
                   linewidth=0.5, zorder=2)
        if sem > 0:
            ax_bar.errorbar(xi, mean, yerr=sem, fmt="none", ecolor="0.15",
                            elinewidth=0.5, capsize=1.2, capthick=0.5, zorder=3)

    ax_bar.set_xlim(-0.6, len(metals) - 0.4)
    ax_bar.set_ylim(0, ymax)
    ax_bar.set_ylabel(ylabel)
    if title:
        ax_bar.set_title(title)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["bottom"].set_visible(False)
    ax_bar.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_bar.tick_params(axis="y", pad=1.5)

    # Heatmap row: one vector Rectangle per metal (avoids imshow's embedded
    # raster, which Illustrator sometimes renders as blank/wrong).
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle
    norm = Normalize(vmin=0, vmax=data_max)
    ax_hm.set_xlim(-0.5, len(metals) - 0.5)
    ax_hm.set_ylim(0, 1)
    for xi, mean in zip(x, means):
        ax_hm.add_patch(Rectangle(
            (xi - 0.5, 0), 1, 1,
            facecolor=_ND_PURPLE_CMAP(norm(mean)),
            edgecolor="none", linewidth=0, zorder=1))
    # Frame the whole heatmap row with a single thin outline.
    ax_hm.add_patch(Rectangle(
        (-0.5, 0), len(metals), 1,
        facecolor="none", edgecolor="0.15", linewidth=0.5, zorder=2))
    for spine in ax_hm.spines.values():
        spine.set_visible(False)
    ax_hm.set_yticks([])
    ax_hm.set_xticks(x)
    ax_hm.set_xticklabels(metals, rotation=90 if len(metals) > 12 else 0)
    ax_hm.tick_params(axis="x", pad=1.5, length=0)

    sm = ScalarMappable(norm=norm, cmap=_ND_PURPLE_CMAP)
    cb = fig.colorbar(sm, cax=ax_cb)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=6, width=0.5, length=2, pad=1.5)
    cb.set_label("%", fontsize=6, labelpad=2)

    fig.tight_layout(pad=0.2)

    # Shrink the colorbar's physical extent to (data_max / cmax) of the full
    # bar+heatmap height, anchored to the bottom.
    fig.canvas.draw()
    full_bbox = ax_cb.get_position()
    frac = min(1.0, data_max / cmax) if cmax > 0 else 1.0
    ax_cb.set_position([
        full_bbox.x0, full_bbox.y0,
        full_bbox.width, full_bbox.height * frac,
    ])

    for xi, m, mean in zip(x, metals, means):
        if m in not_spiked and mean > 0:
            _draw_diagonal_stripes(ax_bar, xi - bar_w / 2, 0.0, bar_w, mean,
                                   spacing_pt=2.5, linewidth=0.5, color="0.15")
    return fig, ax_bar


if __name__ == "__main__":
    main()
