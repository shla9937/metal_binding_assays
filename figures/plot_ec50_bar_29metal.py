#!/usr/bin/env python3
"""Bar+heatmap chart of DSF EC50 across many metals (parallel to plot_percent_metal_bar_29metal.py --heatmap).

Input format: same as plot_ec50_heatmap.py — a wide table with a `Protein`
column, one column per metal (names may include oxidation-state suffixes
like `Ce3+`), and matching `<Metal>_error` columns. EDTA columns are
skipped, metals are sorted left-to-right by atomic number.

Y-axis is 1/EC50 on a log scale (decade ticks labeled as EC50 in M so
tighter binding is higher up). The heatmap row below is colored by the
same LogNorm(vmin, vmax) mapping used in plot_ec50_heatmap.py (dark
purple = tight, white = weak), and the colorbar shows the two extra
decades outside [vmin, vmax] as saturated dark/white extensions.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_ec50_heatmap import (  # noqa: E402
    ATOMIC_NUMBER,
    SKIP_COLUMNS,
    UNIT_SCALE,
    _extend_cmap,
    _strip_oxidation,
    configure_nature_style,
    extract_pairs,
    load_table,
)
from plot_percent_metal_bar import save  # noqa: E402
from plot_percent_metal_bar_29metal import METAL_COLORS  # noqa: E402

# Nd-purple: dark #8A2BE2 at low EC50 (tight binding) -> white at high EC50.
_ND_PURPLE_CMAP = LinearSegmentedColormap.from_list(
    "nd_purple_r", ["#8A2BE2", "#ffffff"]
)

# 45 mm tall panel to match plot_percent_metal_bar_29metal.py.
PANEL_HEIGHT_MM = 45.0
PANEL_HEIGHT_IN = PANEL_HEIGHT_MM / 25.4


def main() -> None:
    args = parse_args()
    configure_nature_style()
    df = load_table(args.input)
    mean_df, err_df = extract_pairs(df, args.metals, args.input_unit)
    # Use the first protein row (parallel single-protein bar chart).
    protein = args.protein or str(mean_df.index[0])
    if protein not in mean_df.index:
        raise ValueError(f"Protein '{protein}' not in file. Available: {list(mean_df.index)}")
    ec50_m = mean_df.loc[protein]
    err_m = err_df.loc[protein]
    metals = list(ec50_m.index)

    fig, _ = plot(ec50_m, err_m, metals, title=args.title or protein,
                  vmin_m=args.vmin, vmax_m=args.vmax,
                  cbar_pad_decades=args.cbar_pad_decades,
                  panel_width_mm=args.panel_mm)
    save(fig, args.output, args.formats)
    if args.show or not args.output:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", type=Path, required=True,
                   help="Path to input .xlsx or .csv (Protein + metal + metal_error columns).")
    p.add_argument("-o", "--output", type=Path,
                   default=Path(__file__).resolve().parent.parent
                   / "outputs" / "lanm_dsf_29metal_ec50",
                   help="Output path WITHOUT extension.")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"],
                   help="Output formats (default: pdf png).")
    p.add_argument("--protein", default=None,
                   help="Which protein row to plot (default: first row).")
    p.add_argument("--metals", nargs="+", default=None,
                   help="Override metal columns and left-to-right order.")
    p.add_argument("--input-unit", default="µM", choices=list(UNIT_SCALE),
                   help="Unit of EC50 values in the file (default: µM).")
    p.add_argument("--vmin", type=float, default=1e-6,
                   help="Bottom of the color scale in M (default 1e-6 = 1 µM).")
    p.add_argument("--vmax", type=float, default=1e-3,
                   help="Top of the color scale in M (default 1e-3 = 1 mM).")
    p.add_argument("--cbar-pad-decades", type=int, default=2,
                   help="Number of decades to extend the colorbar past --vmin/--vmax "
                        "as saturated dark/white regions (default: 2).")
    p.add_argument("--panel-mm", type=float, default=90.0,
                   help="Figure width in mm (default: 90; height is 45 mm).")
    p.add_argument("--title", default=None, help="Optional plot title.")
    p.add_argument("--show", action="store_true", help="Show plot window.")
    return p.parse_args()


def plot(ec50: pd.Series, err: pd.Series, metals: list[str], title: str | None,
         vmin_m: float, vmax_m: float, cbar_pad_decades: int,
         panel_width_mm: float) -> tuple[plt.Figure, plt.Axes]:
    """Bars of 1/EC50 (colored by metal) above a Nd-purple heatmap row.

    Bars use the DSF metal palette so identity is obvious. The heatmap and
    colorbar are scaled to just this dataset's EC50 range (per-panel), and
    the colorbar's physical height reflects the fraction of the shared
    reference log range [vmin_m, vmax_m] this dataset actually spans.
    """
    ec50_arr = ec50.to_numpy(dtype=float)

    # Per-dataset log-EC50 range for the heatmap and colorbar. Expanded to
    # whole decades so the colorbar always contains at least one labeled tick
    # that lines up with a bar-axis decade.
    finite = ec50_arr[np.isfinite(ec50_arr) & (ec50_arr > 0)]
    if finite.size == 0:
        data_lo = vmin_m
        data_hi = vmax_m
    else:
        data_lo = 10.0 ** np.floor(np.log10(float(np.min(finite))))
        data_hi = 10.0 ** np.ceil(np.log10(float(np.max(finite))))
        if data_lo == data_hi:
            data_hi = data_lo * 10.0

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
    # Bar y-axis spans the shared reference range so multiple panels align.
    y_bot = 1.0 / vmax_m
    y_top = 1.0 / vmin_m

    for xi, m in zip(x, metals):
        v = float(ec50.get(m, np.nan))
        e = float(err.get(m, np.nan))
        if not np.isfinite(v) or v <= 0:
            continue
        inv = 1.0 / v
        color = METAL_COLORS.get(_strip_oxidation(m), "#888888")
        ax_bar.bar(xi, inv - y_bot, width=bar_w, bottom=y_bot,
                   color=color, edgecolor="0.15", linewidth=0.5, zorder=2)
        if np.isfinite(e) and e > 0:
            lo = 1.0 / max(v + e, v * 1e-6)
            hi = 1.0 / max(v - e, v * 1e-3)
            ax_bar.errorbar(xi, inv, yerr=[[inv - lo], [hi - inv]], fmt="none",
                            ecolor="0.15", elinewidth=0.5, capsize=1.2, capthick=0.5,
                            zorder=3)

    ax_bar.set_yscale("log")
    ax_bar.set_xlim(-0.6, len(metals) - 0.4)
    ax_bar.set_ylim(y_bot, y_top)
    ax_bar.set_ylabel("EC$_{50}$ (M)")
    if title:
        ax_bar.set_title(title)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["bottom"].set_visible(False)
    ax_bar.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_bar.tick_params(axis="y", pad=1.5)

    # Decade y-ticks labeled as EC50 (M): position = 1/EC50, label = 10^log(EC50).
    log_lo = int(np.ceil(np.log10(vmin_m)))
    log_hi = int(np.floor(np.log10(vmax_m)))
    decades = list(range(log_lo, log_hi + 1))
    positions = [10.0 ** (-d) for d in decades]
    labels = [f"$10^{{{d}}}$" for d in decades]
    ax_bar.set_yticks(positions)
    ax_bar.set_yticklabels(labels)
    ax_bar.yaxis.set_minor_locator(mpl.ticker.NullLocator())

    # Heatmap row scaled to this dataset's EC50 range.
    hm_norm = LogNorm(vmin=data_lo, vmax=data_hi, clip=True)
    ax_hm.set_xlim(-0.5, len(metals) - 0.5)
    ax_hm.set_ylim(0, 1)
    for xi, v in zip(x, ec50_arr):
        if not np.isfinite(v) or v <= 0:
            face = "#ffffff"
        else:
            face = _ND_PURPLE_CMAP(hm_norm(v))
        ax_hm.add_patch(Rectangle(
            (xi - 0.5, 0), 1, 1,
            facecolor=face, edgecolor="none", linewidth=0, zorder=1))
    ax_hm.add_patch(Rectangle(
        (-0.5, 0), len(metals), 1,
        facecolor="none", edgecolor="0.15", linewidth=0.5, zorder=2))
    for spine in ax_hm.spines.values():
        spine.set_visible(False)
    ax_hm.set_yticks([])
    ax_hm.set_xticks(x)
    ax_hm.set_xticklabels(metals, rotation=90 if len(metals) > 12 else 0)
    ax_hm.tick_params(axis="x", pad=1.5, length=0)

    # Colorbar uses the full reference log range so decade ticks land at the
    # same positions as the bar axis; only the physical extent is trimmed to
    # the data span, so ticks line up with the bar axis's own decade ticks.
    sm = ScalarMappable(cmap=_ND_PURPLE_CMAP,
                        norm=LogNorm(vmin=vmin_m, vmax=vmax_m, clip=True))
    cb = fig.colorbar(sm, cax=ax_cb)
    cb.outline.set_linewidth(0.5)
    cb.set_label("EC$_{50}$ (M)", fontsize=6, labelpad=2)
    cb.ax.set_yscale("log")
    cb.ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0, subs=(1.0,), numticks=20))
    cb.ax.yaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10.0, labelOnlyBase=True))
    cb.ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    cb.ax.tick_params(labelsize=6, width=0.5, length=2, pad=1.5)
    # Invert so tightest EC50 (dark purple) is at the top — matches the bar
    # axis where taller = tighter binding; then clamp visible range to the
    # data span so the colorbar height reflects this dataset.
    cb.ax.set_ylim(data_hi, data_lo)

    fig.tight_layout(pad=0.2)

    # Position colorbar to occupy exactly the log-y span [1/data_hi, 1/data_lo]
    # of the bar axis, so its ticks physically line up with the bar values.
    fig.canvas.draw()
    bar_bbox = ax_bar.get_position()
    ref_span = np.log10(vmax_m) - np.log10(vmin_m)
    frac_lo = (np.log10(vmax_m) - np.log10(data_hi)) / ref_span
    frac_hi = (np.log10(vmax_m) - np.log10(data_lo)) / ref_span
    frac_lo = float(np.clip(frac_lo, 0.0, 1.0))
    frac_hi = float(np.clip(frac_hi, 0.0, 1.0))
    cb_bbox = ax_cb.get_position()
    y_new = bar_bbox.y0 + bar_bbox.height * frac_lo
    h_new = bar_bbox.height * (frac_hi - frac_lo)
    ax_cb.set_position([cb_bbox.x0, y_new, cb_bbox.width, h_new])

    return fig, ax_bar


if __name__ == "__main__":
    main()
