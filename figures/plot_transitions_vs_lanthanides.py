#!/usr/bin/env python3
"""Plot summed transition-metal vs summed lanthanide ICP-MS signal per protein.

Input file: wide table with columns [Protein, Mn, Co, Ni, Cu, Zn, La, Nd, Eu, Dy, Tm]
and three replicate rows per protein.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

TRANSITIONS = ["Mn", "Co", "Ni", "Cu", "Zn"]
LANTHANIDES = ["La", "Nd", "Eu", "Dy", "Tm"]
ALL_METALS = [*TRANSITIONS, *LANTHANIDES]  # already atomic-number ordered

# Nature: single-column panel = 89 mm (~3.5 in). One panel of a 4-panel figure.
PANEL_IN = 89.0 / 25.4

# Nd-purple: white -> #8A2BE2 (matches the other ICP-MS / DSF heatmaps in this figure set).
_ND_PURPLE_CMAP = LinearSegmentedColormap.from_list(
    "nd_purple", ["#ffffff", "#8A2BE2"]
)
try:
    mpl.colormaps.register(_ND_PURPLE_CMAP, name="nd_purple")
except ValueError:
    pass  # already registered on re-import


def configure_nature_style() -> None:
    """Editable-text vector output, sans-serif, Nature-compliant sizing."""
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 6,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 0.75,
        "lines.markersize": 4})


def main() -> None:
    args = parse_args()
    configure_nature_style()
    df = load_table(args.input)
    stats = summarize(df)
    fig, ax = plot(stats, threshold=args.threshold)
    if args.output:
        fig.savefig(args.output, dpi=600, bbox_inches="tight")
        print(f"Saved figure to {args.output}")

    per_metal = summarize_per_metal(df)
    fig_hm, _ = plot_heatmap(per_metal)
    hm_out = args.heatmap_output or _derive_heatmap_path(args.output)
    if hm_out:
        fig_hm.savefig(hm_out, dpi=600, bbox_inches="tight")
        print(f"Saved heatmap to {hm_out}")

    if args.show or not args.output:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input",type=Path,help="Path to input .xlsx or .csv (e.g. ../source/ped_cluster_icp_ms.xlsx)",)
    p.add_argument("-o", "--output", type=Path, default=None,help="Path to save the scatter figure. Use .pdf or .svg for editable-text vector output.",)
    p.add_argument("--heatmap-output", type=Path, default=None,
                   help="Path to save the per-metal heatmap. Defaults to <output stem>_heatmap<ext> when --output is set.")
    p.add_argument("-t", "--threshold", type=float, default=0.83, help="Fractional lanthanide-selectivity threshold for a hit (default 0.83)",)
    p.add_argument("--show", action="store_true", help="Show the plot window")
    return p.parse_args()


def _derive_heatmap_path(output: Path | None) -> Path | None:
    if output is None:
        return None
    return output.with_name(f"{output.stem}_heatmap{output.suffix}")


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    missing = [c for c in ["Protein", *TRANSITIONS, *LANTHANIDES] if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-replicate transition/lanthanide sums, then per-protein mean and SD."""
    per_rep = pd.DataFrame({
        "Protein": df["Protein"],
        "transitions": df[TRANSITIONS].clip(lower=0).sum(axis=1),
        "lanthanides": df[LANTHANIDES].clip(lower=0).sum(axis=1),
    })
    grouped = per_rep.groupby("Protein", sort=False)
    stats = grouped.agg(["mean", "std"])
    stats.columns = ["_".join(c) for c in stats.columns]
    return stats.reset_index()


def plot(stats: pd.DataFrame, threshold: float) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(PANEL_IN, PANEL_IN))

    _, caps, _ = ax.errorbar(
        stats["transitions_mean"], stats["lanthanides_mean"],
        xerr=stats["transitions_std"], yerr=stats["lanthanides_std"],
        fmt="o", color="#1f77b4", ecolor="0.3",
        capsize=1.5, elinewidth=0.5, capthick=0.5,
        markersize=4, markeredgewidth=0, zorder=3,)
    for cap in caps:
        cap.set_markeredgewidth(0.5)

    for _, row in stats.iterrows():
        ax.annotate(
            row["Protein"],
            xy=(row["transitions_mean"], row["lanthanides_mean"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=6, fontstyle="italic",)

    xmax = float(np.nanmax(stats["transitions_mean"] + stats["transitions_std"].fillna(0)))
    ymax = float(np.nanmax(stats["lanthanides_mean"] + stats["lanthanides_std"].fillna(0)))
    upper = 16

    # Line where lanthanide fraction = threshold: y/(x+y) = t  =>  y = t/(1-t) * x
    slope = threshold / (1.0 - threshold)
    xs = np.array([0.0, upper])
    ax.plot(xs, slope * xs, linestyle="--", color="seagreen", linewidth=0.75,
        label=f"{threshold * 100:.0f}% lanthanide selectivity", zorder=2,)
    ax.fill_between(xs, slope * xs, upper, color="seagreen", alpha=0.08, linewidth=0, zorder=1)

    ax.set_xlim(0, 4)
    ax.set_xticks([0,1,2,3,4])
    ax.set_ylim(0, upper)
    ax.set_xlabel("Σ transition metals (µM)")
    ax.set_ylabel("Σ lanthanides (µM)")
    ax.legend(loc="lower right", frameon=False, handlelength=1.5, borderaxespad=0.2, handletextpad=0.4)
    ax.set_title("Competetative Lanthanide selectivity of Ped cluster (ICP-MS)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.2)
    return fig, ax


def summarize_per_metal(df: pd.DataFrame) -> pd.DataFrame:
    """Per-protein mean concentration per metal (rows=Protein, cols=ALL_METALS)."""
    vals = df[["Protein", *ALL_METALS]].copy()
    vals[ALL_METALS] = vals[ALL_METALS].clip(lower=0)
    mean = vals.groupby("Protein", sort=False)[ALL_METALS].mean()
    return mean


def plot_heatmap(mean: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    """Heatmap of per-protein mean [metal] (µM). White -> Nd-purple, linear scale."""
    proteins = list(mean.index)
    metals = list(mean.columns)
    data = mean.to_numpy(dtype=float)
    n_rows, n_cols = data.shape

    vmax = float(np.nanmax(data)) if np.isfinite(data).any() else 1.0
    vmax = vmax if vmax > 0 else 1.0
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = mpl.colormaps["nd_purple"]

    cell_in = 0.16
    label_pad_in = 0.55
    top_pad_in = 0.42  # metal labels on top + title
    right_pad_in = 0.55  # colorbar
    bottom_pad_in = 0.10
    fig_w_in = label_pad_in + cell_in * n_cols + right_pad_in
    fig_h_in = top_pad_in + cell_in * n_rows + bottom_pad_in

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    ax = fig.add_axes([
        label_pad_in / fig_w_in,
        bottom_pad_in / fig_h_in,
        (cell_in * n_cols) / fig_w_in,
        (cell_in * n_rows) / fig_h_in,
    ])

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_aspect("equal")

    side = 0.9
    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if not np.isfinite(v):
                continue
            ax.add_patch(mpl.patches.Rectangle(
                (j - side / 2, i - side / 2), side, side,
                facecolor=cmap(norm(v)), edgecolor="none",
            ))

    # Divider between transition metals and lanthanides.
    split = len(TRANSITIONS) - 0.5
    ax.axvline(split, color="0.35", linewidth=0.5, linestyle="--", zorder=4)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(metals)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(proteins, fontstyle="italic")
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="0.85", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, aspect=14, pad=0.02, fraction=0.05)
    cbar.set_label("[metal] (µM)")
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(width=0.5, length=2.5)

    fig.suptitle("Per-metal ICP-MS signal, Ped cluster",
                 fontsize=8, y=1.0 - 0.02 / fig_h_in)
    return fig, ax


if __name__ == "__main__":
    main()
