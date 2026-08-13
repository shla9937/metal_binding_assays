#!/usr/bin/env python3
"""Bar chart of % of total metal from an ICP-MS selectivity assay.

Input format: wide table with one column per metal (no `Protein` column
required) and one row per replicate. Each row is normalized to its own total
(negative values clipped to zero) before averaging across replicates, so the
plotted bars are the mean ± SEM of the per-replicate metal fraction.

Metals that are detected but were not spiked into the assay (default: Fe, Zn)
are drawn with the same fill color but overlaid with diagonal hatching to
match the striped bars in the source figure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Order and fill colors matched to the reference 6-metal competition panel.
METAL_ORDER: tuple[str, ...] = ("Mn", "Co", "Ni", "Cu", "Nd", "Dy", "Fe", "Zn")
METAL_COLORS: dict[str, str] = {
    "Mn": "#FF1493",
    "Co": "#8B008B",
    "Ni": "#00C853",
    "Cu": "#1E90FF",
    "Nd": "#8A2BE2",
    "Dy": "#FFD700",
    "Fe": "#ffffff",  # not in DSF palette; hatched (not spiked, detected)
    "Zn": "#C0C0C0",  # hatched (not spiked, detected)
}
# Metals shown with diagonal hatching (detected but not added to the mix).
NOT_SPIKED: frozenset[str] = frozenset({"Fe", "Zn"})

# 45 mm square panel per the request.
PANEL_MM = 45.0
PANEL_IN = PANEL_MM / 25.4


def configure_nature_style() -> None:
    """Editable-text vector output, sans-serif, Nature-compliant sizing."""
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 6,
        "axes.labelsize": 6,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "hatch.linewidth": 0.5,
        "hatch.color": "0.15",
        "savefig.transparent": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def main() -> None:
    args = parse_args()
    configure_nature_style()
    df = load_table(args.input)
    metals = list(args.metals) if args.metals else [m for m in METAL_ORDER if m in df.columns]
    missing = [m for m in metals if m not in df.columns]
    if missing:
        raise ValueError(f"Input is missing metal columns: {missing}")
    stats = summarize(df, metals)
    fig, _ = plot(stats, metals, title=args.title, ylabel=args.ylabel, ymax=args.ymax)
    save(fig, args.output, args.formats)
    if args.show or not args.output:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", type=Path, required=True,
                   help="Path to input .xlsx or .csv (one row per replicate)")
    p.add_argument("-o", "--output", type=Path,
                   default=Path(__file__).resolve().parent.parent / "outputs" / "lanm_icp_percent_metal",
                   help="Output path WITHOUT extension. Extensions from --formats are appended.")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"],
                   help="One or more output formats (default: pdf png). PDF keeps text editable in Illustrator (pdf.fonttype: 42).")
    p.add_argument("--metals", nargs="+", default=None,
                   help="Override metal columns and left-to-right order")
    p.add_argument("--title", default=None, help="Optional plot title")
    p.add_argument("--ylabel", default="% of bound metal",
                   help="Y-axis label (default: '% of bound metal')")
    p.add_argument("--ymax", type=float, default=100.0,
                   help="Fixed y-axis upper limit (default: 100)")
    p.add_argument("--show", action="store_true", help="Show the plot window")
    return p.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    return df


def summarize(df: pd.DataFrame, metals: list[str]) -> pd.DataFrame:
    """Per-replicate metal fractions, then mean and SEM per metal (percent)."""
    vals = df[metals].apply(pd.to_numeric, errors="coerce").clip(lower=0)
    totals = vals.sum(axis=1)
    # Drop replicates whose total is zero to avoid divide-by-zero.
    keep = totals > 0
    frac = vals.loc[keep].div(totals.loc[keep], axis=0) * 100.0
    n = frac.shape[0]
    mean = frac.mean(axis=0)
    sem = frac.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else pd.Series(0.0, index=metals)
    return pd.DataFrame({"metal": metals, "mean": mean.values, "sem": sem.values})


def _draw_diagonal_stripes(ax, x0: float, y0: float, w: float, h: float,
                           spacing_pt: float, linewidth: float, color: str) -> None:
    """Draw 45° stripes (visual) inside the data-coord rect (x0, y0, w, h).

    Each stripe is analytically clipped to the rectangle before drawing, so
    the SVG contains plain Line2D segments — no <pattern> element, which
    Illustrator renders as blank.
    """
    from matplotlib.lines import Line2D

    fig = ax.figure
    bbox = ax.get_position()
    ax_w_in = bbox.width * fig.get_figwidth()
    ax_h_in = bbox.height * fig.get_figheight()
    xr = ax.get_xlim()[1] - ax.get_xlim()[0]
    yr = ax.get_ylim()[1] - ax.get_ylim()[0]
    # Data slope that appears as 45° visually.
    slope = (yr / ax_h_in) / (xr / ax_w_in)
    # Spacing in data-x units: 1 pt = 1/72 inch. Perpendicular spacing along
    # the stripe normal in inches -> along +x in data units.
    spacing_x = (spacing_pt / 72.0) * (xr / ax_w_in) * np.sqrt(2)

    # Stripe origin range along the bottom edge; extend left so stripes cover
    # the whole rectangle (including entering through the left edge).
    x_left = x0 - h / slope
    x_right = x0 + w
    n = int(np.ceil((x_right - x_left) / spacing_x)) + 2
    for i in range(n + 1):
        xs = x_left + i * spacing_x
        # Full-length ray: y = y0 + (x - xs)*slope, starting at y=y0.
        # Intersect with rectangle edges:
        x_enter = max(x0, xs)
        y_enter = y0 + (x_enter - xs) * slope
        if y_enter > y0 + h:
            continue  # ray enters above top of rect
        x_exit_right = x0 + w
        y_exit_right = y0 + (x_exit_right - xs) * slope
        if y_exit_right <= y0 + h:
            x_exit, y_exit = x_exit_right, y_exit_right
        else:
            y_exit = y0 + h
            x_exit = xs + h / slope
        if x_exit <= x_enter:
            continue
        line = Line2D([x_enter, x_exit], [y_enter, y_exit],
                      transform=ax.transData,
                      color=color, linewidth=linewidth,
                      solid_capstyle="butt", zorder=2.5)
        ax.add_line(line)


def plot(stats: pd.DataFrame, metals: list[str], title: str | None,
         ylabel: str, ymax: float | None) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(PANEL_IN, PANEL_IN))
    x = np.arange(len(metals))
    means = stats["mean"].to_numpy()
    sems = stats["sem"].to_numpy()

    for xi, m, mean, sem in zip(x, metals, means, sems):
        color = METAL_COLORS.get(m, "#888888")
        bar_w = 0.75
        ax.bar(xi, mean, width=bar_w, color=color, edgecolor="0.15",
               linewidth=0.5, zorder=2)
        if sem > 0:
            ax.errorbar(xi, mean, yerr=sem, fmt="none", ecolor="0.15",
                        elinewidth=0.5, capsize=1.2, capthick=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(metals)
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

    # Draw hatch stripes AFTER layout so display-coord transform is final.
    fig.canvas.draw()
    bar_w = 0.75
    for xi, m, mean in zip(x, metals, means):
        if m in NOT_SPIKED and mean > 0:
            _draw_diagonal_stripes(ax, xi - bar_w / 2, 0.0, bar_w, mean,
                                   spacing_pt=2.5, linewidth=0.5, color="0.15")
    return fig, ax


def save(fig: plt.Figure, base: Path | None, formats: list[str]) -> None:
    if base is None:
        return
    base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = base.with_suffix(f".{fmt.lstrip('.')}")
        fig.savefig(out, dpi=600, bbox_inches="tight")
        if fmt.lower() == "svg":
            _fix_svg_for_illustrator(out)
        print(f"Saved figure to {out}")


def _fix_svg_for_illustrator(path: Path) -> None:
    """Rewrite the root <svg> width/height from pt to px.

    Illustrator's SVG importer sometimes renders matplotlib SVGs as blank
    when width/height are in pt. Using px (with the same numeric value, which
    matches matplotlib's 1pt = 1 user unit convention in the viewBox) makes
    Illustrator open the file at ~1.33× physical size but with the correct
    proportions, and avoids the blank-import bug.
    """
    import re
    text = path.read_text(encoding="utf-8")
    text = re.sub(r'(<svg\b[^>]*?)\bwidth="([\d.]+)pt"', r'\1width="\2px"', text, count=1)
    text = re.sub(r'(<svg\b[^>]*?)\bheight="([\d.]+)pt"', r'\1height="\2px"', text, count=1)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
