#!/usr/bin/env python3
"""Bar chart of 1/EC50 for a 6-metal DSF assay (parallel to plot_percent_metal_bar.py).

Input format: single-row wide table with one column per metal holding the
EC50 (in M by default) and a matching `<Metal>_err` column for the SEM.
Non-binders are encoded as NaN and rendered as an `NB` label.

Y-axis is 1/EC50 on a log scale, with tick labels showing the equivalent
EC50 as decades in M (default range 1 nM – 1 mM). Bars grow upward from
the bottom, so a taller bar means tighter binding.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_percent_metal_bar import (  # noqa: E402
    METAL_COLORS,
    METAL_ORDER,
    NOT_SPIKED,
    _draw_diagonal_stripes,
    _fix_svg_for_illustrator,
    configure_nature_style,
)

UNIT_SCALE = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "µM": 1e-6, "nM": 1e-9, "pM": 1e-12}

# 45 mm square panel to match plot_percent_metal_bar.py.
PANEL_MM = 45.0
PANEL_IN = PANEL_MM / 25.4

_OX_RE = re.compile(r"\s*\d*[+−\-]$")


def _strip_oxidation(name: str) -> str:
    return _OX_RE.sub("", str(name)).strip()


def main() -> None:
    args = parse_args()
    configure_nature_style()
    ec50, err = load_table(args.input, args.input_unit)
    metals = list(args.metals) if args.metals else [m for m in METAL_ORDER if m in ec50.index]
    missing = [m for m in metals if m not in ec50.index]
    if missing:
        raise ValueError(f"Input is missing metal columns: {missing}")
    fig, _ = plot(ec50.loc[metals], err.loc[metals], metals,
                  title=args.title, kd_min_m=args.kd_min, kd_max_m=args.kd_max)
    save(fig, args.output, args.formats)
    if args.show or not args.output:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", type=Path, required=True,
                   help="Path to input .xlsx or .csv (single-row 6-metal EC50 table)")
    p.add_argument("-o", "--output", type=Path,
                   default=Path(__file__).resolve().parent.parent
                   / "outputs" / "lanm_dsf_6metal_ec50",
                   help="Output path WITHOUT extension.")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"],
                   help="One or more output formats (default: pdf png).")
    p.add_argument("--input-unit", default="M", choices=list(UNIT_SCALE),
                   help="Unit of EC50 values in the file (default: M)")
    p.add_argument("--metals", nargs="+", default=None,
                   help="Override metal order.")
    p.add_argument("--kd-min", type=float, default=1e-9,
                   help="Bottom of the Kd axis in M (default 1e-9 = 1 nM).")
    p.add_argument("--kd-max", type=float, default=1e-3,
                   help="Top of the Kd axis in M (default 1e-3 = 1 mM).")
    p.add_argument("--title", default=None, help="Optional plot title")
    p.add_argument("--show", action="store_true", help="Show plot window")
    return p.parse_args()


def load_table(path: Path, input_unit: str) -> tuple[pd.Series, pd.Series]:
    """Return (ec50, err) Series indexed by plain metal symbol, both in M."""
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    if df.empty:
        raise ValueError("Input has no data rows")
    row = df.iloc[0]

    # Accept `_err` or `_error` suffix for the SEM column.
    means: dict[str, float] = {}
    errs: dict[str, float] = {}
    for c in df.columns:
        s = str(c)
        if s.endswith("_err") or s.endswith("_error"):
            continue
        m = _strip_oxidation(s)
        if m.lower() == "protein":
            continue
        err_col = None
        for suffix in ("_err", "_error"):
            if f"{s}{suffix}" in df.columns:
                err_col = f"{s}{suffix}"
                break
        means[m] = pd.to_numeric(row.get(s), errors="coerce")
        errs[m] = pd.to_numeric(row.get(err_col), errors="coerce") if err_col else np.nan

    scale = UNIT_SCALE[input_unit]
    ec50 = pd.Series(means, dtype=float) * scale
    err = pd.Series(errs, dtype=float) * scale
    return ec50, err


def plot(ec50: pd.Series, err: pd.Series, metals: list[str], title: str | None,
         kd_min_m: float, kd_max_m: float) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(PANEL_IN, PANEL_IN))
    x = np.arange(len(metals))
    bar_w = 0.75

    # Internal y-units: 1/M (higher on axis = tighter binding).
    y_bot = 1.0 / kd_max_m   # small y  = weakest binding
    y_top = 1.0 / kd_min_m   # large y  = tightest binding
    ax.set_yscale("log")
    ax.set_ylim(y_bot, y_top)
    ax.set_xlim(-0.6, len(metals) - 0.4)

    hatched_and_visible: list[tuple[float, str, float]] = []
    for xi, m in zip(x, metals):
        v = float(ec50.get(m, np.nan))
        e = float(err.get(m, np.nan))
        color = METAL_COLORS.get(m, "#888888")
        if not np.isfinite(v) or v <= 0:
            ax.text(xi, y_bot * (y_top / y_bot) ** 0.02, "NB",
                    ha="center", va="bottom", fontsize=6, fontweight="bold")
            continue
        inv = 1.0 / v
        ax.bar(xi, inv - y_bot, width=bar_w, bottom=y_bot,
               color=color, edgecolor="0.15", linewidth=0.5, zorder=2)
        if np.isfinite(e) and e > 0:
            lo = 1.0 / max(v + e, v * 1e-6)
            hi = 1.0 / max(v - e, v * 1e-3)
            ax.errorbar(xi, inv, yerr=[[inv - lo], [hi - inv]], fmt="none",
                        ecolor="0.15", elinewidth=0.5, capsize=1.2, capthick=0.5,
                        zorder=3)
        if m in NOT_SPIKED:
            hatched_and_visible.append((xi, m, inv))

    ax.set_xticks(x)
    ax.set_xticklabels(metals)

    # Decade ticks labeled as EC50 in M (10^log_kd), position = 1/Kd.
    log_lo = int(np.ceil(np.log10(kd_min_m)))
    log_hi = int(np.floor(np.log10(kd_max_m)))
    decades = list(range(log_lo, log_hi + 1))
    positions = [10.0 ** (-d) for d in decades]
    labels = [f"$10^{{{d}}}$" for d in decades]
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())

    ax.set_ylabel("EC$_{50}$ (M)")
    if title:
        ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", pad=1.5)
    ax.tick_params(axis="y", pad=1.5)
    fig.tight_layout(pad=0.2)

    # Hatch stripes on a log axis: draw in log-space so 45° stays visual.
    fig.canvas.draw()
    for xi, m, inv in hatched_and_visible:
        # Convert the bar's log-space height into linear data span for the
        # stripe helper (which assumes a linear axis) by temporarily working
        # in log10 units.
        _hatch_log_bar(ax, xi - bar_w / 2, y_bot, bar_w, inv)
    return fig, ax


def _hatch_log_bar(ax: plt.Axes, x0: float, y_bot: float, w: float, top: float) -> None:
    """Overlay diagonal stripes on a bar drawn on a log-y axis."""
    # Save + swap to a linear-y helper axes bound to the same data window.
    from matplotlib.lines import Line2D

    # Direct data-coord stripe drawing tailored to log-y: compute slope in
    # display space, then place lines as (x, y) pairs already in log-scaled y.
    bbox = ax.get_position()
    fig = ax.figure
    ax_w_in = bbox.width * fig.get_figwidth()
    ax_h_in = bbox.height * fig.get_figheight()
    y0_log = np.log10(y_bot)
    y1_log = np.log10(top)
    dy_log = y1_log - y0_log
    if dy_log <= 0 or w <= 0:
        return
    xr = ax.get_xlim()[1] - ax.get_xlim()[0]
    # Effective slope in log-y space so 45° looks 45°.
    slope = (dy_log / ax_h_in) / (xr / ax_w_in)
    spacing_x = (2.5 / 72.0) * (xr / ax_w_in) * np.sqrt(2)
    x_left = x0 - dy_log / slope
    x_right = x0 + w
    n = int(np.ceil((x_right - x_left) / spacing_x)) + 2
    for i in range(n + 1):
        xs = x_left + i * spacing_x
        x_enter = max(x0, xs)
        y_enter_log = y0_log + (x_enter - xs) * slope
        if y_enter_log > y1_log:
            continue
        x_exit_right = x0 + w
        y_exit_right_log = y0_log + (x_exit_right - xs) * slope
        if y_exit_right_log <= y1_log:
            x_exit, y_exit_log = x_exit_right, y_exit_right_log
        else:
            y_exit_log = y1_log
            x_exit = xs + dy_log / slope
        if x_exit <= x_enter:
            continue
        line = Line2D([x_enter, x_exit], [10 ** y_enter_log, 10 ** y_exit_log],
                      transform=ax.transData, color="0.15", linewidth=0.5,
                      solid_capstyle="butt", zorder=2.5)
        ax.add_line(line)


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


if __name__ == "__main__":
    main()
