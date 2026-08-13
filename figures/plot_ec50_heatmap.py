#!/usr/bin/env python3
"""Heatmap of EC50 (or Kd) values: proteins (rows) x metals (columns).

Input format: wide table with a `Protein` column, then all metal mean columns
(names may include oxidation-state suffixes like `Ca2+` or `La3+`), followed by
matching `<Metal>_error` columns. Metals are auto-discovered, EDTA is skipped,
and columns are sorted left-to-right by increasing atomic number.

Blank / NaN mean values are treated as non-binding and rendered as white cells.
Color scale is log10(EC50); tighter binding = darker.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

# Nd-purple: dark #8A2BE2 at low EC50 (tight binding) -> white at high EC50.
_ND_PURPLE_CMAP = LinearSegmentedColormap.from_list(
    "nd_purple_r", ["#8A2BE2", "#ffffff"]
)
try:
    mpl.colormaps.register(_ND_PURPLE_CMAP, name="nd_purple_r")
except ValueError:
    pass  # already registered on re-import

# Atomic numbers for elements 1-92; used to sort metal columns left-to-right.
ATOMIC_NUMBER: dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34,
    "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42,
    "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58,
    "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66,
    "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W": 74,
    "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82,
    "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
    "Pa": 91, "U": 92,
}

# Columns to always ignore (case-insensitive) even when they appear in the sheet.
SKIP_COLUMNS: frozenset[str] = frozenset({"EDTA"})

_OX_RE = re.compile(r"\s*\d*[+−\-]$")


def _strip_oxidation(name: str) -> str:
    return _OX_RE.sub("", str(name)).strip()


# Nature panel widths (Fig. Guide 2023): single column = 89 mm, 1.5-col = 120 mm,
# double column = 183 mm. Max figure height 247 mm.
PANEL_IN = 89.0 / 25.4
DOUBLE_COL_IN = 183.0 / 25.4

UNIT_SCALE = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "µM": 1e-6, "nM": 1e-9, "pM": 1e-12}

EC50_LABEL = "EC50"  # Plain text keeps the label fully editable in every viewer/font.


def configure_nature_style() -> None:
    """Editable-text vector output, sans-serif, Nature-compliant sizing."""
    mpl.rcParams.update({
        # Editable text: TrueType (Type 42) in PDF/PS; keep SVG text as text.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "pdf.compression": 6,
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 6,
        "axes.labelsize": 6,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "hatch.linewidth": 0.4,
        "hatch.color": "0.25",
        "savefig.transparent": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def main() -> None:
    args = parse_args()
    configure_nature_style()
    df = load_table(args.input)
    mean, err = extract_pairs(df, args.metals, args.input_unit)
    fig, _ = plot(
        mean, err, unit=args.display_unit, annotate=args.annotate,
        size_by_error=args.size_by_error, hatch_by_error=args.hatch_by_error,
        split_panels=args.split_panels, panel_width_mm=args.panel_width,
        cmap_name=args.cmap, title=args.title,
        vmin_display=args.vmin, vmax_display=args.vmax,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=600, bbox_inches="tight")
        print(f"Saved figure to {args.output}")
    if args.show or not args.output:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", type=Path, required=True, help="Path to input .xlsx or .csv")
    p.add_argument("-o", "--output", type=Path, default=None,help="Save path (.pdf/.svg keep editable text)")
    p.add_argument("--metals", nargs="+", default=None,help="Optional metal column names (as they appear in the file) to override auto-discovery and set left-to-right order.")
    p.add_argument("--input-unit", default="µM", choices=list(UNIT_SCALE),help="Unit of EC50 values in the input file (default: M)")
    p.add_argument("--display-unit", default="M", choices=list(UNIT_SCALE),help="Unit used for the colorbar/annotations (default: M)")
    p.add_argument("--annotate", action="store_true",help="Write mean ± err inside each cell")
    p.add_argument("--size-by-error", action="store_true",help="Shrink each cell's colored square by 1/(1+CV) so noisy fits are visibly smaller")
    p.add_argument("--hatch-by-error", action="store_true", help="Overlay diagonal hatching whose density grows with the relative error (CV) of each cell")
    p.add_argument("--split-panels", type=int, default=1,help="Split metal columns evenly across N side-by-side panels (default 2, so ~30 metals -> two column-width panels)")
    p.add_argument("--panel-width", type=float, default=178.0, metavar="MM",help="Max width per panel in mm (default 89 = Nature single column)")
    p.add_argument("--cmap", default="nd_purple_r", help="Matplotlib colormap name (default nd_purple_r: dark purple -> white, tighter binding = darker). Append _r to any built-in cmap to reverse it.")
    p.add_argument("--vmin", type=float, default=1e-6, help="Lower color bound in --display-unit (default 1e-6 = 1 µM in M). Values below clamp to the darkest color.")
    p.add_argument("--vmax", type=float, default=1e-3, help="Upper color bound in --display-unit (default 1e-3 = 1 mM in M). Values above clamp to white.")
    p.add_argument("--title", default=None,
                   help="Optional figure title placed above the heatmap (e.g. "
                        "'Binding affinity of Ped Cluster (DSF)').")
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
    if "Protein" not in df.columns:
        raise ValueError("Input is missing a `Protein` column")
    return df


def extract_pairs(
    df: pd.DataFrame, metals: list[str] | None, input_unit: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (mean, err) DataFrames indexed by Protein with one column per metal.

    Columns in `df` for means may include an oxidation-state suffix (`Ca2+`,
    `La3+`, `K+`); each is paired with `<name>_error`. When `metals` is None,
    metals are auto-discovered from the sheet and sorted by atomic number,
    with entries in `SKIP_COLUMNS` (e.g. EDTA) dropped. Column labels in the
    returned frames are the display symbols with oxidation state stripped.
    """
    all_cols = list(df.columns)
    all_cols_set = set(all_cols)

    if metals is None:
        candidates = [
            c for c in all_cols
            if c != "Protein"
            and not str(c).endswith("_error")
            and _strip_oxidation(c).upper() not in {s.upper() for s in SKIP_COLUMNS}
        ]
        # Sort by atomic number; unknown symbols land at the end alphabetically.
        metals = sorted(
            candidates,
            key=lambda c: (ATOMIC_NUMBER.get(_strip_oxidation(c), 10_000), _strip_oxidation(c)),
        )
    else:
        missing = [m for m in metals if m not in all_cols_set]
        if missing:
            raise ValueError(f"Input is missing metal columns: {missing}")

    means: dict[str, pd.Series] = {}
    errs: dict[str, pd.Series] = {}
    for m in metals:
        display = _strip_oxidation(m)
        err_col = f"{m}_error"
        means[display] = pd.to_numeric(df[m], errors="coerce")
        errs[display] = (
            pd.to_numeric(df[err_col], errors="coerce")
            if err_col in all_cols_set
            else pd.Series(np.nan, index=df.index)
        )

    scale = UNIT_SCALE[input_unit]
    mean = pd.DataFrame(means).set_index(df["Protein"]).astype(float) * scale
    err = pd.DataFrame(errs).set_index(df["Protein"]).astype(float) * scale
    # Drop rows that are entirely NaN across all requested metals.
    mean = mean.loc[~mean.isna().all(axis=1)]
    err = err.reindex(mean.index)
    return mean, err


# Diagonal hatch patterns matched to quantile bins of the observed CV
# distribution. Ordered lowest -> highest error: the lowest-error bin gets no
# hatching, then bar density grows monotonically with CV.
HATCH_PATTERNS: tuple[str, ...] = ("", "///", "//////", "//////////")


def plot(
    mean: pd.DataFrame, err: pd.DataFrame, unit: str, annotate: bool,
    size_by_error: bool = False, hatch_by_error: bool = False,
    split_panels: int = 2, panel_width_mm: float = 89.0,
    cmap_name: str = "nd_purple_r", title: str | None = None,
    vmin_display: float | None = None, vmax_display: float | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    n_rows, n_cols = mean.shape
    split_panels = max(1, min(split_panels, n_cols))
    per_panel = math.ceil(n_cols / split_panels)
    chunks: list[tuple[int, int]] = []
    for k in range(split_panels):
        lo, hi = k * per_panel, min(n_cols, (k + 1) * per_panel)
        if lo < hi:
            chunks.append((lo, hi))
    split_panels = len(chunks)
    per_max = max(hi - lo for lo, hi in chunks)

    display_scale = UNIT_SCALE[unit]
    disp = (mean / display_scale).to_numpy()
    disp_err = (err / display_scale).to_numpy()

    finite = disp[np.isfinite(disp) & (disp > 0)]
    if finite.size == 0:
        raise ValueError("No positive EC50 values to plot")
    if vmin_display is not None and vmax_display is not None:
        vmin, vmax = float(vmin_display), float(vmax_display)
    else:
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmin == vmax:
            vmin, vmax = vmin / 3.0, vmax * 3.0

    cmap = mpl.colormaps[cmap_name]
    # clip=True clamps out-of-range EC50 values to the endpoints, so anything
    # tighter than vmin renders as the darkest color and anything looser than
    # vmax renders as white.
    norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
    hatch_edges = _hatch_edges(disp, disp_err) if hatch_by_error else None

    # Sizing: pick a cell edge so each panel fits within panel_width_mm.
    panel_in = panel_width_mm / 25.4
    label_pad_in = 0.50   # room on the far left for protein tick labels
    inter_pad_in = 0.10   # gap between adjacent panels
    cbar_pad_in = 0.30    # room reserved for the (compact) shared colorbar
    cell_in = min(0.28, max(0.10, (panel_in - 0.10) / per_max))
    panel_w_in = cell_in * per_max
    fig_w_in = (
        label_pad_in
        + split_panels * panel_w_in
        + (split_panels - 1) * inter_pad_in
        + cbar_pad_in
    )
    fig_w_in = min(fig_w_in, DOUBLE_COL_IN)
    top_pad_in = 0.16 + (0.14 if title else 0.0)
    bottom_pad_in = 0.04
    fig_h_in = top_pad_in + cell_in * n_rows + bottom_pad_in

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    gs = fig.add_gridspec(
        1, split_panels,
        wspace=inter_pad_in / panel_w_in,
        left=label_pad_in / fig_w_in,
        right=1.0 - cbar_pad_in / fig_w_in,
        top=1.0 - top_pad_in / fig_h_in,
        bottom=bottom_pad_in / fig_h_in,
    )
    axes: list[plt.Axes] = [fig.add_subplot(gs[0, k]) for k in range(split_panels)]

    for k, (ax, (lo, hi)) in enumerate(zip(axes, chunks)):
        _draw_panel(
            ax, disp[:, lo:hi], disp_err[:, lo:hi],
            metal_labels=list(mean.columns[lo:hi]),
            protein_labels=list(mean.index) if k == 0 else None,
            cmap=cmap, norm=norm,
            size_by_error=size_by_error, hatch_edges=hatch_edges,
            annotate=annotate,
        )

    sm = mpl.cm.ScalarMappable(cmap=_extend_cmap(cmap, vmin, vmax,
                                                  vmin / 100.0, vmax * 100.0),
                                norm=LogNorm(vmin=vmin / 100.0,
                                             vmax=vmax * 100.0, clip=True))
    cbar = fig.colorbar(
        sm, ax=axes, shrink=0.45, aspect=14, pad=0.015, fraction=0.03,
    )
    cbar.set_label(f"{EC50_LABEL} ({unit})")
    cbar.outline.set_linewidth(0.5)
    # Label every other decade (10^0, 10^2, 10^4, ...) with unlabelled minor
    # ticks at every remaining decade.
    cbar.ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=100.0, subs=(1.0,), numticks=12))
    cbar.ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=10.0, subs=(1.0,), numticks=40))
    cbar.ax.yaxis.set_major_formatter(mpl.ticker.LogFormatterMathtext(base=10.0, labelOnlyBase=True))
    cbar.ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    cbar.ax.tick_params(which="major", width=0.5, length=2.5)
    cbar.ax.tick_params(which="minor", width=0.4, length=1.4)
    # Flip so tighter binding (low EC50, dark purple) is at the top.
    cbar.ax.invert_yaxis()

    if title:
        fig.suptitle(title, fontsize=8, y=1.0 - 0.015 / fig_h_in)

    return fig, axes


def _draw_panel(
    ax: plt.Axes, mean: np.ndarray, err: np.ndarray,
    metal_labels: list[str], protein_labels: list[str] | None,
    cmap: mpl.colors.Colormap, norm: mpl.colors.Normalize,
    size_by_error: bool, hatch_edges: np.ndarray | None, annotate: bool,
) -> None:
    n_rows, n_cols = mean.shape
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_aspect("equal")

    for i in range(n_rows):
        for j in range(n_cols):
            v = mean[i, j]
            if not np.isfinite(v) or v <= 0:
                continue
            e = err[i, j]
            side = _cell_side(e, v, size_by_error)
            face = cmap(norm(v))
            ax.add_patch(mpl.patches.Rectangle(
                (j - side / 2, i - side / 2), side, side,
                facecolor=face, edgecolor="none",
            ))
            if hatch_edges is not None:
                hatch = _hatch_for_cv(e, v, hatch_edges)
                if hatch:
                    ax.add_patch(mpl.patches.Rectangle(
                        (j - side / 2, i - side / 2), side, side,
                        facecolor="none", edgecolor=mpl.rcParams["hatch.color"],
                        hatch=hatch, linewidth=0.0,
                    ))

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(metal_labels)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)

    if protein_labels is not None:
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(protein_labels, fontstyle="italic")
    else:
        ax.set_yticks([])

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="0.85", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if annotate:
        _annotate_cells(ax, mean, err, cmap, norm)


def _extend_cmap(cmap: mpl.colors.Colormap, vmin: float, vmax: float,
                 cbar_vmin: float, cbar_vmax: float) -> mpl.colors.Colormap:
    """Colormap that spans [cbar_vmin, cbar_vmax] in log space but keeps the
    original `cmap` gradient confined to the [vmin, vmax] sub-range.

    Positions below vmin render as `cmap(0)` (solid), positions above vmax as
    `cmap(1)` (solid). Used to give the colorbar labeled tick decades outside
    the clipped cell range while showing that those regions are saturated.
    """
    log_range = np.log10(cbar_vmax) - np.log10(cbar_vmin)
    f_lo = (np.log10(vmin) - np.log10(cbar_vmin)) / log_range
    f_hi = (np.log10(vmax) - np.log10(cbar_vmin)) / log_range
    n = 256
    stops = np.linspace(0.0, 1.0, n)
    colors_list = []
    for s in stops:
        if s <= f_lo:
            colors_list.append(cmap(0.0))
        elif s >= f_hi:
            colors_list.append(cmap(1.0))
        else:
            colors_list.append(cmap((s - f_lo) / (f_hi - f_lo)))
    return LinearSegmentedColormap.from_list("ec50_ext", colors_list)


def _hatch_edges(mean: np.ndarray, err: np.ndarray) -> np.ndarray | None:
    """Quantile edges over the observed CV distribution, matched to HATCH_PATTERNS."""
    mask = np.isfinite(mean) & np.isfinite(err) & (mean > 0)
    if not mask.any():
        return None
    cv = err[mask] / mean[mask]
    if cv.size == 0 or np.nanmax(cv) <= 0:
        return None
    n = len(HATCH_PATTERNS)
    # n+1 edges -> n bins; use equal quantiles so cells are spread across levels
    # regardless of skew. Bin i is [edges[i], edges[i+1]).
    qs = np.linspace(0.0, 1.0, n + 1)
    edges = np.quantile(cv, qs)
    edges[-1] = np.nextafter(edges[-1], np.inf)  # ensure max value lands in last bin
    return edges


def _hatch_for_cv(err: float, mean: float, edges: np.ndarray) -> str:
    if not np.isfinite(err) or mean <= 0:
        return ""
    cv = err / mean
    idx = int(np.searchsorted(edges, cv, side="right")) - 1
    idx = max(0, min(idx, len(HATCH_PATTERNS) - 1))
    return HATCH_PATTERNS[idx]


def _cell_side(err: float, mean: float, size_by_error: bool) -> float:
    """Side length (in data units) for the colored square in a cell."""
    if not size_by_error or not np.isfinite(err) or mean <= 0:
        return 0.9
    cv = err / mean
    # Map CV=0 -> 0.9 (full-ish cell), CV>=0.5 -> 0.35 (small square), linearly.
    frac = 1.0 - min(cv / 0.5, 1.0)
    return 0.35 + frac * (0.9 - 0.35)


def _annotate_cells(
    ax: plt.Axes, mean: np.ndarray, err: np.ndarray,
    cmap: mpl.colors.Colormap, norm: mpl.colors.Normalize,
) -> None:
    thresh = 0.55  # switch text to white above this normalized value
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            v = mean[i, j]
            if not np.isfinite(v):
                continue
            label = _format_value(v, err[i, j])
            color = "white" if norm(v) < 1 - thresh else "black"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=5, color=color)


def _decade_fmt(x: float, _pos: int) -> str:
    """Plain-text decade tick label (e.g. 0.001, 0.1, 1, 10, 100). Editable."""
    if not np.isfinite(x) or x <= 0:
        return ""
    return f"{x:g}"


def _format_value(v: float, e: float) -> str:
    def fmt(x: float) -> str:
        if x >= 100:
            return f"{x:.0f}"
        if x >= 10:
            return f"{x:.1f}"
        if x >= 1:
            return f"{x:.2f}"
        return f"{x:.2g}"
    if np.isfinite(e):
        return f"{fmt(v)}\n± {fmt(e)}"
    return fmt(v)


if __name__ == "__main__":
    main()
