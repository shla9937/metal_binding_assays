#!/usr/bin/env python3
"""Diagnostic plot for the per-protein Tobit fit used in
`plot_dsf_selectivity_scatter.py`.

For each protein it draws, on a shared log(EC50) axis:
  * every metal's observed EC50 as a point on the appropriate band
    (REE/target on top, contaminant/off-target on bottom);
  * every censored metal (Prism failed fit, or fit above `--max-fit-ec50`)
    as a right-arrow at the censoring ceiling C, since we only know
    EC50 >= C;
  * the fitted per-class log-normal densities N(mu_L, sigma^2) and
    N(mu_O, sigma^2) as smooth curves above their bands;
  * mu_L and mu_O as vertical marks with 95% CI shading from the
    observed Fisher information (same source as the scatter plot);
  * the assay ceiling C as a vertical dotted line;
  * a header giving Delta = mu_O - mu_L (log-fold selectivity),
    exp(Delta) (fold), and the Wald p-value.

Together this shows what the Tobit MLE is actually doing:

  log EC50_{i in class j} ~ N(mu_j, sigma^2),   j in {L, O}

  observed contribution:   log phi((y_i - mu_j)/sigma)/sigma
  censored contribution:   log(1 - Phi((log C - mu_j)/sigma))
                        = log Phi((mu_j - log C)/sigma)

  MLE:    (mu_L, mu_O, sigma)  =  argmax  sum of contributions above.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_dsf_selectivity_scatter as sel  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", type=Path, required=True)
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--protein", nargs="+", default=None,
                   help="Subset of protein names to show (default: all in the table).")
    p.add_argument("--input-unit", default="µM", choices=list(sel.UNIT_SCALE))
    p.add_argument("--display-unit", default="µM", choices=list(sel.UNIT_SCALE))
    p.add_argument("--nonbinding-ec50", type=float, default=1000.0,
                   help="Right-censoring ceiling in --display-unit (default 1000 µM).")
    p.add_argument("--max-fit-ec50", type=float, default=1e5)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def _prepare_per_protein(df: pd.DataFrame,
                         input_unit: str, display_unit: str,
                         nonbinding_ec50: float, max_fit_ec50: float):
    """Return (all_vals, lcols, ocols, log_ceiling, log_floor, ceiling)."""
    scale = sel.UNIT_SCALE[input_unit] / sel.UNIT_SCALE[display_unit]
    lcols, ocols = sel._partition_columns(df)
    all_vals = df[lcols + ocols].apply(pd.to_numeric, errors="coerce") * scale
    all_vals = all_vals.where((all_vals > 0) & (all_vals <= max_fit_ec50))
    all_vals = all_vals.where(all_vals <= nonbinding_ec50)
    log_ceiling = float(np.log(nonbinding_ec50))
    finite_pos = all_vals.to_numpy().ravel()
    finite_pos = finite_pos[np.isfinite(finite_pos) & (finite_pos > 0)]
    log_floor = float(np.log(finite_pos.min())) - np.log(100.0) if finite_pos.size else -50.0
    return all_vals, lcols, ocols, log_ceiling, log_floor, nonbinding_ec50


def _panel(ax, protein: str, l_row: pd.Series, o_row: pd.Series,
           lcols: list[str], ocols: list[str],
           log_ceiling: float, log_floor: float,
           display_unit: str, ceiling: float, xlim: tuple[float, float]) -> None:
    y_L_obs = np.log(l_row.dropna().to_numpy(dtype=float))
    y_O_obs = np.log(o_row.dropna().to_numpy(dtype=float))
    n_L_cens = int(l_row.isna().sum())
    n_O_cens = int(o_row.isna().sum())

    l_cens_syms = [sel._strip_oxidation(c) for c, v in zip(lcols, l_row) if pd.isna(v)]
    o_cens_syms = [sel._strip_oxidation(c) for c, v in zip(ocols, o_row) if pd.isna(v)]
    l_obs_syms = [sel._strip_oxidation(c) for c, v in zip(lcols, l_row) if not pd.isna(v)]
    o_obs_syms = [sel._strip_oxidation(c) for c, v in zip(ocols, o_row) if not pd.isna(v)]

    fit = sel._tobit_two_group(y_L_obs, n_L_cens, y_O_obs, n_O_cens,
                               log_ceiling, log_floor)

    # y-band centers.
    yL_base, yO_base = 1.0, 0.0
    band_h = 0.35

    # Points: log-scale plot on natural EC50 axis.
    def _scatter(vals_log, syms, y_base, color):
        if not len(vals_log):
            return
        x = np.exp(vals_log)
        rng = np.random.default_rng(hash(protein) & 0xFFFF)
        jitter = rng.uniform(-band_h * 0.4, band_h * 0.4, size=len(x))
        ax.scatter(x, y_base + jitter, s=10, color=color,
                   edgecolor="white", linewidth=0.3, zorder=4)
        for xi, yi, s in zip(x, y_base + jitter, syms):
            ax.annotate(s, xy=(xi, yi), xytext=(2, 2),
                        textcoords="offset points", fontsize=5, color="0.25")

    _scatter(y_L_obs, l_obs_syms, yL_base, "#1f77b4")
    _scatter(y_O_obs, o_obs_syms, yO_base, "0.4")

    # Censored: right-arrows at the ceiling for every metal we know is >= C.
    def _cens_arrows(n, syms, y_base, color):
        if not n:
            return
        rng = np.random.default_rng((hash(protein) ^ hash(y_base)) & 0xFFFF)
        ys = rng.uniform(-band_h * 0.4, band_h * 0.4, size=n) + y_base
        for yi, s in zip(ys, syms):
            ax.annotate("", xy=(ceiling * 3.0, yi), xytext=(ceiling, yi),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        lw=0.6, shrinkA=0, shrinkB=0),
                        zorder=3)
            ax.annotate(s, xy=(ceiling, yi), xytext=(2, band_h * 15),
                        textcoords="offset points", fontsize=5, color=color)

    _cens_arrows(n_L_cens, l_cens_syms, yL_base, "#1f77b4")
    _cens_arrows(n_O_cens, o_cens_syms, yO_base, "0.4")

    # Fitted per-class log-normal densities (on log-EC50 axis).
    if fit is not None:
        mu_L, mu_O, sigma = fit["mu_L"], fit["mu_O"], fit["sigma"]
        xs_log = np.linspace(np.log(xlim[0]), np.log(xlim[1]), 400)
        xs = np.exp(xs_log)
        pdf_L = norm.pdf(xs_log, loc=mu_L, scale=sigma)
        pdf_O = norm.pdf(xs_log, loc=mu_O, scale=sigma)
        # Scale each PDF to a uniform display height so both curves are visible.
        pdf_L = pdf_L / pdf_L.max() * (band_h * 0.9)
        pdf_O = pdf_O / pdf_O.max() * (band_h * 0.9)
        ax.plot(xs, yL_base + band_h * 0.6 + pdf_L,
                color="#1f77b4", lw=0.9, zorder=3)
        ax.fill_between(xs, yL_base + band_h * 0.6, yL_base + band_h * 0.6 + pdf_L,
                        color="#1f77b4", alpha=0.12, lw=0, zorder=2)
        ax.plot(xs, yO_base - band_h * 0.6 - pdf_O,
                color="0.4", lw=0.9, zorder=3)
        ax.fill_between(xs, yO_base - band_h * 0.6 - pdf_O, yO_base - band_h * 0.6,
                        color="0.4", alpha=0.12, lw=0, zorder=2)

        # Vertical marks for mu_L / mu_O and their 95% CI (from Fisher info).
        for mu, se, y_base, color in ((mu_L, fit["se_L"], yL_base, "#1f77b4"),
                                      (mu_O, fit["se_O"], yO_base, "0.4")):
            x_mu = float(np.exp(mu))
            ax.plot([x_mu, x_mu],
                    [y_base - band_h, y_base + band_h],
                    color=color, lw=1.0, zorder=5)
            if np.isfinite(se) and se > 0:
                se_c = min(se, 3.0)
                lo = float(np.exp(mu - 1.96 * se_c))
                hi = float(np.exp(mu + 1.96 * se_c))
                ax.axvspan(lo, hi, ymin=(y_base - band_h + 1.6) / 3.2,
                           ymax=(y_base + band_h + 1.6) / 3.2,
                           color=color, alpha=0.10, lw=0, zorder=1)

        delta = fit["delta_log"]
        p = fit["p"]
        header = (f"{protein}:  Δ = μ$_O$ − μ$_L$ = {delta:+.2f}   "
                  f"({np.exp(delta):.1f}× selective for REEs)")
        if np.isfinite(p):
            header += f"   p = {p:.1e}"
    else:
        header = f"{protein}:  Tobit unidentified (a group has no observed EC50)"

    # Censoring ceiling.
    ax.axvline(ceiling, color="0.5", lw=0.5, ls=":", zorder=1)
    ax.text(ceiling, 1.7, f"C = {ceiling:g} {display_unit}",
            fontsize=5, color="0.4", ha="right", va="top",
            rotation=90, backgroundcolor="white")

    # Band labels.
    ax.text(xlim[0] * 1.05, yL_base + band_h + 0.05, "REE / target",
            fontsize=6, color="#1f77b4", ha="left", va="bottom")
    ax.text(xlim[0] * 1.05, yO_base - band_h - 0.05, "contaminant / off-target",
            fontsize=6, color="0.35", ha="left", va="top")

    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(-1.6, 1.9)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_title(header, loc="left", fontsize=7)


def main() -> None:
    args = parse_args()
    sel.configure_nature_style()
    df = sel.load_table(args.input)
    all_vals, lcols, ocols, log_ceiling, log_floor, ceiling = _prepare_per_protein(
        df, args.input_unit, args.display_unit,
        args.nonbinding_ec50, args.max_fit_ec50)

    proteins = args.protein or df["Protein"].astype(str).tolist()
    proteins = [p for p in proteins if p in df["Protein"].astype(str).tolist()]
    if not proteins:
        raise SystemExit("No matching proteins in the table.")

    finite = all_vals.to_numpy().ravel()
    finite = finite[np.isfinite(finite) & (finite > 0)]
    lo = 10 ** np.floor(np.log10(finite.min()) - 0.3)
    hi = 10 ** np.ceil(np.log10(max(finite.max(), ceiling)) + 0.5)
    xlim = (float(lo), float(hi))

    n = len(proteins)
    fig, axes = plt.subplots(n, 1, figsize=(6.0, max(1.4, 1.1 * n)),
                             sharex=True, squeeze=False)
    for ax, protein in zip(axes[:, 0], proteins):
        idx = df.index[df["Protein"].astype(str) == protein][0]
        l_row = all_vals[lcols].iloc[idx]
        o_row = all_vals[ocols].iloc[idx]
        _panel(ax, protein, l_row, o_row, lcols, ocols,
               log_ceiling, log_floor, args.display_unit, ceiling, xlim)
    axes[-1, 0].set_xlabel(f"EC50 ({args.display_unit}, log scale)")
    fig.tight_layout(pad=0.4)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=600, bbox_inches="tight")
        print(f"Saved diagnostic to {args.output}")
    if args.show or not args.output:
        plt.show()


if __name__ == "__main__":
    main()
