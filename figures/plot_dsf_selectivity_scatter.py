#!/usr/bin/env python3
"""Composite lanthanide-selectivity scatter with censoring-aware statistics.

Companion to `plot_ec50_heatmap.py`. Uses the same wide DSF EC50 table
(columns like `La3+`, `Cu2+`, `Ba2+_error`, ...) to identify selective
lanthanide binders based on binding affinity.

For each protein, log(EC50) values that fit are treated as observed and
metals that failed to fit are treated as right-censored at the assay
ceiling (default 1 mM = 1000 µM, the DSF titration ceiling; override with
`--nonbinding-ec50`). Observed EC50s above the ceiling are reclassified
as censored. A per-protein Tobit MLE

    log EC50 | class ~ Normal(mu_class, sigma^2)     (mu_L, mu_O, sigma)

is fit by numerical maximum likelihood; the selectivity effect is

    Delta = mu_other - mu_lanthanide

with a 95% CI from the observed Fisher information (Hessian of the negative
log-likelihood at the MLE). Points below y = x on the log-log scatter are
lanthanide-selective; the shaded band marks Delta >= log(threshold).

The kinome-style selectivity scores S(10) and S(100) (Fabian 2005 /
Karaman 2008) are also reported per protein:

    S(x) = (# non-lanthanide metals with EC50 <= x * geomean(lanthanide EC50))
           / (# non-lanthanide metals tested)

Lower S(x) = more selective. Censored non-lanthanide metals correctly
fail the S(x) numerator without any imputation.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

# The REE group = lanthanides (La-Lu, minus Pm) plus Sc and Y, per IUPAC.
# Everything else in the sheet (transitions, alkali, alkaline earth) is the
# "other" (off-target) group.
LANTHANIDES = ("Sc", "Y", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd",
               "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu")

SKIP = frozenset({"EDTA"})

UNIT_SCALE = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "µM": 1e-6, "nM": 1e-9, "pM": 1e-12}

PANEL_IN = 89.0 / 25.4  # Nature single-column panel

_OX_RE = re.compile(r"\s*\d*[+−\-]$")


def _strip_oxidation(name: str) -> str:
    return _OX_RE.sub("", str(name)).strip()


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
    stats, ceiling = summarize(df, input_unit=args.input_unit,
                               display_unit=args.display_unit,
                               nonbinding_ec50=args.nonbinding_ec50,
                               max_fit_ec50=args.max_fit_ec50,
                               s_folds=tuple(args.s_fold))
    _print_table(stats, args.display_unit, args.threshold, args.s_fold)
    if args.stats_output:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(args.stats_output, index=False)
        print(f"Saved stats table to {args.stats_output}")
    fig, _ = plot(stats, threshold=args.threshold,
                  display_unit=args.display_unit, ceiling=ceiling)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=600, bbox_inches="tight")
        print(f"Saved figure to {args.output}")
    if args.show or not args.output:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", type=Path, required=True,
                   help="Path to input .xlsx or .csv (e.g. source/ped_cluster_dsf_ec50.xlsx)")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Path to save the figure. Use .pdf or .svg for editable-text vector output.")
    p.add_argument("--stats-output", type=Path, default=None,
                   help="Optional path to save the per-protein Tobit / S(x) stats as CSV.")
    p.add_argument("--input-unit", default="µM", choices=list(UNIT_SCALE),
                   help="Unit of EC50 values in the input file (default: µM)")
    p.add_argument("--display-unit", default="µM", choices=list(UNIT_SCALE),
                   help="Unit for axis labels and annotations (default: µM)")
    p.add_argument("-t", "--threshold", type=float, default=10.0,
                   help="Fold selectivity threshold for the highlighted band "
                        "(default 10x: Delta = log(threshold))")
    p.add_argument("--nonbinding-ec50", type=float, default=1000.0,
                   help="Right-censoring ceiling for failed fits (default 1000 µM = 1 mM, "
                        "the DSF titration ceiling). Value is interpreted in --display-unit.")
    p.add_argument("--max-fit-ec50", type=float, default=1e5,
                   help="EC50 values (in --display-unit) above this are treated as "
                        "spurious/failed fits and censored (default 1e5 µM = 100 mM, "
                        "well above any realistic DSF titration).")
    p.add_argument("--s-fold", type=float, nargs="+", default=[10.0, 100.0],
                   help="Fold thresholds for kinome-style S(x) scores (default: 10 100).")
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


def _partition_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split mean columns (not `_error`) into (lanthanides, others). EDTA skipped."""
    ln = {s.upper() for s in LANTHANIDES}
    lanth: list[str] = []
    other: list[str] = []
    for c in df.columns:
        if c == "Protein" or str(c).endswith("_error"):
            continue
        sym = _strip_oxidation(c).upper()
        if sym in SKIP:
            continue
        (lanth if sym in ln else other).append(c)
    return lanth, other


def _tobit_two_group(
    y_L_obs: np.ndarray, n_L_cens: int,
    y_O_obs: np.ndarray, n_O_cens: int,
    log_ceiling: float, log_floor: float,
) -> dict | None:
    """MLE for (mu_L, mu_O, sigma) with right-censoring at `log_ceiling`.

    Requires >=1 observed value in EACH group so both intercepts are identified;
    otherwise returns None. Parameters are bounded (L-BFGS-B) to prevent the
    optimizer from wandering to infinity when censoring is heavy:

        mu_L, mu_O in [log_floor,  log_ceiling + log(10)]
        sigma       in [0.1,       10]                        (on log scale)

    Delta CI uses the observed Fisher information at the MLE (numerical Hessian).
    """
    if y_L_obs.size == 0 or y_O_obs.size == 0:
        return None

    mu_hi = log_ceiling + np.log(10.0)  # allow group mean up to 10x the ceiling
    bounds = [(log_floor, mu_hi), (log_floor, mu_hi), (np.log(0.1), np.log(10.0))]

    def neg_ll(params: np.ndarray) -> float:
        mu_L, mu_O, log_sigma = params
        sigma = float(np.exp(log_sigma))
        ll = float(norm.logpdf(y_L_obs, loc=mu_L, scale=sigma).sum())
        ll += float(norm.logpdf(y_O_obs, loc=mu_O, scale=sigma).sum())
        if n_L_cens:
            ll += n_L_cens * float(norm.logsf(log_ceiling, loc=mu_L, scale=sigma))
        if n_O_cens:
            ll += n_O_cens * float(norm.logsf(log_ceiling, loc=mu_O, scale=sigma))
        return -ll

    mu_L0 = float(np.clip(y_L_obs.mean(), log_floor, mu_hi))
    mu_O0 = float(np.clip(y_O_obs.mean(), log_floor, mu_hi))
    pooled = np.concatenate([y_L_obs - mu_L0, y_O_obs - mu_O0])
    sigma0 = float(np.std(pooled, ddof=1)) if pooled.size > 1 else 1.0
    sigma0 = float(np.clip(sigma0, 0.25, 5.0))
    x0 = np.array([mu_L0, mu_O0, np.log(sigma0)])

    res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds)
    if not res.success or not np.all(np.isfinite(res.x)):
        return None
    mu_L, mu_O, log_sigma = (float(v) for v in res.x)
    sigma = float(np.exp(log_sigma))

    # Numerical Hessian at the MLE for CIs; skip covariance if fit hits a bound.
    at_bound = any(
        abs(v - b[0]) < 1e-4 or abs(v - b[1]) < 1e-4
        for v, b in zip(res.x, bounds)
    )
    se_L = se_O = se_D = float("nan")
    z = float("nan")
    p = float("nan")
    if not at_bound:
        H = _numerical_hessian(neg_ll, res.x)
        try:
            cov = np.linalg.inv(H)
            var_L = max(float(cov[0, 0]), 0.0)
            var_O = max(float(cov[1, 1]), 0.0)
            var_D = max(float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]), 0.0)
            se_L, se_O, se_D = float(np.sqrt(var_L)), float(np.sqrt(var_O)), float(np.sqrt(var_D))
            if se_D > 0:
                z = (mu_O - mu_L) / se_D
                p = float(2.0 * norm.sf(abs(z)))
        except np.linalg.LinAlgError:
            pass

    delta = mu_O - mu_L
    return {
        "mu_L": mu_L, "mu_O": mu_O, "sigma": sigma,
        "se_L": se_L, "se_O": se_O,
        "delta_log": delta, "se_delta_log": se_D,
        "delta_ci_low": delta - 1.96 * se_D if np.isfinite(se_D) else float("nan"),
        "delta_ci_high": delta + 1.96 * se_D if np.isfinite(se_D) else float("nan"),
        "z": z, "p": p,
        "at_bound": at_bound,
    }


def _numerical_hessian(f, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Central-difference Hessian of scalar-valued f at x."""
    n = x.size
    H = np.zeros((n, n))
    f0 = float(f(x))
    for i in range(n):
        for j in range(i, n):
            xi = x.copy(); xi[i] += eps; xi[j] += eps
            fpp = float(f(xi))
            xi = x.copy(); xi[i] += eps; xi[j] -= eps
            fpm = float(f(xi))
            xi = x.copy(); xi[i] -= eps; xi[j] += eps
            fmp = float(f(xi))
            xi = x.copy(); xi[i] -= eps; xi[j] -= eps
            fmm = float(f(xi))
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps)
    return H


def _s_score(y_L_obs: np.ndarray, y_O_obs: np.ndarray,
             n_O_cens: int, fold: float) -> float:
    """Kinome-style S(x): fraction of non-lanthanide metals whose EC50 <=
    fold * geomean(observed lanthanide EC50). Censored fits count as > fold * ref."""
    n_O_tot = len(y_O_obs) + n_O_cens
    if n_O_tot == 0 or y_L_obs.size == 0:
        return float("nan")
    log_ref = float(y_L_obs.mean())  # log of geometric mean of observed Ln EC50
    thresh = log_ref + np.log(fold)
    hits = int((y_O_obs <= thresh).sum())
    return hits / n_O_tot


def summarize(df: pd.DataFrame, input_unit: str, display_unit: str,
              nonbinding_ec50: float | None,
              max_fit_ec50: float,
              s_folds: tuple[float, ...]) -> tuple[pd.DataFrame, float]:
    """Per-protein Tobit MLE and S(x) scores, censoring failed fits at the assay ceiling.

    EC50 values > `max_fit_ec50` (in `display_unit`) are treated as Prism-style
    failed fits and re-classified as censored non-binders before any statistics
    are computed. Returns (stats, ceiling_in_display_unit). Delta and Tobit
    intercepts are on the natural-log EC50 scale, in `display_unit`;
    `mu_L_ec50` / `mu_O_ec50` translate back to EC50 units (µM by default).
    """
    scale = UNIT_SCALE[input_unit] / UNIT_SCALE[display_unit]
    lcols, ocols = _partition_columns(df)
    if not lcols:
        raise ValueError(f"No lanthanide columns found (expected any of {LANTHANIDES})")
    if not ocols:
        raise ValueError("No non-lanthanide competitor columns found")

    all_vals = df[lcols + ocols].apply(pd.to_numeric, errors="coerce") * scale
    # Positive fits below the plausibility cap are "observed"; everything else
    # (NaN, non-positive, or absurdly large Prism-failure values) is censored.
    all_vals = all_vals.where((all_vals > 0) & (all_vals <= max_fit_ec50))

    if nonbinding_ec50 is None:
        finite = all_vals.to_numpy().ravel()
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError("Input has no positive EC50 values")
        nonbinding_ec50 = float(np.nanmax(finite))
    log_ceiling = float(np.log(nonbinding_ec50))

    # Any "observed" EC50 above the censoring ceiling is really a non-binder;
    # reclassify it as censored so it can't pull a group mean past the ceiling.
    all_vals = all_vals.where(all_vals <= nonbinding_ec50)

    finite_pos = all_vals.to_numpy().ravel()
    finite_pos = finite_pos[np.isfinite(finite_pos) & (finite_pos > 0)]
    log_floor = float(np.log(finite_pos.min())) - np.log(100.0) if finite_pos.size else -50.0

    rows: list[dict] = []
    for i, protein in enumerate(df["Protein"].astype(str)):
        l_row = all_vals[lcols].iloc[i]
        o_row = all_vals[ocols].iloc[i]
        y_L = np.log(l_row.dropna().to_numpy(dtype=float))
        y_O = np.log(o_row.dropna().to_numpy(dtype=float))
        n_L_cens = int(l_row.isna().sum())
        n_O_cens = int(o_row.isna().sum())
        if (len(y_L) + n_L_cens) == 0 or (len(y_O) + n_O_cens) == 0:
            continue

        fit = _tobit_two_group(y_L, n_L_cens, y_O, n_O_cens, log_ceiling, log_floor)
        entry = {
            "Protein": protein,
            "n_lanthanides": len(y_L),
            "n_lanthanides_censored": n_L_cens,
            "n_lanthanides_total": len(y_L) + n_L_cens,
            "n_other": len(y_O),
            "n_other_censored": n_O_cens,
            "n_other_total": len(y_O) + n_O_cens,
        }
        if fit is None:
            # At least one group has NO observed fits: Tobit intercept is
            # unidentified. Anchor that group's mean at the ceiling (a lower
            # bound on the true composite EC50).
            mu_L = float(y_L.mean()) if y_L.size else log_ceiling
            mu_O = float(y_O.mean()) if y_O.size else log_ceiling
            entry.update({
                "mu_L": mu_L, "mu_O": mu_O, "sigma": float("nan"),
                "se_L": float("nan"), "se_O": float("nan"),
                "delta_log": mu_O - mu_L, "se_delta_log": float("nan"),
                "delta_ci_low": float("nan"), "delta_ci_high": float("nan"),
                "z": float("nan"), "p": float("nan"),
                "at_bound": True,
                "L_censored_only": bool(y_L.size == 0),
                "O_censored_only": bool(y_O.size == 0),
            })
        else:
            entry.update(fit)
            entry["L_censored_only"] = False
            entry["O_censored_only"] = False
        entry["mu_L_ec50"] = float(np.exp(entry["mu_L"])) if np.isfinite(entry.get("mu_L", np.nan)) else np.nan
        entry["mu_O_ec50"] = float(np.exp(entry["mu_O"])) if np.isfinite(entry.get("mu_O", np.nan)) else np.nan
        for fold in s_folds:
            entry[f"S({fold:g})"] = _s_score(y_L, y_O, n_O_cens, fold)
        rows.append(entry)

    stats = pd.DataFrame(rows)
    return stats, nonbinding_ec50


def _print_table(stats: pd.DataFrame, display_unit: str,
                 threshold: float, s_folds: list[float]) -> None:
    if stats.empty:
        return
    log_thresh = float(np.log(threshold))
    cols = ["Protein",
            "mu_L_ec50", "mu_O_ec50",
            "delta_log", "delta_ci_low", "delta_ci_high", "p",
            "n_lanthanides", "n_lanthanides_censored",
            "n_other", "n_other_censored"]
    cols += [f"S({f:g})" for f in s_folds]
    show = stats[cols].copy()
    show.insert(1, "selective",
                (show["delta_ci_low"] >= log_thresh).map({True: "yes", False: "no"}))
    formatters = {
        "mu_L_ec50": lambda v: f"{v:8.3g}",
        "mu_O_ec50": lambda v: f"{v:8.3g}",
        "delta_log": lambda v: f"{v:+.2f}",
        "delta_ci_low": lambda v: f"{v:+.2f}",
        "delta_ci_high": lambda v: f"{v:+.2f}",
        "p": lambda v: f"{v:.2g}" if np.isfinite(v) else "nan",
    }
    for f in s_folds:
        formatters[f"S({f:g})"] = lambda v: f"{v:.2f}" if np.isfinite(v) else "nan"
    print()
    print(f"Per-protein Tobit MLE (log EC50 in {display_unit}), "
          f"selective at Delta >= log({threshold:g}) = {log_thresh:.2f}:")
    print(show.to_string(index=False, formatters=formatters))
    print()


def plot(stats: pd.DataFrame, threshold: float,
         display_unit: str, ceiling: float) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(PANEL_IN, PANEL_IN))
    ax.set_xscale("log")
    ax.set_yscale("log")

    good = stats.dropna(subset=["mu_L_ec50", "mu_O_ec50"]).copy()
    if good.empty:
        raise ValueError("Tobit MLE failed for every protein")
    log_thresh = float(np.log(threshold))
    good["is_selective"] = good["delta_ci_low"] >= log_thresh
    good["at_bound"] = good["at_bound"].fillna(False).astype(bool)

    finite = pd.concat([good["mu_L_ec50"], good["mu_O_ec50"]]).astype(float)
    finite = finite[np.isfinite(finite) & (finite > 0)]
    lo = 10 ** np.floor(np.log10(finite.min()) - 0.3)
    hi = 10 ** np.ceil(np.log10(max(finite.max(), ceiling)) + 0.3)

    ref = np.array([lo, hi])
    ax.plot(ref, ref, linestyle=":", color="0.5", linewidth=0.75,
            label="equal composite affinity", zorder=2)
    # Selective binders sit above y = threshold * x (non-REE composite EC50 is
    # >= threshold-fold higher than REE composite EC50 -> tighter REE binding).
    ax.plot(ref, ref * threshold, linestyle="--", color="seagreen", linewidth=0.75,
            label=f"{threshold:g}× REE-selective", zorder=2)
    ax.fill_between(ref, ref * threshold, hi * np.ones_like(ref),
                    color="seagreen", alpha=0.08, linewidth=0, zorder=1)

    ax.axhline(ceiling, color="0.75", linewidth=0.4, linestyle=":", zorder=1)
    ax.axvline(ceiling, color="0.75", linewidth=0.4, linestyle=":", zorder=1)

    # Regular Tobit points with 95% CI error bars (colour by significance).
    # X = REE composite (mu_L), Y = non-REE composite (mu_O) so the layout
    # mirrors the ICP-MS transitions-vs-lanthanides scatter.
    fitted = good[~good["at_bound"]]
    for is_sel, sub, color in (
        (True, fitted[fitted["is_selective"]], "#1f77b4"),
        (False, fitted[~fitted["is_selective"]], "0.55"),
    ):
        if sub.empty:
            continue
        x = sub["mu_L_ec50"].to_numpy(dtype=float)
        y = sub["mu_O_ec50"].to_numpy(dtype=float)
        # Cap the log-scale SE at 3 (~2000-fold CI) so extreme fits don't blow up
        # the arithmetic errorbar magnitudes on the log axis.
        se_x = np.minimum(sub["se_L"].to_numpy(dtype=float), 3.0)
        se_y = np.minimum(sub["se_O"].to_numpy(dtype=float), 3.0)
        xerr_lo = x - np.exp(np.log(x) - 1.96 * se_x)
        xerr_hi = np.exp(np.log(x) + 1.96 * se_x) - x
        yerr_lo = y - np.exp(np.log(y) - 1.96 * se_y)
        yerr_hi = np.exp(np.log(y) + 1.96 * se_y) - y
        ax.errorbar(x, y, xerr=[xerr_lo, xerr_hi], yerr=[yerr_lo, yerr_hi],
                    fmt="o", color=color, ecolor=color,
                    markersize=4, markeredgewidth=0,
                    elinewidth=0.4, capsize=1.0, capthick=0.4, zorder=3,
                    label=("Δ 95% CI ≥ log(threshold)" if is_sel
                           else "not significant"))

    # Unidentified fits: at least one group is fully censored (no observed EC50s).
    # Anchor the composite at the ceiling with a directional arrow to indicate a
    # lower bound on the mean EC50 for that group.
    bounded = good[good["at_bound"]]
    for _, row in bounded.iterrows():
        x = row["mu_L_ec50"]
        y = row["mu_O_ec50"]
        marker = "o"
        if row.get("L_censored_only", False):
            x = ceiling
            marker = ">"  # REE composite EC50 is >= ceiling (non-binder)
        if row.get("O_censored_only", False):
            y = ceiling
            marker = "^" if marker == "o" else marker  # non-REE composite EC50 >= ceiling (selective)
        color = "#1f77b4" if row.get("O_censored_only", False) else "0.55"
        ax.scatter([x], [y], s=22, color=color, edgecolor="none",
                   marker=marker, zorder=3)

    for _, row in good.iterrows():
        x = row["mu_L_ec50"] if not row.get("L_censored_only", False) else ceiling
        y = row["mu_O_ec50"] if not row.get("O_censored_only", False) else ceiling
        ax.annotate(row["Protein"], xy=(x, y),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=6, fontstyle="italic")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Tobit composite REE EC50 ({display_unit})")
    ax.set_ylabel(f"Tobit composite non-REE EC50 ({display_unit})")
    ax.set_title("REE selectivity of Ped cluster (DSF)")
    ax.legend(loc="lower right", frameon=False, handlelength=1.5,
              borderaxespad=0.2, handletextpad=0.4,
              title=f"censored at EC50 = {ceiling:.0f} {display_unit}",
              title_fontsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.2)
    return fig, ax


if __name__ == "__main__":
    main()
