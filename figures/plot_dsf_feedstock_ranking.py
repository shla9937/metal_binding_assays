#!/usr/bin/env python3
"""Feedstock-weighted Tobit composite for ranking REE binders.

Extension of `plot_dsf_selectivity_scatter.py`. Given (i) a DSF EC50 table
(wide layout with columns like `La3+`, `Cu2+`, ...) and (ii) a feedstock
composition table (columns `Metal` and `Concentration`), each metal's
per-protein log-likelihood contribution to the two-group Tobit MLE is
weighted by that metal's concentration in the feedstock. This turns the
generic REE-vs-competitor selectivity fit into a feedstock-specific one:

    log EC50 | class ~ Normal(mu_class, sigma^2)     class in {REE, contam.}

    -log L = - sum_{i in REE, obs}    w_i log phi((y_i - mu_L)/sigma)/sigma
             - sum_{i in REE, cens}   w_i log(1 - Phi((C - mu_L)/sigma))
             - sum_{i in contam obs}  w_i log phi((y_i - mu_O)/sigma)/sigma
             - sum_{i in contam cens} w_i log(1 - Phi((C - mu_O)/sigma))

Weights within each group are normalized to sum to the number of metals
in that group so effective sample sizes (and CI magnitudes) remain
comparable to the unweighted fit. Metals absent from the feedstock get
weight 0 -> excluded. Metals absent from the DSF panel are ignored with
a warning.

Selectivity effect Delta = mu_O - mu_L is reported per protein with a
95% CI from the observed Fisher information at the MLE; proteins are
ranked by Delta so the top row of the forest plot is the predicted
best puller for the given feedstock, followed by the rest.
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
from scipy.optimize import minimize
from scipy.stats import norm

LANTHANIDES = ("Sc", "Y", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd",
               "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu")

SKIP = frozenset({"EDTA"})

UNIT_SCALE = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "µM": 1e-6, "nM": 1e-9, "pM": 1e-12}

PANEL_IN = 89.0 / 25.4  # Nature single-column panel

_OX_RE = re.compile(r"\s*\d*[+−\-]$")


def _strip_oxidation(name: str) -> str:
    return _OX_RE.sub("", str(name)).strip()


def configure_nature_style() -> None:
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
    feedstock = load_feedstock(args.feedstock)
    stats, ceiling, coverage = summarize(
        df, feedstock,
        input_unit=args.input_unit,
        display_unit=args.display_unit,
        nonbinding_ec50=args.nonbinding_ec50,
        max_fit_ec50=args.max_fit_ec50,
        target_group=tuple(s.upper() for s in args.target_group),
    )
    _print_coverage(coverage)
    _print_table(stats, args.display_unit, args.threshold)
    if args.stats_output:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(args.stats_output, index=False)
        print(f"Saved stats table to {args.stats_output}")
    fig, _ = plot(stats, threshold=args.threshold, display_unit=args.display_unit,
                  ceiling=ceiling, feedstock_name=args.feedstock_name or args.feedstock.stem)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=600, bbox_inches="tight")
        print(f"Saved figure to {args.output}")
    if args.show or not args.output:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", type=Path, required=True,
                   help="Path to the DSF EC50 table (.xlsx or .csv), same layout "
                        "as `plot_dsf_selectivity_scatter.py`.")
    p.add_argument("-f", "--feedstock", type=Path, required=True,
                   help="Path to a feedstock composition table (.csv/.xlsx) with "
                        "columns `Metal` (element symbol, e.g. Nd) and "
                        "`Concentration` (any unit -- only relative values matter).")
    p.add_argument("--feedstock-name", default=None,
                   help="Human-readable feedstock label for the figure title "
                        "(default: feedstock file stem).")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Path to save the figure. Use .pdf or .svg for editable-text vectors.")
    p.add_argument("--stats-output", type=Path, default=None,
                   help="Optional path to save the per-protein weighted Tobit stats as CSV.")
    p.add_argument("--input-unit", default="µM", choices=list(UNIT_SCALE),
                   help="Unit of EC50 values in the input file (default: µM)")
    p.add_argument("--display-unit", default="µM", choices=list(UNIT_SCALE),
                   help="Unit for axis labels and annotations (default: µM)")
    p.add_argument("-t", "--threshold", type=float, default=10.0,
                   help="Fold selectivity threshold marked on the forest plot (default 10x).")
    p.add_argument("--nonbinding-ec50", type=float, default=1000.0,
                   help="Right-censoring ceiling for failed fits, in --display-unit "
                        "(default 1000 µM = 1 mM, the DSF titration ceiling).")
    p.add_argument("--max-fit-ec50", type=float, default=1e5,
                   help="EC50 values above this (in --display-unit) are treated as "
                        "spurious/failed fits and censored (default 1e5 µM).")
    p.add_argument("--target-group", nargs="+", default=list(LANTHANIDES),
                   help="Element symbols that define the 'target' group (default: REEs). "
                        "Every other tested metal is treated as a contaminant.")
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


def load_feedstock(path: Path) -> dict[str, float]:
    """Return {element_symbol_upper: concentration} with non-positive rows dropped."""
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        fs = pd.read_excel(path, comment="#")
    elif suffix == ".csv":
        fs = pd.read_csv(path, comment="#")
    else:
        raise ValueError(f"Unsupported feedstock file type: {suffix}")
    cols = {c.lower(): c for c in fs.columns}
    if "metal" not in cols or "concentration" not in cols:
        raise ValueError("Feedstock table must have `Metal` and `Concentration` columns")
    fs = fs.rename(columns={cols["metal"]: "Metal", cols["concentration"]: "Concentration"})
    fs["Metal"] = fs["Metal"].astype(str).map(_strip_oxidation).str.upper()
    fs["Concentration"] = pd.to_numeric(fs["Concentration"], errors="coerce")
    fs = fs.dropna(subset=["Metal", "Concentration"])
    fs = fs[fs["Concentration"] > 0]
    # Combine duplicate rows (e.g. Fe listed twice) by summing concentrations.
    fs = fs.groupby("Metal", as_index=True)["Concentration"].sum()
    return fs.to_dict()


def _partition_columns(df: pd.DataFrame,
                       target_syms: tuple[str, ...]) -> tuple[list[str], list[str]]:
    target = set(target_syms)
    lanth: list[str] = []
    other: list[str] = []
    for c in df.columns:
        if c == "Protein" or str(c).endswith("_error"):
            continue
        sym = _strip_oxidation(c).upper()
        if sym in SKIP:
            continue
        (lanth if sym in target else other).append(c)
    return lanth, other


def _column_weights(cols: list[str], feedstock: dict[str, float]) -> np.ndarray:
    return np.array([float(feedstock.get(_strip_oxidation(c).upper(), 0.0)) for c in cols])


def _weighted_tobit_two_group(
    y_L_obs: np.ndarray, w_L_obs: np.ndarray, w_L_cens: float,
    y_O_obs: np.ndarray, w_O_obs: np.ndarray, w_O_cens: float,
    log_ceiling: float, log_floor: float,
) -> dict | None:
    """Weighted Tobit MLE for (mu_L, mu_O, sigma) with right-censoring at
    `log_ceiling`. Requires positive total weight in BOTH groups plus >=1
    observed value in each group; otherwise returns None.
    """
    if y_L_obs.size == 0 or y_O_obs.size == 0:
        return None
    if w_L_obs.sum() + w_L_cens <= 0 or w_O_obs.sum() + w_O_cens <= 0:
        return None

    mu_hi = log_ceiling + np.log(10.0)
    bounds = [(log_floor, mu_hi), (log_floor, mu_hi), (np.log(0.1), np.log(10.0))]

    def neg_ll(params: np.ndarray) -> float:
        mu_L, mu_O, log_sigma = params
        sigma = float(np.exp(log_sigma))
        ll = 0.0
        if y_L_obs.size:
            ll += float((w_L_obs * norm.logpdf(y_L_obs, loc=mu_L, scale=sigma)).sum())
        if y_O_obs.size:
            ll += float((w_O_obs * norm.logpdf(y_O_obs, loc=mu_O, scale=sigma)).sum())
        if w_L_cens > 0:
            ll += w_L_cens * float(norm.logsf(log_ceiling, loc=mu_L, scale=sigma))
        if w_O_cens > 0:
            ll += w_O_cens * float(norm.logsf(log_ceiling, loc=mu_O, scale=sigma))
        return -ll

    # Weighted starting values.
    if w_L_obs.sum() > 0:
        mu_L0 = float(np.average(y_L_obs, weights=w_L_obs))
    else:
        mu_L0 = log_ceiling
    if w_O_obs.sum() > 0:
        mu_O0 = float(np.average(y_O_obs, weights=w_O_obs))
    else:
        mu_O0 = log_ceiling
    mu_L0 = float(np.clip(mu_L0, log_floor, mu_hi))
    mu_O0 = float(np.clip(mu_O0, log_floor, mu_hi))
    pooled = np.concatenate([y_L_obs - mu_L0, y_O_obs - mu_O0])
    pooled_w = np.concatenate([w_L_obs, w_O_obs])
    if pooled.size > 1 and pooled_w.sum() > 0:
        mean = float(np.average(pooled, weights=pooled_w))
        var = float(np.average((pooled - mean) ** 2, weights=pooled_w))
        sigma0 = float(np.sqrt(var)) if var > 0 else 1.0
    else:
        sigma0 = 1.0
    sigma0 = float(np.clip(sigma0, 0.25, 5.0))
    x0 = np.array([mu_L0, mu_O0, np.log(sigma0)])

    res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds)
    if not res.success or not np.all(np.isfinite(res.x)):
        return None
    mu_L, mu_O, log_sigma = (float(v) for v in res.x)
    sigma = float(np.exp(log_sigma))

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
    n = x.size
    H = np.zeros((n, n))
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


def summarize(df: pd.DataFrame, feedstock: dict[str, float],
              input_unit: str, display_unit: str,
              nonbinding_ec50: float | None,
              max_fit_ec50: float,
              target_group: tuple[str, ...]
              ) -> tuple[pd.DataFrame, float, dict]:
    """Per-protein weighted Tobit MLE. Returns (stats, ceiling, coverage_info)."""
    scale = UNIT_SCALE[input_unit] / UNIT_SCALE[display_unit]
    lcols, ocols = _partition_columns(df, target_group)
    if not lcols:
        raise ValueError(f"No target columns found (target group = {target_group})")
    if not ocols:
        raise ValueError("No contaminant columns found in the DSF table")

    w_L_raw = _column_weights(lcols, feedstock)
    w_O_raw = _column_weights(ocols, feedstock)

    # Normalize weights within each group so total weight = number of metals
    # in that group (keeps CI scale comparable to unweighted fit).
    def _normalize(w: np.ndarray) -> np.ndarray:
        s = w.sum()
        return w * (len(w) / s) if s > 0 else w

    w_L = _normalize(w_L_raw)
    w_O = _normalize(w_O_raw)

    tested_syms = {_strip_oxidation(c).upper() for c in lcols + ocols}
    feedstock_syms = set(feedstock)
    coverage = {
        "target_metals_used": [c for c, w in zip(lcols, w_L_raw) if w > 0],
        "target_metals_zero_weight": [c for c, w in zip(lcols, w_L_raw) if w == 0],
        "contam_metals_used": [c for c, w in zip(ocols, w_O_raw) if w > 0],
        "contam_metals_zero_weight": [c for c, w in zip(ocols, w_O_raw) if w == 0],
        "feedstock_missing_from_panel": sorted(feedstock_syms - tested_syms),
    }

    all_vals = df[lcols + ocols].apply(pd.to_numeric, errors="coerce") * scale
    all_vals = all_vals.where((all_vals > 0) & (all_vals <= max_fit_ec50))

    if nonbinding_ec50 is None:
        finite = all_vals.to_numpy().ravel()
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            raise ValueError("Input has no positive EC50 values")
        nonbinding_ec50 = float(np.nanmax(finite))
    log_ceiling = float(np.log(nonbinding_ec50))
    all_vals = all_vals.where(all_vals <= nonbinding_ec50)

    finite_pos = all_vals.to_numpy().ravel()
    finite_pos = finite_pos[np.isfinite(finite_pos) & (finite_pos > 0)]
    log_floor = float(np.log(finite_pos.min())) - np.log(100.0) if finite_pos.size else -50.0

    rows: list[dict] = []
    for i, protein in enumerate(df["Protein"].astype(str)):
        l_row = all_vals[lcols].iloc[i]
        o_row = all_vals[ocols].iloc[i]

        l_obs_mask = l_row.notna().to_numpy()
        o_obs_mask = o_row.notna().to_numpy()

        y_L = np.log(l_row.to_numpy(dtype=float)[l_obs_mask])
        y_O = np.log(o_row.to_numpy(dtype=float)[o_obs_mask])
        w_L_obs = w_L[l_obs_mask]
        w_O_obs = w_O[o_obs_mask]
        w_L_cens = float(w_L[~l_obs_mask].sum())
        w_O_cens = float(w_O[~o_obs_mask].sum())

        if (w_L_obs.sum() + w_L_cens) <= 0 or (w_O_obs.sum() + w_O_cens) <= 0:
            # No feedstock metals landed in this group (e.g. no contaminant
            # present in the feedstock overlapped the panel) -> skip.
            continue

        fit = _weighted_tobit_two_group(
            y_L, w_L_obs, w_L_cens,
            y_O, w_O_obs, w_O_cens,
            log_ceiling, log_floor)
        entry = {
            "Protein": protein,
            "n_target": int(l_obs_mask.sum()),
            "n_target_censored": int((~l_obs_mask).sum()),
            "n_contam": int(o_obs_mask.sum()),
            "n_contam_censored": int((~o_obs_mask).sum()),
            "w_target_obs": float(w_L_obs.sum()),
            "w_target_cens": w_L_cens,
            "w_contam_obs": float(w_O_obs.sum()),
            "w_contam_cens": w_O_cens,
        }
        if fit is None:
            mu_L = float(np.average(y_L, weights=w_L_obs)) if w_L_obs.sum() > 0 else log_ceiling
            mu_O = float(np.average(y_O, weights=w_O_obs)) if w_O_obs.sum() > 0 else log_ceiling
            entry.update({
                "mu_L": mu_L, "mu_O": mu_O, "sigma": float("nan"),
                "se_L": float("nan"), "se_O": float("nan"),
                "delta_log": mu_O - mu_L, "se_delta_log": float("nan"),
                "delta_ci_low": float("nan"), "delta_ci_high": float("nan"),
                "z": float("nan"), "p": float("nan"),
                "at_bound": True,
                "L_censored_only": bool(w_L_obs.sum() == 0),
                "O_censored_only": bool(w_O_obs.sum() == 0),
            })
        else:
            entry.update(fit)
            entry["L_censored_only"] = False
            entry["O_censored_only"] = False
        entry["mu_L_ec50"] = float(np.exp(entry["mu_L"])) if np.isfinite(entry["mu_L"]) else np.nan
        entry["mu_O_ec50"] = float(np.exp(entry["mu_O"])) if np.isfinite(entry["mu_O"]) else np.nan
        rows.append(entry)

    stats = pd.DataFrame(rows)
    if not stats.empty:
        stats = stats.sort_values("delta_log", ascending=False, kind="mergesort").reset_index(drop=True)
        stats.insert(0, "rank", np.arange(1, len(stats) + 1))
    return stats, nonbinding_ec50, coverage


def _print_coverage(coverage: dict) -> None:
    def _fmt(cols):
        return ", ".join(cols) if cols else "(none)"
    print("Feedstock coverage vs DSF panel:")
    print(f"  target metals with weight > 0: {_fmt(coverage['target_metals_used'])}")
    if coverage["target_metals_zero_weight"]:
        print(f"  target metals absent from feedstock (weight 0): "
              f"{_fmt(coverage['target_metals_zero_weight'])}")
    print(f"  contaminant metals with weight > 0: {_fmt(coverage['contam_metals_used'])}")
    if coverage["contam_metals_zero_weight"]:
        print(f"  contaminant metals absent from feedstock (weight 0): "
              f"{_fmt(coverage['contam_metals_zero_weight'])}")
    if coverage["feedstock_missing_from_panel"]:
        print(f"  feedstock species with no DSF column (ignored): "
              f"{_fmt(coverage['feedstock_missing_from_panel'])}",
              file=sys.stderr)
    print()


def _print_table(stats: pd.DataFrame, display_unit: str, threshold: float) -> None:
    if stats.empty:
        print("(no proteins scored -- check feedstock/panel overlap)")
        return
    log_thresh = float(np.log(threshold))
    cols = ["rank", "Protein",
            "mu_L_ec50", "mu_O_ec50",
            "delta_log", "delta_ci_low", "delta_ci_high", "p",
            "n_target", "n_target_censored",
            "n_contam", "n_contam_censored"]
    show = stats[cols].copy()
    show.insert(2, "selective",
                (show["delta_ci_low"] >= log_thresh).map({True: "yes", False: "no"}))
    formatters = {
        "mu_L_ec50": lambda v: f"{v:8.3g}",
        "mu_O_ec50": lambda v: f"{v:8.3g}",
        "delta_log": lambda v: f"{v:+.2f}",
        "delta_ci_low": lambda v: f"{v:+.2f}",
        "delta_ci_high": lambda v: f"{v:+.2f}",
        "p": lambda v: f"{v:.2g}" if np.isfinite(v) else "nan",
    }
    print(f"Feedstock-weighted Tobit ranking (log EC50 in {display_unit}), "
          f"selective at Delta >= log({threshold:g}) = {log_thresh:.2f}:")
    print(show.to_string(index=False, formatters=formatters))
    print()


def plot(stats: pd.DataFrame, threshold: float, display_unit: str,
         ceiling: float, feedstock_name: str) -> tuple[plt.Figure, plt.Axes]:
    if stats.empty:
        raise ValueError("No proteins to plot")
    log_thresh = float(np.log(threshold))

    # Forest plot: proteins ranked top -> bottom, Δ (log fold) with 95% CI.
    n = len(stats)
    height = max(PANEL_IN, 0.22 * n + 0.6)
    fig, ax = plt.subplots(figsize=(PANEL_IN * 1.2, height))

    y = np.arange(n)[::-1]  # rank 1 at top
    delta = stats["delta_log"].to_numpy(dtype=float)
    lo = stats["delta_ci_low"].to_numpy(dtype=float)
    hi = stats["delta_ci_high"].to_numpy(dtype=float)
    at_bound = stats["at_bound"].fillna(False).to_numpy(dtype=bool)

    # Cap CIs for display so extreme fits don't blow up x-scale.
    cap = np.log(1e4)
    lo_disp = np.where(np.isfinite(lo), np.maximum(lo, -cap), delta)
    hi_disp = np.where(np.isfinite(hi), np.minimum(hi, cap), delta)

    is_selective = np.where(np.isfinite(lo), lo >= log_thresh, False)

    colors = np.where(is_selective, "#1f77b4", "0.55")

    xerr_lo = np.where(np.isfinite(lo), delta - lo_disp, 0.0)
    xerr_hi = np.where(np.isfinite(hi), hi_disp - delta, 0.0)

    for i in range(n):
        marker = "s" if at_bound[i] else "o"
        ax.errorbar(delta[i], y[i],
                    xerr=[[xerr_lo[i]], [xerr_hi[i]]],
                    fmt=marker, color=colors[i], ecolor=colors[i],
                    markersize=4, markeredgewidth=0,
                    elinewidth=0.5, capsize=1.5, capthick=0.5, zorder=3)

    ax.axvline(0.0, color="0.7", linewidth=0.5, linestyle="-", zorder=1)
    ax.axvline(log_thresh, color="seagreen", linewidth=0.5,
               linestyle="--", zorder=1,
               label=f"{threshold:g}× target-selective")

    ax.set_yticks(y)
    ax.set_yticklabels(stats["Protein"].to_numpy(), fontsize=6, fontstyle="italic")
    ax.set_ylim(-0.5, n - 0.5)

    # x-axis: Δ = log fold. Show fold ticks on a symmetric log-like axis.
    all_finite = np.concatenate([delta, lo_disp, hi_disp])
    all_finite = all_finite[np.isfinite(all_finite)]
    if all_finite.size:
        xmin = float(min(all_finite.min(), -log_thresh * 0.2, -0.5))
        xmax = float(max(all_finite.max(), log_thresh * 1.2))
        pad = 0.15 * max(1.0, xmax - xmin)
        ax.set_xlim(xmin - pad, xmax + pad)

    # Redraw the shaded selective band now that xlim is finalized.
    xmin_final, xmax_final = ax.get_xlim()
    ax.axvspan(log_thresh, xmax_final, color="seagreen", alpha=0.06,
               linewidth=0, zorder=0)

    # Nice fold-based tick labels: -100x, -10x, 1x, 10x, 100x, 1000x...
    fold_ticks = [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
    log_ticks = [np.log(f) for f in fold_ticks
                 if np.log(f) >= xmin_final and np.log(f) <= xmax_final]
    tick_labels = []
    for lt in log_ticks:
        f = np.exp(lt)
        if f >= 1:
            tick_labels.append(f"{f:g}×")
        else:
            tick_labels.append(f"1/{1/f:g}×")
    ax.set_xticks(log_ticks)
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel("Contaminant / target composite EC50  (Δ = μ$_{contam}$ − μ$_{target}$)")
    ax.set_title(f"Feedstock-weighted REE selectivity: {feedstock_name}")
    ax.legend(loc="lower right", frameon=False, handlelength=1.5,
              borderaxespad=0.2, handletextpad=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(pad=0.3)
    return fig, ax


if __name__ == "__main__":
    main()
