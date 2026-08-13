#!/usr/bin/env python3
"""
dsf_apo_tm.py  –  Find the melting temperature for every well in a DSF plate.

For each well the script:
  • plots the raw fluorescence curve
  • overlays a dashed vertical line at the Tm
  • annotates with the well ID (e.g. A1, O15) and the Tm value

Exports:  <output>_per_well_tm.csv   (index = Well ID, column = Tm_C)
          <output>_per_well_tm.pdf   (plate-layout grid of melt curves)
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['pdf.use14corefonts'] = False
matplotlib.rcParams['savefig.transparent'] = True
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter


ROWS = list("ABCDEFGHIJKLMNOP")  # 16 rows for a 384-well plate


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute and plot per-well melting temperatures from a DSF plate.")
    parser.add_argument('-c', '--csv', type=str, required=True, help="Raw DSF CSV from the DA2 instrument")
    parser.add_argument('-o', '--output', type=str, default=None, help="Output file stem (default: CSV filename without extension)")
    parser.add_argument('-ht', '--high_temp', type=float, default=None, help="Exclude temperatures above this value (°C)")
    parser.add_argument('-lt', '--low_temp', type=float, default=None, help="Exclude temperatures below this value (°C)")
    parser.add_argument('-w', '--exclude_wells', type=str, nargs='+', default=[], help="Well positions to exclude (e.g. A1 B3 C12)")
    args = parser.parse_args()

    import os
    stem = args.output if args.output else os.path.splitext(os.path.basename(args.csv))[0]

    df = parse_csv(args.csv)

    if args.exclude_wells:
        excluded = [w.strip().upper() for w in args.exclude_wells]
        df = df[~df['Well Position'].str.upper().isin(excluded)]

    if args.high_temp is not None:
        df = df[df['Temperature'] <= args.high_temp]
    if args.low_temp is not None:
        df = df[df['Temperature'] >= args.low_temp]

    tm_df = compute_tms(df)
    plot_wells(df, tm_df, stem)
    save_csv(tm_df, stem)


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def parse_csv(path):
    """Read a DA2-style DSF CSV, skipping comment lines that start with '#'."""
    with open(path, 'r') as f:
        lines = f.readlines()
    header_idx = None
    for i, line in enumerate(lines):
        if not line.startswith('#') and line.strip():
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not find data table header in: {path}")
    df = pd.read_csv(path, skiprows=header_idx)
    df.columns = [c.strip() for c in df.columns]
    if 'Target' in df.columns:
        df = df.drop(columns=['Target'])
    return df


# ---------------------------------------------------------------------------
# Tm calculation
# ---------------------------------------------------------------------------

def _smooth(arr, wl=51, poly=3):
    """Savitzky-Golay smooth; returns arr unchanged if too short."""
    wl = min(len(arr), wl)
    if wl % 2 == 0:
        wl -= 1
    if wl < 5:
        return arr.astype(float)
    return savgol_filter(arr.astype(float), wl, poly)


def find_tm(temps, fluor, deriv=None):
    """
    Return the melting temperature as the peak of the smoothed negative derivative.

    Parameters
    ----------
    temps : array-like  Temperature values (°C)
    fluor : array-like  Raw fluorescence values
    deriv : array-like or None
        Pre-computed derivative column from the instrument.  When present it is
        preferred over the numerical gradient of the smoothed fluorescence.
    """
    temps = np.asarray(temps, dtype=float)
    fluor = np.asarray(fluor, dtype=float)

    smooth_f = _smooth(fluor)

    if deriv is not None:
        # Instrument derivative: negate so the melt peak is a positive maximum
        smooth_d = -_smooth(np.asarray(deriv, dtype=float))
    else:
        smooth_d = -np.gradient(smooth_f, temps)

    global_max = smooth_d.max()
    if global_max <= 0:
        # Fallback: midpoint of the fluorescence transition
        return float(temps[np.argmax(smooth_f)])

    peaks, _ = find_peaks(smooth_d, height=global_max * 0.15, distance=5)
    if len(peaks) > 0:
        return float(temps[peaks[0]])
    return float(temps[np.argmax(smooth_d)])


def compute_tms(df):
    """Return a DataFrame with columns [Well_ID, Tm_C] sorted in plate order."""
    records = []
    deriv_col = 'Derivative' if 'Derivative' in df.columns else None
    for well_pos, group in df.groupby('Well Position'):
        group = group.sort_values('Temperature')
        temps = group['Temperature'].values
        fluor = group['Fluorescence'].values
        deriv = group[deriv_col].values if deriv_col else None
        tm = find_tm(temps, fluor, deriv)
        records.append({'Well_ID': well_pos, 'Tm_C': round(float(tm), 2)})

    tm_df = pd.DataFrame(records)
    tm_df['_sort'] = tm_df['Well_ID'].apply(_well_sort_key)
    tm_df = tm_df.sort_values('_sort').drop(columns='_sort').reset_index(drop=True)
    return tm_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _well_sort_key(well_id):
    """Sort key that orders wells A1, A2, … A24, B1, … P24."""
    row = well_id[0].upper()
    col = int(well_id[1:])
    return (ROWS.index(row) if row in ROWS else 99, col)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_wells(df, tm_df, stem):
    """
    Draw each well as a small subplot arranged in plate-layout order.
    Each subplot shows:
      - the raw fluorescence curve (blue)
      - a dashed vertical line at the Tm (red)
      - an annotation with the well ID and Tm value
    """
    all_wells = sorted(df['Well Position'].unique(), key=_well_sort_key)

    rows_in_data = sorted({w[0] for w in all_wells},
                          key=lambda r: ROWS.index(r) if r in ROWS else 99)
    cols_in_data = sorted({int(w[1:]) for w in all_wells})

    row_idx = {r: i for i, r in enumerate(rows_in_data)}
    col_idx = {c: i for i, c in enumerate(cols_in_data)}
    n_plot_rows = len(rows_in_data)
    n_plot_cols = len(cols_in_data)

    # Aim for roughly 0.9 × 0.72 inches per cell
    cell_w, cell_h = 0.9, 0.72
    fig_w = max(6.0, n_plot_cols * cell_w + 0.9)
    fig_h = max(4.0, n_plot_rows * cell_h + 0.7)

    fig, axes = plt.subplots(n_plot_rows, n_plot_cols,
                             figsize=(fig_w, fig_h),
                             squeeze=False)

    tm_lookup = dict(zip(tm_df['Well_ID'], tm_df['Tm_C']))

    # Shared y-axis limits across all wells for visual consistency
    y_min = np.nanmin(df['Fluorescence'].values)
    y_max = np.nanmax(df['Fluorescence'].values)
    y_pad = (y_max - y_min) * 0.05

    well_set = set(df['Well Position'].unique())

    for ri, row_letter in enumerate(rows_in_data):
        for ci, col_num in enumerate(cols_in_data):
            ax = axes[ri][ci]
            well_pos = row_letter + str(col_num)

            if well_pos not in well_set:
                ax.axis('off')
                continue

            well_data = (df[df['Well Position'] == well_pos]
                         .sort_values('Temperature'))
            temps = well_data['Temperature'].values
            fluor = well_data['Fluorescence'].values
            tm = tm_lookup.get(well_pos, np.nan)

            ax.plot(temps, fluor, color='steelblue', linewidth=0.6, alpha=0.85)

            if not np.isnan(tm):
                ax.axvline(tm, color='crimson', linestyle='--',
                           linewidth=0.9, alpha=0.9)

            ax.set_xlim(temps.min(), temps.max())
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_linewidth(0.4)

            tm_str = f"{tm:.1f}°C" if not np.isnan(tm) else "N/A"
            label = f"{well_pos}\n{tm_str}"
            ax.text(0.04, 0.97, label,
                    transform=ax.transAxes,
                    fontsize=3.5, va='top', ha='left', color='black',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white',
                              alpha=0.65, ec='none', linewidth=0))

    fig.suptitle(f'{stem} — per-well melting temperatures',
                 fontsize=9, y=1.001)
    fig.supxlabel('Temperature (°C)', fontsize=7, y=0.002)
    fig.supylabel('Fluorescence', fontsize=7, x=0.002)
    plt.tight_layout(pad=0.3, h_pad=0.15, w_pad=0.15)

    out_pdf = f'{stem}_per_well_tm.pdf'
    plt.savefig(out_pdf, bbox_inches='tight', backend='pdf')
    plt.show()
    print(f"Plot saved → {out_pdf}")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_csv(tm_df, stem):
    out_csv = f'{stem}_per_well_tm.csv'
    out = tm_df.set_index('Well_ID')
    out.to_csv(out_csv)
    print(f"CSV saved  → {out_csv}")
    print(out.to_string())





# ---------------------------------------------------------------------------

if __name__ == '__main__':
    main()
