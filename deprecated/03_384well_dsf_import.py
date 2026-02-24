#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Import and plot 384-well DSF data from tab-separated .csv export")
    parser.add_argument('-f', '--file', type=str, required=True, help="Path to .csv")
    args = parser.parse_args()

    df = parse_csv_file(args.file)
    metal_df = plot_384well_data(df, os.path.dirname(args.file))
    out_csv = os.path.join(os.path.dirname(args.file), "384well_tm_values.csv")
    metal_df.to_csv(out_csv)

def parse_csv_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    header_idx = None
    for i, line in enumerate(lines):
        if not line.startswith('#') and line.strip():
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find data table header in file.")
    df = pd.read_csv(filepath, skiprows=header_idx)
    df.columns = [c.strip() for c in df.columns]
    return df

def metals_and_concentrations():
    metal_symbols = [
        "Al³⁺", "K⁺", "Ca²⁺", "Sc³⁺", "V³⁺", "Cr³⁺", "Mn²⁺", "Fe³⁺", "Co²⁺", "Ni²⁺", "Cu²⁺", "Zn²⁺", "Y³⁺",
        "Ag⁺", "La³⁺", "Ce³⁺", "Pr³⁺", "Nd³⁺", "Sm³⁺", "Eu³⁺", "Gd³⁺", "Tb³⁺", "Dy³⁺", "Ho³⁺", "Er³⁺", "Tm³⁺", "Yb³⁺", "Lu³⁺", "Pt⁴⁺", "Au³⁺", "Acid"
    ]
    concentrations = [1000, 250, 62.5, 15.625, 3.90625, 0.9765625, 0.244140625, 0.061035156, 0.015258789, 0.003814697, 0.000953674, 0.000238419]
    index = [str(c) for c in concentrations] + ['WT', 'EDTA', 'Blank']
    metal_df = pd.DataFrame(index=index, columns=metal_symbols)

    rows = [chr(i) for i in range(ord('A'), ord('P')+1)]  # A to P
    for m_idx, metal in enumerate(metal_symbols[:16]):
        row = rows[m_idx]
        for c_idx, conc in enumerate(concentrations):
            well = f"{row}{c_idx+1}"
            metal_df.loc[str(conc), metal] = well

    for m_idx, metal in enumerate(metal_symbols[16:31]):
        row = rows[m_idx]
        for c_idx, conc in enumerate(concentrations):
            well = f"{row}{c_idx+13}"
            metal_df.loc[str(conc), metal] = well

    metal_df.loc['WT', :] = [None]*len(metal_symbols)
    metal_df.loc['EDTA', :] = [None]*len(metal_symbols)
    metal_df.loc['Blank', :] = [None]*len(metal_symbols)
    for i, ctrl in enumerate(['WT', 'EDTA', 'Blank']):
        for j in range(4):
            well = f"P{13 + i*4 + j}"
            metal_df.loc[ctrl, metal_symbols[j]] = well
    
    print(metal_df)
    return metal_df

def find_tm(temperature, derivative):
    peak_height_mult = 0.25
    derivative_std = np.nanstd(derivative)
    height = peak_height_mult * derivative_std
    peaks, properties = find_peaks(derivative, height=None, distance=None, threshold=None, prominence=height)
    if len(peaks) == 0:
        return None
    most_prominent_index = np.argmax(properties['prominences'])
    most_prominent_peak = peaks[most_prominent_index]
    return temperature[most_prominent_peak]

def plot_384well_data(df, directory):
    metal_df = metals_and_concentrations()
    wells = [f"{chr(row)}{col}" for row in range(65, 81) for col in range(1, 25)]  # A1-P24 for 384 wells
    tm_records = []
    fig, axes = plt.subplots(16, 24, figsize=(24, 16))
    axes = axes.flatten()

    well_to_tm = {}

    for i, well in enumerate(wells):
        well_data = df[df['Well Position'] == well]
        if well_data.empty:
            continue
        temp = well_data['Temperature'].astype(float).values
        fluorescence = well_data['Fluorescence'].astype(float).values
        derivative = -well_data['Derivative'].astype(float).values
        tms = find_tm(temp, derivative)
        if tms is None:
            tms = 0

        metal = ""
        concentration = ""
        conc_numeric = ""
        row_idx, col_idx = np.where(metal_df == well)
        if len(row_idx) and len(col_idx) > 0:
            metal = metal_df.columns[col_idx[0]]
            conc_str = metal_df.index[row_idx[0]]
            try:
                conc_val = float(conc_str)
                conc_numeric = conc_val  # always µM for CSV
                if conc_val < 1:
                    concentration = f"{conc_val*1000:.1f} nM"
                else:
                    concentration = f"{conc_val:.1f} µM"
            except ValueError:
                concentration = conc_str
                conc_numeric = conc_str
            metal_df.iloc[row_idx[0], col_idx[0]] = tms
        else:
            metal = "Blank"
            concentration = ""
            conc_numeric = ""

        tm_records.append({
            "Well": well,
            "Metal": metal,
            "Concentration": concentration,   # for display
            "Conc_numeric": conc_numeric,     # for CSV index
            "Tm": tms
        })

        ax = axes[i]
        ax2 = ax.twinx()
        ax.plot(temp, fluorescence, color='blue', label='Raw')
        ax2.plot(temp, derivative, color='orange', label='Neg Deriv')
        if tms:
            ax.axvline(tms, color='red', linestyle='--')

        ax.set_xticks([])
        ax.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax.set_title(well, fontsize=6)

        text_content = f"{metal}\n{concentration}\n{tms:.0f}" if metal != "Blank" else "Blank"
        ax2.text(
            0.95, 0.95, text_content,
            transform=ax.transAxes,
            ha='right', va='top',
            fontsize=8, color='black',
            zorder=4)

    plt.tight_layout()
    plt.savefig(os.path.join(directory, "384well_raw_and_derivative.png"), dpi=300)
    print(metal_df)
    return metal_df

if __name__ == '__main__':
    main()