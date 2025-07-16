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
    tm_df = plot_384well_data(df, os.path.dirname(args.file))
    out_csv = os.path.join(os.path.dirname(args.file), "384well_tm_values.csv")
    tm_df.to_csv(out_csv, index=False)

def parse_csv_file(filepath):
    # Find the first non-comment, non-empty line (header)
    with open(filepath, 'r') as f:
        lines = f.readlines()
    header_idx = None
    for i, line in enumerate(lines):
        if not line.startswith('#') and line.strip():
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find data table header in file.")
    # Now read the data table into a DataFrame
    df = pd.read_csv(filepath, skiprows=header_idx)
    df.columns = [c.strip() for c in df.columns]
    return df

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

def plot_384well_data(df, output_dir):
    wells = sorted(df['Well Position'].unique(), key=lambda x: (x[0], int(x[1:])))
    tm_dict = {"Well": [], "Tm": []}
    fig, axes = plt.subplots(16, 24, figsize=(24, 16))
    axes = axes.flatten()
    for i, well in enumerate(wells):
        well_df = df[df['Well Position'] == well]
        temp = well_df['Temperature'].values
        fluorescence = well_df['Fluorescence'].values
        derivative = -well_df['Derivative'].values
        tms = find_tm(temp, derivative)
        if tms is None:
            tms = 0
        tm_dict["Well"].append(well)
        tm_dict["Tm"].append(tms)
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
        ax.set_title(well+": "+f"{tms:.1f}", fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "384well_raw_and_derivative.png"), dpi=300)
    plt.show()
    return pd.DataFrame(tm_dict)

if __name__ == '__main__':
    main()