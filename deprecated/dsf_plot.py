#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.signal import find_peaks
from scipy.optimize import curve_fit 
import matplotlib.cm as cm
from functools import partial
from hillfit import HillFit

def main():
    parser = argparse.ArgumentParser(
        description="Plot raw fluorescence data and negative derivative data from CSV files in a directory, with Tm detection."
    )
    parser.add_argument('-d', '--directory', type=str, required=True, help="Path to the directory containing the data files")
    parser.add_argument('-o', '--output', type=str, default="tm_values.csv", help="Output CSV file for Tm values")
    args = parser.parse_args()

    raw_file, derivative_file = find_files(args.directory)
    raw_data = load_data(raw_file)
    derivative_data = load_data(derivative_file)

    fluorescence_columns = [f"{chr(i)}{j}" for i in range(65, 73) for j in range(1, 13)]
    tm_dict = {"Well": [], "Tm": []}

    fig, axes = plt.subplots(8, 12, figsize=(12, 8))
    axes = axes.flatten()

    for i, col in enumerate(fluorescence_columns):
        if col not in raw_data.columns or col not in derivative_data.columns:
            tm_dict["Well"].append(col)
            tm_dict["Tm"].append("Missing")
            continue

        raw_fluorescence = raw_data[col]
        derivative = -derivative_data[col]
        temperature = raw_data['Temperature']

        if col.endswith("12"):
            tms = "Blank"
        else:
            tms = find_tm(temperature, derivative)
            if tms is None:
                tms = 0

        tm_dict["Well"].append(col)
        tm_dict["Tm"].append(tms)

        ax = axes[i]
        ax2 = ax.twinx()

        ax.plot(temperature, raw_fluorescence, color='blue', label='Raw Data')
        ax2.plot(temperature, derivative, color='orange', label='Negative Derivative')

        if tms != "Blank" and tms is not None:
            ax.axvline(tms, color='red', linestyle='--')

        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax2.grid(False)
        ax2.set_xticks([])
        ax2.set_yticks([])

        row_label = i // 12
        col_label = i % 12

        if col_label == 0:
            ax.set_ylabel(f"{chr(65 + row_label)}", rotation=0, labelpad=10)
        if row_label == 7:
            ax.set_xlabel(f"{col_label + 1}", labelpad=10)

        text_content = f'{col}'
        if tms == "Blank":
            text_content += '\nBlank'
        elif tms is not None:
            text_content += f'\n{tms:.0f}'
        ax.text(0.95, 0.95, text_content, transform=ax.transAxes, ha='right', va='top', fontsize=8, color='black')

    plt.tight_layout()
    plt.show()

    tm_df = pd.DataFrame(tm_dict)
    tm_df.to_csv(args.output, index=False)
    plot_tm_scatter(tm_df)

def find_files(directory):
    raw_file = None
    derivative_file = None

    for file in os.listdir(directory):
        if file.endswith(".csv"):
            if "amplification" in file.lower():
                raw_file = os.path.join(directory, file)
            elif "derivative" in file.lower():
                derivative_file = os.path.join(directory, file)

    if not raw_file or not derivative_file:
        print("Error: Could not find both raw fluorescence and derivative files in the directory.")
        exit(1)

    return raw_file, derivative_file

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns=['Unnamed: 0'])
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        exit(1)

def find_tm(temperature, derivative):
    peak_height_mult = 0.5
    derivative_std = np.nanstd(derivative)
    height = peak_height_mult * derivative_std

    peaks, properties = find_peaks(derivative, height=None, distance=None, threshold=None, prominence=height)

    if len(peaks) == 0:
        return None

    most_prominent_index = np.argmax(properties['prominences'])
    most_prominent_peak = peaks[most_prominent_index]

    return temperature[most_prominent_peak]

def hill_eq(concentration, ymax=None, K=None, n=None, ymin=None):
    """
    Hill equation with either ymin or ymax fixed.
    """
    if ymin is not None:  # Binding goes up
        return ymin + (ymax - ymin) * (concentration**n) / (K**n + concentration**n)
    elif ymax is not None:  # Binding goes down
        return ymax - (ymax - ymin) * (concentration**n) / (K**n + concentration**n)

def plot_tm_scatter(tm_df):
    concentrations = [446, 89.2, 17.84, 3.57, 0.71]  # µM
    kd_summary = []
    metals = {
        "Li$^+$": ["A1", "A2", "A3", "A4", "A5"],
        "Na$^+$": ["B1", "B2", "B3", "B4", "B5"],
        "Mg$^{2+}$": ["C1", "C2", "C3", "C4", "C5"],
        "Al$^{3+}$": ["D1", "D2", "D3", "D4", "D5"],
        "K$^+$": ["E1", "E2", "E3", "E4", "E5"],
        "Ca$^{2+}$": ["F1", "F2", "F3", "F4", "F5"],
        "Mn$^{2+}$": ["G1", "G2", "G3", "G4", "G5"],
        "Fe$^{3+}$": ["H1", "H2", "H3", "H4", "H5"],
        "Co$^{2+}$": ["A6", "A7", "A8", "A9", "A10"],
        "Ni$^{2+}$": ["B6", "B7", "B8", "B9", "B10"],
        "Cu$^{2+}$": ["C6", "C7", "C8", "C9", "C10"],
        "Zn$^{2+}$": ["D6", "D7", "D8", "D9", "D10"],
        "La$^{3+}$": ["E6", "E7", "E8", "E9", "E10"],
        "Pr$^{3+}$": ["F6", "F7", "F8", "F9", "F10"],
        "Nd$^{3+}$": ["G6", "G7", "G8", "G9", "G10"],
        "HCl": ["H6", "H7", "H8", "H9", "H10"],
        "WT": ["A11", "B11", "C11", "D11"],
        "EDTA": ["E11", "F11", "G11", "H11"],
    }

    wt_tm_values = [row["Tm"] for _, row in tm_df.iterrows() if row["Well"] in metals["WT"] and pd.notna(row["Tm"]) and row["Tm"] != 0]
    edta_tm_values = [row["Tm"] for _, row in tm_df.iterrows() if row["Well"] in metals["EDTA"] and pd.notna(row["Tm"]) and row["Tm"] != 0]
    wt_avg_tm = np.mean(wt_tm_values) if wt_tm_values else None
    edta_avg_tm = np.mean(edta_tm_values) if edta_tm_values else None

    # Remove highest, if HCl effect
    hcl_wells = metals["HCl"]
    hcl_tm_values = [row["Tm"] for _, row in tm_df.iterrows() if row["Well"] in hcl_wells and pd.notna(row["Tm"])]
    if abs(hcl_tm_values[0] - sum(hcl_tm_values[1:]) / len(hcl_tm_values[1:])) > 1:
        concentrations = concentrations[1:]
        for metal, wells in metals.items():
            if metal not in ["WT", "EDTA", "HCl"]:
                metals[metal] = wells[1:]

    fig, ax = plt.subplots(figsize=(10, 6))
    kd_summary = []

    for metal, wells in metals.items():
        if metal in ["WT", "EDTA", "HCl"]: 
            continue

        tm_values = [row["Tm"] for _, row in tm_df.iterrows() if row["Well"] in wells and pd.notna(row["Tm"])]
        filtered_concentrations = [conc for conc, tm in zip(concentrations, tm_values) if tm != 0]
        filtered_tm_values = [tm for tm in tm_values if tm != 0]

        # Plot scatter points
        ax.scatter(filtered_concentrations, filtered_tm_values, label=metal)
        if len(filtered_tm_values) >= 4:  # Ensure at least 4 data points for fitting
            filtered_tm_values = np.array(filtered_tm_values)
            try:
                # Try fitting with ymin = wt_avg_tm (binding goes up)
                try:
                    popt, _ = curve_fit(
                        lambda concentration, ymax, K, n: hill_eq(concentration, ymin=wt_avg_tm, ymax=ymax, K=K, n=n),
                        filtered_concentrations,
                        filtered_tm_values,
                        p0=[max(filtered_tm_values), np.median(filtered_concentrations), 1.0]  # Initial guesses
                    )
                    ymax, K, n = popt
                    ymin = wt_avg_tm  # Assign ymin explicitly
                    delta_tm = ymax - wt_avg_tm
                except RuntimeError:
                    print("trying negative for "+metal)
                    # If the first fit fails, try fitting with ymax = wt_avg_tm (binding goes down)
                    popt, _ = curve_fit(
                        lambda concentration, ymin, K, n: hill_eq(concentration, ymin=ymin, ymax=wt_avg_tm, K=K, n=n),
                        filtered_concentrations,
                        filtered_tm_values,
                        p0=[min(filtered_tm_values), np.median(filtered_concentrations), -1.0]  # Initial guesses
                    )
                    ymin, K, n = popt
                    ymax = wt_avg_tm  # Assign ymax explicitly
                    delta_tm = wt_avg_tm - ymin

                # Generate the fit curve
                fit_x = np.logspace(np.log10(min(filtered_concentrations)), np.log10(max(filtered_concentrations)), 100)
                fit_y = hill_eq(fit_x, ymin=ymin, ymax=ymax, K=K, n=n)
                ax.plot(fit_x, fit_y, color='gray')  # No legend entry for fit line
                kd_summary.append(f"{metal}: Kd = {K:.2f} µM, ΔTm = {delta_tm:.2f} °C")
            except RuntimeError:
                print(f"Could not fit Hill equation for {metal}")
                kd_summary.append(f"{metal}: N.B.")  # No Binding
        else:
            print(f"Not enough data points to fit Hill equation for {metal}")
            kd_summary.append(f"{metal}: N.B.")  # No Binding


    ax.axhline(wt_avg_tm, color='red', linestyle='-', label="WT")
    kd_summary.append(f"WT: {wt_avg_tm:.2f} °C")
    ax.axhline(edta_avg_tm, color='black', linestyle='-', label="EDTA")
    kd_summary.append(f"EDTA: {edta_avg_tm:.2f} °C")
    ax.set_xlabel("Concentration (µM)")
    ax.set_ylabel("Tm (°C)")
    ax.set_title("Tm Scatter Plot with Hill Equation Fit")

    summary_text = "\n".join(kd_summary)
    ax.text(1.3, 0.5, summary_text, transform=ax.transAxes, fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"))
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()