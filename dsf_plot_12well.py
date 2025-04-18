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
from matplotlib.patches import Patch
from matplotlib.colors import Normalize

def main():
    parser = argparse.ArgumentParser(
        description="Plot raw fluorescence data and negative derivative data from CSV files in a directory, with Tm detection."
    )
    parser.add_argument('-d', '--directory', type=str, required=True, help="Path to the directory containing the data files")
    parser.add_argument('-o', '--output', type=str, default="tm_values.csv", help="Output CSV file for Tm values")
    parser.add_argument('-m', '--metals', type=str, required=True, help="trans or lans")
    parser.add_argument('-e', '--exclude_high', type=bool, default=False, help="Exclude highest point")
    args = parser.parse_args()

    global directory
    directory = args.directory
    raw_file, derivative_file = find_files(directory)
    raw_data = load_data(raw_file)
    derivative_data = load_data(derivative_file)
    metals, concentrations = metals_and_concentrations(args.metals)
    tm_df = plot_data(raw_data, derivative_data)
    tm_df.to_csv(args.output, index=False)
    metals = plot_tm_scatter(tm_df, metals, concentrations, args.exclude_high)
    plot_tm_bar(metals)

def metals_and_concentrations(metal_type):
    if metal_type == "trans":
        metals = {
            "La$^{3+}$": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11"],
            "Ce$^{3+}$": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"],
            "Pr$^{3+}$": ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11"],
            "Nd$^{3+}$": ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11"],
            "Eu${^3+}$": ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11"],
            "Gd$^{3+}$": ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11"],
            "Tb$^{3+}$": ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8", "G9", "G10", "G11"],
            "HCl": ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11"],
            "WT": ["A12", "B12", "C12"],
            "EDTA": ["D12", "E12", "F12"],
            "Blank": ["G12", "H12"]}
        concentrations = [1000, 500, 250, 125, 62.5, 12.5, 6.25, 3.25, 1.625, 0.8125, 0.40625]  # µM
    return metals, concentrations

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
    peak_height_mult = 0.25
    derivative_std = np.nanstd(derivative)
    height = peak_height_mult * derivative_std
    peaks, properties = find_peaks(derivative, height=None, distance=None, threshold=None, prominence=height)
    if len(peaks) == 0:
        return None
    most_prominent_index = np.argmax(properties['prominences'])
    most_prominent_peak = peaks[most_prominent_index]
    return temperature[most_prominent_peak]

def plot_data(raw_data, derivative_data):
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
    plt.savefig(directory+"/raw_and_derivate.png", dpi=300)
    tm_df = pd.DataFrame(tm_dict)
    return tm_df

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

def hill_eq(concentration, ymin, ymax, K, n):
    return ymin + (ymax - ymin) * (concentration**n) / (K**n + concentration**n)

def hcl_effect(tm_df, metals, concentrations, exclude_high):
    print("HCL effect")
    hcl_wells = metals["HCl"]
    hcl_tm_values = [row["Tm"] for _, row in tm_df.iterrows() if row["Well"] in hcl_wells and pd.notna(row["Tm"])]
    if exclude_high is False:
        for well in hcl_wells:
            print(well)
            if abs(hcl_tm_values[0] - sum(hcl_tm_values[1:]) / len(hcl_tm_values[1:])) > 1:
                concentrations = concentrations[1:]
                for metal, wells in metals.items():
                    if metal not in ["WT", "EDTA", "HCl"]:
                        metals[metal] = wells[1:]
                print("Excluding highest point")
    elif exclude_high is True:
        concentrations = concentrations[1:]
        for metal, wells in metals.items():
            if metal not in ["WT", "EDTA", "HCl"]:
                metals[metal] = wells[1:]

    return metals, concentrations

def fit_hill(metals, metal, filtered_concentrations, filtered_tm_values, ax, kd_summary, wt_avg_tm):
    if len(filtered_tm_values) >= 4:
        filtered_tm_values = np.array(filtered_tm_values)
        try:
            # if abs(wt_avg_tm - np.average(filtered_tm_values)) > 1:
            #     if (wt_avg_tm - np.average(filtered_tm_values)) > 1:
            #         p0 = [wt_avg_tm, max(filtered_tm_values), np.median(filtered_concentrations), 0]
            #         bounds = ([wt_avg_tm - 1, 0, 0, -1], [wt_avg_tm + 1, 125, 1000, 1])
            #         popt, _ = curve_fit(lambda concentration, ymin, ymax, K, n: hill_eq(concentration, ymin, ymax, K, n),
            #             filtered_concentrations,
            #             filtered_tm_values,
            #             p0=p0,
            #             bounds=bounds)
            #         ymin, ymax, K, n = popt
            #         delta_tm = ymax - wt_avg_tm
            #     else:
            #         p0 = [min(filtered_tm_values), wt_avg_tm, np.median(filtered_concentrations), 0]
            #         bounds = ([0, wt_avg_tm - 1, 0, -1], [125, wt_avg_tm + 1, 1000, 1])
            #         popt, _ = curve_fit(lambda concentration, ymin, ymax, K, n: hill_eq(concentration, ymin, ymax, K, n),
            #             filtered_concentrations,
            #             filtered_tm_values,
            #             p0=p0,
            #             bounds=bounds)
            #         ymin, ymax, K, n = popt
            #         delta_tm = ymin - wt_avg_tm

            p0 = [min(filtered_tm_values), max(filtered_tm_values), np.median(filtered_concentrations), 0]
            bounds = ([0, 0, 0, -1], [100, 100, 1000, 1])
            popt, _ = curve_fit(lambda concentration, ymin, ymax, K, n: hill_eq(concentration, ymin, ymax, K, n),
                filtered_concentrations,
                filtered_tm_values,
                p0=p0,
                bounds=bounds)
            ymin, ymax, K, n = popt
            delta_tm = filtered_tm_values[0] - wt_avg_tm
            
            fit_x = np.logspace(np.log10(min(filtered_concentrations)), np.log10(max(filtered_concentrations)), 100)
            fit_y = hill_eq(fit_x, ymin, ymax, K, n)
            ax.plot(fit_x, fit_y, color='gray') 
            
            if abs(delta_tm) > 1:
                kd_summary.append(f"{metal}: Kd = {K:.2f} µM, ΔTm = {delta_tm:.2f} °C")
                metals[metal].extend([round(K, 2), round(delta_tm, 2)])
            else:
                kd_summary.append(f"{metal}: N.B.")
                metals[metal].extend(["N.B.", 0])
        except RuntimeError:
            print(f"Could not fit Hill equation for {metal}")
            kd_summary.append(f"{metal}: N.B.")
            metals[metal].extend(["N.B.", 0])
    else:
        print(f"Not enough data points to fit Hill equation for {metal}")
        kd_summary.append(f"{metal}: N.B.")
        metals[metal].extend(["N.B.", 0])
    return metals, ax, kd_summary

def plot_tm_scatter(tm_df, metals, concentrations, exclude_high):
    kd_summary = []
    wt_tm_values = [row["Tm"] for _, row in tm_df.iterrows() if row["Well"] in metals["WT"] and pd.notna(row["Tm"]) and row["Tm"] != 0]
    edta_tm_values = [row["Tm"] for _, row in tm_df.iterrows() if row["Well"] in metals["EDTA"] and pd.notna(row["Tm"]) and row["Tm"] != 0]
    wt_avg_tm = np.mean(wt_tm_values) if wt_tm_values else None
    edta_avg_tm = np.mean(edta_tm_values) if edta_tm_values else None
    metals, concentrations = hcl_effect(tm_df, metals, concentrations, exclude_high)
    fig, ax = plt.subplots(figsize=(10, 6))
    kd_summary = []
    
    for metal, wells in metals.items():
        if metal in ["WT", "EDTA", "HCl"]: 
            continue

        tm_values = [row["Tm"] for _, row in tm_df.iterrows() if row["Well"] in wells and pd.notna(row["Tm"])]
        filtered_concentrations = [conc for conc, tm in zip(concentrations, tm_values) if tm != 0]
        filtered_tm_values = [tm for tm in tm_values if tm != 0]
        ax.scatter(filtered_concentrations, filtered_tm_values, label=metal)
        metals, ax, kd_summary = fit_hill(metals, metal, filtered_concentrations, filtered_tm_values, ax, kd_summary, wt_avg_tm)

    ax.axhline(wt_avg_tm, color='red', linestyle='-', label="WT")
    kd_summary.append(f"WT: {wt_avg_tm:.2f} °C")
    ax.axhline(edta_avg_tm, color='black', linestyle='-', label="EDTA")
    kd_summary.append(f"EDTA: {edta_avg_tm:.2f} °C")
    ax.set_xlabel("Concentration (µM)")
    ax.set_ylabel("Tm (°C)")
    ax.set_ylim(25, 100)
    ax.set_title("Tm Scatter Plot with Hill Equation Fit")
    summary_text = "\n".join(kd_summary)
    ax.text(1.3, 0.5, summary_text, transform=ax.transAxes, fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"))
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.savefig(directory+"/hill_fit.png", dpi=300)
    return metals

def plot_tm_bar(metals_data):
    metals = []
    Kd_values = [] 
    ΔTm = [] 
    for metal in metals_data:
        if metal in ["WT", "EDTA", "HCl"]: 
            continue
        metals.append(metal)
        try:
            Kd_values.append(float(metals_data[metal][-2]))
        except ValueError:
            Kd_values.append(10000)
        ΔTm.append(float(metals_data[metal][-1]))

    inv_Kd = [1/kd for kd in Kd_values]
    norm = Normalize(vmin=-10, vmax=10)
    colors = [plt.cm.coolwarm(norm(tm)) for tm in ΔTm]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(metals, inv_Kd, color=colors, edgecolor='black')
    ax.set_ylabel('Binding Affinity (1/Kd)', fontsize=12)
    ax.set_yscale('log')
    # ax.set_ylim(0, 1)
    ax.set_xlabel('Metal Ion', fontsize=12)
    ax.set_title('Metal Binding Affinities and Stability Effects', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, kd in zip(bars, Kd_values):
        if kd < 10000:
            height = bar.get_height()
            if height < 0.02:
                new_height = 0.05
            elif height < 0.5:
                new_height = height + 0.05
            else: 
                new_height = height - 0.4
            ax.text(bar.get_x() + bar.get_width()/2, new_height,
                f'{kd} μM', ha='center', va='bottom', fontsize=10, rotation=90)

    # Add ΔTm color legend
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([]) 
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('ΔTm (°C)', fontsize=12)

    plt.savefig(directory+"/kd_bar.png", dpi=300)

if __name__ == '__main__':
    main()