#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.signal import find_peaks
from scipy.optimize import curve_fit 
from matplotlib.colors import Normalize

def main():
    parser = argparse.ArgumentParser(
        description="Plot raw fluorescence data and negative derivative data from CSV files in a directory, with Tm detection."
    )
    parser.add_argument('-c', '--csvs', type=str, nargs='+', required=True, help="List of csvs from 01_dsf_import.py")
    parser.add_argument('-e', '--exclude', type=int, default=0, help="Exclude highests point")
    parser.add_argument('-t', '--title', type=str, required=True, help="Title")
    args = parser.parse_args()

    metal_df = load_data(args.csvs)
    metal_df = plot_tm_scatter(metal_df, args.exclude_high)
    # plot_tm_bar(metal_df)

def load_data(csv_files):
    df_list = []
    for csv_file in csv_files:
        temp_df = pd.read_csv(csv_file, index_col=0)
        df_list.append(temp_df)
    metal_df = pd.concat(df_list, ignore_index=False, axis=1)
    return metal_df

def hill_eq(concentration, ymin, ymax, K, n):
    return ymin + (ymax - ymin) * (concentration**n) / (K**n + concentration**n)

def hcl_effect(tm_df, metals, concentrations, exclude_high):
    concentrations = concentrations[3:]
    for metal, wells in metals.items():
        if metal not in ["WT", "EDTA", "HCl", "Blank"]:
            metals[metal] = wells[3:]
    return metals, concentrations

def fit_hill(metals, metal, filtered_concentrations, filtered_tm_values, ax, kd_summary, wt_avg_tm):
    filtered_tm_values = np.array(filtered_tm_values)
    try:
        if (wt_avg_tm - np.average(filtered_tm_values)) < 0:
            p0 = [wt_avg_tm, max(filtered_tm_values), np.median(filtered_concentrations), 0]
            bounds = ([wt_avg_tm - 5, 0, 0, -1], [wt_avg_tm + 5, 125, np.inf, 1])
            popt, _ = curve_fit(lambda concentration, ymin, ymax, K, n: hill_eq(concentration, ymin, ymax, K, n),
                                filtered_concentrations,
                                filtered_tm_values,
                                p0=p0,
                                bounds=bounds)
            ymin, ymax, K, n = popt
            delta_tm = ymax - wt_avg_tm

        else:
            p0 = [min(filtered_tm_values), wt_avg_tm, np.median(filtered_concentrations), 0]
            bounds = ([0, wt_avg_tm - 5, 0, -1], [125, wt_avg_tm + 5, np.inf, 1])
            popt, _ = curve_fit(lambda concentration, ymin, ymax, K, n: hill_eq(concentration, ymin, ymax, K, n),
                                filtered_concentrations,
                                filtered_tm_values,
                                p0=p0,
                                bounds=bounds)
            ymin, ymax, K, n = popt
            delta_tm = ymin - wt_avg_tm
            print(metal, ymin)

        # Generate the fit curve
        fit_x = np.logspace(np.log10(min(filtered_concentrations)), np.log10(max(filtered_concentrations)), 100)
        fit_y = hill_eq(fit_x, ymin, ymax, K, n)
        ax.plot(fit_x, fit_y, color='gray')

        # Calculate R²
        predicted_tm_values = hill_eq(filtered_concentrations, ymin, ymax, K, n)
        ss_res = np.sum((filtered_tm_values - predicted_tm_values) ** 2)
        ss_tot = np.sum((filtered_tm_values - np.mean(filtered_tm_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        kd_summary.append(f"{metal}: Kd={K:.2f}µM, ΔTm={delta_tm:.2f}°C, R²={r_squared:.2f}")
        metals[metal].extend([round(K, 2), round(delta_tm, 2), round(r_squared, 2)])

    except RuntimeError:
        print(f"Could not fit Hill equation for {metal}")
        kd_summary.append(f"{metal}: N.B.")
        metals[metal].extend(["N.B.", 0, 0])
    return metals, ax, kd_summary

def plot_tm_scatter(metal_df, exclude_high):
    kd_summary = []
    wt_tm_values = [row["Tm"] for _, row in tm_df.iterrows() if row["Well"] in metals["WT"] and pd.notna(row["Tm"]) and row["Tm"] != 0]
    edta_tm_values = [row["Tm"] for _, row in tm_df.iterrows() if row["Well"] in metals["EDTA"] and pd.notna(row["Tm"]) and row["Tm"] != 0]
    wt_avg_tm = np.mean(wt_tm_values)
    edta_avg_tm = np.mean(edta_tm_values)
    metals, concentrations = hcl_effect(tm_df, metals, concentrations, exclude_high)
    fig, ax = plt.subplots(figsize=(10, 6))
    kd_summary = []
    
    for metal, wells in metals.items():
        if metal in ["WT", "EDTA", "HCl", "Blank"]: 
            continue

        tm_values = [row["Tm"] for _, row in tm_df.iterrows() if row["Well"] in wells and pd.notna(row["Tm"])]
        filtered_concentrations = [conc for conc, tm in zip(concentrations, tm_values) if tm != 0]
        filtered_tm_values = [tm for tm in tm_values if tm != 0]
        ax.scatter(filtered_concentrations, filtered_tm_values, label=metal)
        metals, ax, kd_summary = fit_hill(metals, metal, filtered_concentrations, filtered_tm_values, ax, kd_summary, wt_avg_tm)

    ax.axhline(wt_avg_tm, color='red', linestyle='-', label="WT")
    kd_summary.append(f"WT: {wt_avg_tm:.2f}°C")
    ax.axhline(edta_avg_tm, color='black', linestyle='-', label="EDTA")
    kd_summary.append(f"EDTA: {edta_avg_tm:.2f}°C")
    ax.set_xlabel("Concentration (µM)")
    ax.set_ylabel("Tm (°C)")
    # ax.set_ylim(25, 100)
    ax.set_title("Tm Scatter Plot with Hill Equation Fit")
    summary_text = "\n".join(kd_summary)
    ax.text(1.5, 0.5, summary_text, transform=ax.transAxes, fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"))
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.savefig("hill_fit.png", dpi=300)
    return metal_df

def plot_tm_bar(metals_data):
    metals = []
    Kd_values = [] 
    ΔTm = [] 
    for metal in metals_data:
        if metal in ["WT", "EDTA", "HCl", "Blank"]: 
            continue
        metals.append(metal)
        if metals_data[metal][-1] > 0.5:
            Kd_values.append(float(metals_data[metal][-3]))
        else:
            Kd_values.append(10000)
        ΔTm.append(float(metals_data[metal][-2]))
    
    print(Kd_values)
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