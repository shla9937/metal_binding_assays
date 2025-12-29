#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit 
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import Normalize
import matplotlib.colors

'''
Inputs:
-   DSF raw data (CSV exported from DA2)
-   protein name
-   optional analysis temperature
-   optional exclusion temperature

Hardcoded:
-   Titration concentrations
-   Protein concentration
-   Titration metals
-   Excluded points

Outputs:
-   Raw series of each titration, single metal (not averaged, fit is smoothed) - 32 plots, shades of color
-   Noramlized series of each titration, single metal (normalized, averaged, fit is smoothed, calculate Tm of WT or EDTA) - 8 plots, shades of color 
-   Tm as a function of concentration, all metals (averaged, calculate Tm) - 1 plot, each different colors
-   Folded % as function of concentration, all metals (averaged) - 1 plot, each different colors
-   CSV of Kds for each metal (confidences)
-   CSV of df (temp, concentration, metal)

Functions:
-   import csv -> create pd df
-   assign concentrations and identities
-   plot raw data 
-   normalize raw data
-   average
-   fit normalized curve
-   smooth normalized curve
-   calculate Tm
-   choose analysis temperature (Tm of WT)
-   plot normalized data (box at WT Tm)
-   plot concentration vs Tm (linear regression line)
-   extract normalized (and smoothed) values at analysis temp -> new pd df (conc vs fold %)
-   fit titration curve -> find Kd
-   plot titration curves (add Kds and analysis Tm)
-   output Kd values
-   output raw df 
-   output analyzed (fold % df)

'''

def main():
    parser = argparse.ArgumentParser(description="Analyze 384 well DSF 8 metal, triplicate experiment.")
    parser.add_argument('-c', '--csv', type=str, required=True, help="Raw DSF values from DA2")
    parser.add_argument('-p', '--protein', type=str, required=True, help="Name of protein")
    parser.add_argument('-e', '--exclude', type=float, required=False, help="Name of protein")
    args = parser.parse_args()

    df = parse_csv_file(args.csv)
    if args.exclude:
        df = exclude_temps(df, args.exclude)
    raw_df = assign_conc(df)
    # plot_df(raw_df, 'Fluorescence')
    norm_df = normalize(raw_df)
    # plot_df(norm_df, 'Normalized Fluorescence')
    avg_df = average(norm_df)
    avg_tm_df = find_tms(avg_df)
    # plot_df(avg_tm_df, 'Average Normalized Fluorescence', error_column='Standard Error')
    plot_df(avg_tm_df, 'Smoothed Fluorescence', error_column='Standard Error')
    # plot_tms(avg_tm_df)
    titration_df = find_kds(avg_tm_df)
    plot_kds(titration_df)

def parse_csv_file(csv):
    with open(csv, 'r') as f:
        lines = f.readlines()
    header_idx = None
    for i, line in enumerate(lines):
        if not line.startswith('#') and line.strip():
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("Could not find data table header in file.")
    df = pd.read_csv(csv, skiprows=header_idx)
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=['Target'])
    return df

def exclude_temps(df, exclude):
    return df[df['Temperature'] <= exclude]

def assign_conc(df):
    global metals_left, metals_right, rows, metal_colors
    df['Metal'] = None
    df['Concentration'] = np.nan
    # which metal is being titrated in columns 1-12, A-P)
    metals_left = ["Mn²⁺", "Mn²⁺", "Fe³⁺", "Fe³⁺", "Co²⁺", "Co²⁺", "Ni²⁺", "Ni²⁺",
                   "Cu²⁺", "Cu²⁺", "Nd³⁺", "Nd³⁺", "Dy³⁺", "Dy³⁺", "Mix", "Mix"]
    # which metal is being titrated in columns 13-24, A-P)
    metals_right = ["Mn²⁺", "EDTA", "Fe³⁺", "EDTA", "Co²⁺", "EDTA", "Ni²⁺", "EDTA",
                    "Cu²⁺", "EDTA", "Nd³⁺", "EDTA", "Dy³⁺", "EDTA", "Mix", "EDTA"]
    concentrations = [1000, 333, 111, 37, 12.3, 4.1, 1.37, 0.457, 0.152, 0.051, 0.017, 0.006] # in µM (last well in EDTA is 0)
    rows = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P"]
    metal_colors = {
        "Mn²⁺": "#1f77b4",    # blue
        "Fe³⁺": "#ff7f0e",    # orange
        "Co²⁺": "#2ca02c",    # green
        "Ni²⁺": "#d62728",    # red
        "Cu²⁺": "#9467bd",    # purple
        "Nd³⁺": "#8c564b",    # brown
        "Dy³⁺": "#e377c2",    # pink
        "Mix": "#7f7f7f",     # gray
        "EDTA": "#bcbd22"     # olive
    }

    for row in rows:
        for well in range(1,13):
            well_pos = row+str(well)
            df.loc[df['Well Position'] == well_pos, 'Metal'] = metals_left[rows.index(row)]
            df.loc[df['Well Position'] == well_pos, 'Concentration'] = concentrations[well-1]            
        for well in range(13,25):
            well_pos = row+str(well)
            df.loc[df['Well Position'] == well_pos, 'Metal'] = metals_right[rows.index(row)]
            if metals_right[rows.index(row)] == 'EDTA' and well == 24:
                df.loc[df['Well Position'] == well_pos, 'Concentration'] = 0
            else:
                df.loc[df['Well Position'] == well_pos, 'Concentration'] = concentrations[well-13]
    return df

def plot_df(df, y_column, error_column=None):    
    # If error_column provided, use averaged data mode
    if error_column:
        # Dynamically create subplot grid based on number of unique metals
        unique_metals = sorted(df['Metal'].unique())
        n_metals = len(unique_metals)
        n_cols = 4
        n_rows = (n_metals + n_cols - 1) // n_cols  # Ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten()
        
        for idx, metal in enumerate(unique_metals):
            ax = axes[idx]
            metal_data = df[df['Metal'] == metal]
            
            # Get unique concentrations for color gradient
            concentrations = sorted(metal_data['Concentration'].unique())
            
            for conc in concentrations:
                conc_data = metal_data[metal_data['Concentration'] == conc].sort_values('Temperature')
                conc_idx = concentrations.index(conc)
                
                # Use base metal color and vary brightness with concentration
                color = matplotlib.colors.to_rgba(metal_colors[metal])
                color = tuple(c * (0.3 + 0.7 * (1 - conc_idx / max(len(concentrations)-1, 1))) for c in color[:3]) + (color[3],)
                
                ax.errorbar(conc_data['Temperature'], conc_data[y_column], 
                        yerr=conc_data[error_column], label=f"{conc:.3g} µM", 
                        color=color, alpha=0.3)
                
                # Plot Tm as vertical line with same color gradient
                tm_value = conc_data['Tm'].iloc[0]  # All rows in group have same Tm
                ax.axvline(x=tm_value, color=color, linestyle='--', linewidth=1.5, alpha=0.6)
            
            ax.set_xlabel('Temperature (°C)')
            ax.set_ylabel(y_column)
            ax.set_title(f"{metal}")
            # ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide any unused subplots
        for idx in range(n_metals, len(axes)):
            axes[idx].axis('off')
    else:
        # Raw data mode (original code)
        fig, axes = plt.subplots(8, 4, figsize=(16, 16))
        axes = axes.flatten()
        
        plot_idx = 0
        
        for row in rows:
            for well_range, metal_list, label_suffix in [
                (range(1, 13), metals_left, "Wells 1-12"),
                (range(13, 25), metals_right, "Wells 13-24")
            ]:
                ax = axes[plot_idx]
                metal = metal_list[rows.index(row)]
                
                # Get unique concentrations for color gradient
                well_positions = [row + str(w) for w in well_range]
                well_data_subset = df[df['Well Position'].isin(well_positions)]
                concentrations = sorted(well_data_subset['Concentration'].unique())
                
                for well in well_range:
                    well_pos = row + str(well)
                    well_data = df[df['Well Position'] == well_pos].sort_values('Temperature')
                    conc = well_data['Concentration'].iloc[0]
                    conc_idx = concentrations.index(conc) if conc in concentrations else 0
                    
                    color = matplotlib.colors.to_rgba(metal_colors[metal])
                    color = tuple(c * (0.3 + 0.7 * (1 - conc_idx / max(len(concentrations)-1, 1))) for c in color[:3]) + (color[3],)
                    
                    ax.plot(well_data['Temperature'], well_data[y_column], 
                           label=f"{well_pos} ({conc:.3g} µM)", alpha=0.8, color=color)
                
                ax.set_xlabel('Temperature (°C)')
                ax.set_ylabel(y_column)
                ax.set_title(f"Row {row} - {metal} ({label_suffix})")
                ax.grid(True, alpha=0.3)
                plot_idx += 1
    
    plt.tight_layout()
    plt.savefig(f"{y_column.lower().replace(' ', '_')}_melt_curves.png", dpi=150)
    plt.show()

def normalize(df):
    scaler = MinMaxScaler()
    df['Normalized Fluorescence'] = df.groupby('Well')['Fluorescence'].transform(
        lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten())
    return df

def average(df):
    df_copy = df.copy()
    df_copy['Temperature'] = df_copy['Temperature'].round(1)
    avg_df = df_copy.groupby(['Metal', 'Concentration', 'Temperature'])['Normalized Fluorescence'].apply(list).reset_index()
    avg_df['Average Normalized Fluorescence'] = avg_df['Normalized Fluorescence'].apply(np.mean)
    avg_df['Standard Error'] = avg_df['Normalized Fluorescence'].apply(lambda x: np.std(x) / np.sqrt(len(x)))
    return avg_df

def find_tms(df):
    # Calculate Tm (peak of first derivative) for each metal-concentration combination
    tm_values = []
    smoothed_values = []
    group_indices = []
    
    for (metal, conc), group in df.groupby(['Metal', 'Concentration']):
        group = group.sort_values('Temperature')
        temps = group['Temperature'].values
        fluor = group['Average Normalized Fluorescence'].values
        
        # Smooth the fluorescence data using Savitzky-Golay filter
        window_length = min(len(fluor), 51)  # Use odd window length, max 7
        if window_length % 2 == 0:
            window_length -= 1
        if window_length >= 5:
            fluor_smooth = savgol_filter(fluor, window_length, 3)
        else:
            fluor_smooth = fluor
        
        # Calculate first derivative on smoothed data
        derivative = np.gradient(fluor_smooth, temps)
        
        # Find peak of derivative (steepest slope = Tm)
        peak_idx = np.argmax(derivative)
        tm = temps[peak_idx]
        
        tm_values.append(tm)
        smoothed_values.append(fluor_smooth)
        group_indices.append(group.index)
    
    # Add Tm and smoothed fluorescence columns to dataframe
    df['Tm'] = np.nan
    df['Smoothed Fluorescence'] = np.nan
    for tm, smoothed, indices in zip(tm_values, smoothed_values, group_indices):
        df.loc[indices, 'Tm'] = tm
        df.loc[indices, 'Smoothed Fluorescence'] = smoothed
    
    print(df)
    return df

def plot_tms(df):
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Get unique metal-concentration combinations with their Tm values
    tm_data = df.groupby(['Metal', 'Concentration'])['Tm'].first().reset_index()
    
    for metal in sorted(tm_data['Metal'].unique()):
        metal_df = tm_data[tm_data['Metal'] == metal].sort_values('Concentration')
        concentrations = metal_df['Concentration'].values
        tms = metal_df['Tm'].values
        
        color = metal_colors[metal]
        
        # Plot the data points
        ax.scatter(concentrations, tms, color=color, label=metal, s=50, alpha=0.7)
        
        # Fit linear regression
        coeffs = np.polyfit(concentrations, tms, 1)
        fit_line = np.poly1d(coeffs)
        
        # Plot the fitted line
        conc_range = np.linspace(concentrations.min(), concentrations.max(), 100)
        ax.plot(conc_range, fit_line(conc_range), color=color, linestyle='-', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Concentration (µM)', fontsize=8)
    ax.set_ylabel('Tm (°C)', fontsize=8)
    ax.set_title('Tm vs Concentration', fontsize=8)
    ax.set_xscale('log')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tm_vs_concentration.png', dpi=300)
    plt.show()

def find_kds(df):
    # Get reference Tm values
    wt_tm = df[(df['Metal'] == 'EDTA') & (df['Concentration'] == 0)]['Tm'].iloc[0]
    edta_tm = df[(df['Metal'] == 'EDTA') & (df['Concentration'] == 111)]['Tm'].iloc[0]
    
    # Create new dataframe to hold results
    kd_data = []
    
    # For each metal-concentration combination
    for (metal, conc), group in df.groupby(['Metal', 'Concentration']):
        group = group.sort_values('Temperature')
        temps = group['Temperature'].values
        smoothed_fluor = group['Smoothed Fluorescence'].values
        std_err = group['Standard Error'].values
        
        # Find temperature closest to WT Tm
        wt_idx = np.argmin(np.abs(temps - wt_tm))
        wt_fluor = smoothed_fluor[wt_idx]
        wt_se = std_err[wt_idx]
        
        # Find temperature closest to EDTA Tm
        edta_idx = np.argmin(np.abs(temps - edta_tm))
        edta_fluor = smoothed_fluor[edta_idx]
        edta_se = std_err[edta_idx]
        
        kd_data.append({
            'Metal': metal,
            'Concentration': conc,
            'WT Tm Temperature': wt_tm,
            'WT Tm': wt_fluor,
            'WT Tm Standard Error': wt_se,
            'EDTA Tm Temperature': edta_tm,
            'EDTA Tm': edta_fluor,
            'EDTA Tm Standard Error': edta_se
        })
    
    kd_df = pd.DataFrame(kd_data)
    return kd_df

def plot_kds(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Get the actual temperature values
    wt_temp = df['WT Tm Temperature'].iloc[0]
    edta_temp = df['EDTA Tm Temperature'].iloc[0]
    
    # Plot 1: WT Tm titrations
    for metal in sorted(df['Metal'].unique()):
        metal_data = df[df['Metal'] == metal].sort_values('Concentration')
        concentrations = metal_data['Concentration'].values
        wt_values = metal_data['WT Tm'].values
        wt_errors = metal_data['WT Tm Standard Error'].values
        
        color = metal_colors[metal]
        ax1.errorbar(concentrations, wt_values, yerr=wt_errors, 
                    color=color, label=metal, marker='o', linestyle='-', 
                    capsize=3, alpha=0.8)
    
    ax1.set_xlabel('Concentration (µM)', fontsize=10)
    ax1.set_ylabel('Fluorescence', fontsize=10)
    ax1.set_title(f'Titration at WT Tm ({wt_temp:.1f}°C)', fontsize=11)
    ax1.set_xscale('log')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: EDTA Tm titrations
    for metal in sorted(df['Metal'].unique()):
        metal_data = df[df['Metal'] == metal].sort_values('Concentration')
        concentrations = metal_data['Concentration'].values
        edta_values = metal_data['EDTA Tm'].values
        edta_errors = metal_data['EDTA Tm Standard Error'].values
        
        color = metal_colors[metal]
        ax2.errorbar(concentrations, edta_values, yerr=edta_errors, 
                    color=color, label=metal, marker='o', linestyle='-', 
                    capsize=3, alpha=0.8)
    
    ax2.set_xlabel('Concentration (µM)', fontsize=10)
    ax2.set_ylabel('Fluorescence', fontsize=10)
    ax2.set_title(f'Titration at EDTA Tm ({edta_temp:.1f}°C)', fontsize=11)
    ax2.set_xscale('log')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metal_titrations.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()