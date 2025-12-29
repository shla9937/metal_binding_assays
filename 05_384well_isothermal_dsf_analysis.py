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
    parser.add_argument('-o', '--override', type=float, required=False, help="Override analysis temperature")
    parser.add_argument('-x', '--exclude_high', type=int, default=0, help="Number of highest concentrations to exclude from fitting")
    parser.add_argument('-m', '--model', type=str, default='hill', choices=['hill', 'two-site'], 
                        help="Binding model: 'hill' (Hill equation with n) or 'two-site' (two independent sites)")
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
    titration_df, kd_results = find_kds(avg_tm_df, override=args.override, exclude_high=args.exclude_high, model=args.model)
    print(kd_results)
    plot_kds(titration_df, kd_results, model=args.model)

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
    global metals_left, metals_right, rows, metal_colors, protein_conc
    df['Metal'] = None
    df['Concentration'] = np.nan
    protein_conc = 5  # µM
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

# Hill equation with cooperativity coefficient
def binding_curve_hill(conc, kd, ymin, ymax, n):
    """Hill equation with Hill coefficient n for cooperativity"""
    return ymin + (ymax - ymin) * conc**n / (kd**n + conc**n)

# Two independent binding sites
def binding_curve_two_site(conc, kd1, kd2, ymin, ymax):
    """Two independent binding sites with different affinities"""
    site1 = conc / (kd1 + conc)
    site2 = conc / (kd2 + conc)
    return ymin + (ymax - ymin) * 0.5 * (site1 + site2)

# Simple 1:1 binding equation (Hill equation n=1)
# def binding_curve(conc, kd, ymin, ymax):
#     """Hill equation for 1:1 binding"""
#     return ymin + (ymax - ymin) * conc / (kd + conc)

# Quadratic binding equation (Bai et al. 2018) - accounts for ligand depletion
# def binding_curve(conc, kd, ymin, ymax):
#     """Quadratic binding equation accounting for ligand depletion"""
#     global protein_conc
#     Pt = protein_conc  # Total protein concentration in µM
#     Lt = conc  # Total ligand (metal) concentration in µM
#     
#     # Fraction bound = (([P]t + [L]t + Kd) - sqrt((([P]t + [L]t + Kd)^2 - 4[P]t[L]t))) / (2[P]t)
#     fraction_bound = ((Pt + Lt + kd) - np.sqrt((Pt + Lt + kd)**2 - 4*Pt*Lt)) / (2*Pt)
#     
#     return ymin + (ymax - ymin) * fraction_bound

def fit_binding_curve(concentrations, values, errors, model='hill'):
    """Fit binding curve and return parameters with confidence intervals"""
    try:
        # Initial parameter guesses
        ymin_guess = np.min(values)
        ymax_guess = np.max(values)
        kd_guess = np.median(concentrations)
        
        # Replace zero errors with small value to avoid division by zero
        errors = np.where(errors == 0, 1e-10, errors)
        
        if model == 'hill':
            # Hill equation with coefficient
            n_guess = 1.0  # Hill coefficient
            popt, pcov = curve_fit(
                binding_curve_hill, 
                concentrations, 
                values,
                p0=[kd_guess, ymin_guess, ymax_guess, n_guess],
                sigma=errors,
                absolute_sigma=True,
                maxfev=10000,
                bounds=([0, 0, 0, 0.1], [np.inf, 1, 1, 5])  # Allow n from 0.1 to 5
            )
            kd, ymin, ymax, n = popt
            perr = np.sqrt(np.diag(pcov))
            kd_err = perr[0] * 1.96
            n_err = perr[3] * 1.96
            return {'Kd': kd, 'Kd_Error': kd_err, 'Hill_n': n, 'Hill_n_Error': n_err, 
                    'Fit_Params': popt, 'Model': 'hill'}
        
        elif model == 'two-site':
            # Two independent binding sites
            kd1_guess = kd_guess * 0.1  # Higher affinity site
            kd2_guess = kd_guess * 10   # Lower affinity site
            popt, pcov = curve_fit(
                binding_curve_two_site, 
                concentrations, 
                values,
                p0=[kd1_guess, kd2_guess, ymin_guess, ymax_guess],
                sigma=errors,
                absolute_sigma=True,
                maxfev=10000,
                bounds=([0, 0, 0, 0], [np.inf, np.inf, 1, 1])
            )
            kd1, kd2, ymin, ymax = popt
            perr = np.sqrt(np.diag(pcov))
            kd1_err = perr[0] * 1.96
            kd2_err = perr[1] * 1.96
            # Sort Kd values so Kd1 is always the tighter binding
            if kd1 > kd2:
                kd1, kd2 = kd2, kd1
                kd1_err, kd2_err = kd2_err, kd1_err
            return {'Kd1': kd1, 'Kd1_Error': kd1_err, 'Kd2': kd2, 'Kd2_Error': kd2_err,
                    'Fit_Params': popt, 'Model': 'two-site'}
    except:
        if model == 'hill':
            return {'Kd': np.nan, 'Kd_Error': np.nan, 'Hill_n': np.nan, 'Hill_n_Error': np.nan, 
                    'Fit_Params': None, 'Model': 'hill'}
        else:
            return {'Kd1': np.nan, 'Kd1_Error': np.nan, 'Kd2': np.nan, 'Kd2_Error': np.nan,
                    'Fit_Params': None, 'Model': 'two-site'}

def find_kds(df, override=None, exclude_high=0, model='hill'):
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
        
        data_dict = {
            'Metal': metal,
            'Concentration': conc,
            'WT Tm Temperature': wt_tm,
            'WT Tm': wt_fluor,
            'WT Tm Standard Error': wt_se,
            'EDTA Tm Temperature': edta_tm,
            'EDTA Tm': edta_fluor,
            'EDTA Tm Standard Error': edta_se
        }
        
        # Add override temperature data if provided
        if override is not None:
            override_idx = np.argmin(np.abs(temps - override))
            override_fluor = smoothed_fluor[override_idx]
            override_se = std_err[override_idx]
            data_dict['Override Temperature'] = override
            data_dict['Override Tm'] = override_fluor
            data_dict['Override Tm Standard Error'] = override_se
        
        kd_data.append(data_dict)
    
    kd_df = pd.DataFrame(kd_data)
    
    # Now fit binding curves for each metal at each temperature
    # Store Kd values in a list to convert to DataFrame
    kd_list = []
    
    for metal in kd_df['Metal'].unique():
        metal_data = kd_df[kd_df['Metal'] == metal].sort_values('Concentration')
        concs = metal_data['Concentration'].values
        
        # Exclude highest concentrations if requested (but keep zero concentration)
        if exclude_high > 0:
            # Get indices to keep (exclude N highest non-zero concentrations)
            non_zero_mask = concs > 0
            sorted_indices = np.argsort(concs)
            exclude_indices = sorted_indices[-(exclude_high):] if np.sum(non_zero_mask) > exclude_high else []
            fit_mask = np.ones(len(concs), dtype=bool)
            fit_mask[exclude_indices] = False
            
            concs_fit = concs[fit_mask]
            wt_vals_fit = (1 - metal_data['WT Tm'].values)[fit_mask]
            wt_errs_fit = metal_data['WT Tm Standard Error'].values[fit_mask]
            edta_vals_fit = (1 - metal_data['EDTA Tm'].values)[fit_mask]
            edta_errs_fit = metal_data['EDTA Tm Standard Error'].values[fit_mask]
        else:
            concs_fit = concs
            wt_vals_fit = 1 - metal_data['WT Tm'].values
            wt_errs_fit = metal_data['WT Tm Standard Error'].values
            edta_vals_fit = 1 - metal_data['EDTA Tm'].values
            edta_errs_fit = metal_data['EDTA Tm Standard Error'].values
        
        # Fit WT Tm data
        fit_result = fit_binding_curve(concs_fit, wt_vals_fit, wt_errs_fit, model=model)
        fit_result['Metal'] = metal
        fit_result['Temperature'] = 'WT'
        kd_list.append(fit_result)
        
        # Fit EDTA Tm data
        fit_result = fit_binding_curve(concs_fit, edta_vals_fit, edta_errs_fit, model=model)
        fit_result['Metal'] = metal
        fit_result['Temperature'] = 'EDTA'
        kd_list.append(fit_result)
        
        # Fit override data if present
        if override is not None:
            if exclude_high > 0:
                override_vals_fit = (1 - metal_data['Override Tm'].values)[fit_mask]
                override_errs_fit = metal_data['Override Tm Standard Error'].values[fit_mask]
            else:
                override_vals_fit = 1 - metal_data['Override Tm'].values
                override_errs_fit = metal_data['Override Tm Standard Error'].values
            fit_result = fit_binding_curve(concs_fit, override_vals_fit, override_errs_fit, model=model)
            fit_result['Metal'] = metal
            fit_result['Temperature'] = 'Override'
            kd_list.append(fit_result)
    
    kd_results = pd.DataFrame(kd_list)
    return kd_df, kd_results

def plot_kds(df, kd_results, model='hill'):
    # Check if override temperature data exists
    has_override = 'Override Temperature' in df.columns
    
    # Define plot configurations
    plot_configs = [('WT Tm', 'WT Tm Temperature', 'WT'),
        ('EDTA Tm', 'EDTA Tm Temperature', 'EDTA')]
    if has_override:
        plot_configs.append(('Override Tm', 'Override Temperature', 'Override'))
    
    n_plots = len(plot_configs)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    # Concentration range for fitted curves (exclude zero)
    conc_min = df[df['Concentration'] > 0]['Concentration'].min()
    conc_max = df['Concentration'].max()
    conc_fit = np.logspace(np.log10(conc_min), np.log10(conc_max), 200)
    
    # Loop through each plot configuration
    for ax, (data_col, temp_col, kd_key) in zip(axes, plot_configs):
        temp_value = df[temp_col].iloc[0]
        
        for metal in sorted(df['Metal'].unique()):
            metal_data = df[df['Metal'] == metal].sort_values('Concentration')
            concentrations = metal_data['Concentration'].values
            values = 1 - metal_data[data_col].values  # Convert to % folded
            errors = metal_data[f'{data_col} Standard Error'].values
            
            color = metal_colors[metal]
            
            # Get fit parameters from DataFrame
            kd_row = kd_results[(kd_results['Metal'] == metal) & (kd_results['Temperature'] == kd_key)]
            if not kd_row.empty:
                popt = kd_row['Fit_Params'].values[0]
                
                if model == 'hill':
                    kd = kd_row['Kd'].values[0]
                    kd_err = kd_row['Kd_Error'].values[0]
                    n = kd_row['Hill_n'].values[0]
                    n_err = kd_row['Hill_n_Error'].values[0]
                    
                    # Create label with Kd and Hill coefficient
                    if not np.isnan(kd):
                        if kd < 1.0:
                            kd_nm = kd * 1000
                            kd_err_nm = kd_err * 1000
                            label = f"{metal}: Kd={kd_nm:.0f}±{kd_err_nm:.0f} nM, n={n:.2f}±{n_err:.2f}"
                        else:
                            label = f"{metal}: Kd={kd:.1f}±{kd_err:.1f} µM, n={n:.2f}±{n_err:.2f}"
                    else:
                        label = f"{metal}: Kd=N/A"
                        
                elif model == 'two-site':
                    kd1 = kd_row['Kd1'].values[0]
                    kd1_err = kd_row['Kd1_Error'].values[0]
                    kd2 = kd_row['Kd2'].values[0]
                    kd2_err = kd_row['Kd2_Error'].values[0]
                    
                    # Create label with both Kd values
                    if not np.isnan(kd1):
                        # Format Kd1
                        if kd1 < 1.0:
                            kd1_str = f"{kd1*1000:.0f}±{kd1_err*1000:.0f} nM"
                        else:
                            kd1_str = f"{kd1:.1f}±{kd1_err:.1f} µM"
                        # Format Kd2
                        if kd2 < 1.0:
                            kd2_str = f"{kd2*1000:.0f}±{kd2_err*1000:.0f} nM"
                        else:
                            kd2_str = f"{kd2:.1f}±{kd2_err:.1f} µM"
                        label = f"{metal}: Kd1={kd1_str}, Kd2={kd2_str}"
                    else:
                        label = f"{metal}: Kd=N/A"
            else:
                popt = None
                label = f"{metal}: Kd=N/A"
            
            # Plot data points
            ax.errorbar(concentrations, values, yerr=errors, 
                       color=color, marker='o', linestyle='', 
                       capsize=3, alpha=0.7, markersize=6)
            
            # Plot fitted curve
            if popt is not None:
                if model == 'hill':
                    fit_vals = binding_curve_hill(conc_fit, *popt)
                else:  # two-site
                    fit_vals = binding_curve_two_site(conc_fit, *popt)
                ax.plot(conc_fit, fit_vals, color=color, linestyle='-', 
                       linewidth=2, alpha=0.8, label=label)
            else:
                ax.plot([], [], color=color, label=label)  # Just for legend
        
        # Format axis
        ax.set_xlabel('Concentration (µM)', fontsize=10)
        ax.set_ylabel('% Folded', fontsize=10)
        title_name = 'WT Tm' if kd_key == 'WT' else ('EDTA Tm' if kd_key == 'EDTA' else 'Override Temp')
        ax.set_title(f'Titration at {title_name} ({temp_value:.1f}°C)', fontsize=11)
        ax.set_xscale('log')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('metal_titrations.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()