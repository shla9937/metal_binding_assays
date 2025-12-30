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


def main():
    parser = argparse.ArgumentParser(description="Analyze 384 well DSF 8 metal, triplicate experiment.")
    parser.add_argument('-c', '--csv', type=str, required=True, help="Raw DSF values from DA2")
    parser.add_argument('-p', '--protein', type=str, required=True, help="Name of protein")
    parser.add_argument('-e', '--exclude', type=float, required=False, help="Exclude temps over this value")
    parser.add_argument('-o', '--override', type=float, required=False, help="Override analysis temperature")
    parser.add_argument('-x', '--exclude_high', type=int, default=0, help="Number of highest concentrations to exclude from fitting")
    parser.add_argument('-m', '--model', type=str, default='hill', choices=['hill', 'two-site'], 
                        help="Binding model: 'hill' (Hill equation with n) or 'two-site' (two independent sites)")
    args = parser.parse_args()
    
    df = parse_csv_file(args.csv)
    if args.exclude:
        df = exclude_temps(df, args.exclude)
    raw_df = assign_conc(df)
    norm_df = normalize(raw_df)
    avg_df = average(norm_df)
    avg_tm_df = find_tms(avg_df)
    titration_df, kd_results = find_kds(avg_tm_df, override=args.override, exclude_high=args.exclude_high, model=args.model)
    plot_df(raw_df, 'Fluorescence', args.protein)
    plot_df(avg_tm_df, 'Smoothed Fluorescence', args.protein, error_column='Standard Error')
    plot_tms(avg_tm_df, args.protein)
    plot_kds(titration_df, kd_results, args.protein, model=args.model)
    plot_kd_bars(titration_df, kd_results, args.protein, model=args.model)
    save_results_csv(titration_df, kd_results, args.protein)

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
        "Mn²⁺": "#FF1493",    # deep pink
        "Fe³⁺": "#FF8C00",    # dark orange
        "Co²⁺": "#8B008B",    # dark magenta
        "Ni²⁺": "#00C853",    # bright green
        "Cu²⁺": "#1E90FF",    # dodger blue
        "Nd³⁺": "#8A2BE2",    # blue violet
        "Dy³⁺": "#FFD700",    # gold
        "Mix": "#808080",     # gray
        "EDTA": "#228B22"     # forest green
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

# Atomic number sorting helper
def get_atomic_number(metal_name):
    """Get atomic number for sorting metals"""
    atomic_numbers = {
        'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
        'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
        'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
        'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
        'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
        'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
        'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
        'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
        'Pa': 91, 'U': 92
    }
    element = ''.join(c for c in metal_name if c.isalpha())
    return atomic_numbers.get(element, 999)

def plot_df(df, y_column, protein_name, error_column=None):    
    # If error_column provided, use averaged data mode
    if error_column:
        unique_metals = sorted(df['Metal'].unique(), key=get_atomic_number)
        n_metals = len(unique_metals)
        n_cols = 4
        n_rows = (n_metals + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten()
        
        for idx, metal in enumerate(unique_metals):
            ax = axes[idx]
            metal_data = df[df['Metal'] == metal]
            concentrations = sorted(metal_data['Concentration'].unique())
            
            for conc in concentrations:
                conc_data = metal_data[metal_data['Concentration'] == conc].sort_values('Temperature')
                conc_idx = concentrations.index(conc)
                
                color = matplotlib.colors.to_rgba(metal_colors[metal])
                color = tuple(c * (0.3 + 0.7 * (1 - conc_idx / max(len(concentrations)-1, 1))) for c in color[:3]) + (color[3],)
                
                ax.errorbar(conc_data['Temperature'], conc_data[y_column], 
                        yerr=conc_data[error_column], label=f"{conc:.3g} µM", 
                        color=color, alpha=0.3)
            
            wt_tm = df[(df['Metal'] == 'EDTA') & (df['Concentration'] == 0)]['Tm'].iloc[0]
            ax.axvline(x=wt_tm, color='black', linestyle='--', linewidth=1.5, alpha=0.8, label=f'WT Tm ({wt_tm:.1f}°C)')
            
            ax.set_xlabel('Temperature (°C)')
            ax.set_ylabel(y_column)
            ax.set_title(f"{metal}")
            ax.grid(True, alpha=0.3)

        for idx in range(n_metals, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(protein_name, fontsize=14, y=0.995)
    else:
        fig, axes = plt.subplots(8, 4, figsize=(16, 16))
        axes = axes.flatten()
        plot_idx = 0
        
        for row in rows:
            for well_range, metal_list, label_suffix in [
                (range(1, 13), metals_left, "Wells 1-12"),
                (range(13, 25), metals_right, "Wells 13-24")]:
                
                ax = axes[plot_idx]
                metal = metal_list[rows.index(row)]

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
        
        fig.suptitle(protein_name, fontsize=14, y=0.995)
    
    plt.tight_layout()
    protein_lower = protein_name.lower()
    plt.savefig(f"{protein_lower}_{y_column.lower().replace(' ', '_')}_melt_curves.png", dpi=150)
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
        window_length = min(len(fluor), 51)
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

def plot_tms(df, protein_name):
    fig, ax = plt.subplots(figsize=(4, 3))
    tm_data = df.groupby(['Metal', 'Concentration'])['Tm'].first().reset_index()
    
    for metal in sorted(tm_data['Metal'].unique(), key=get_atomic_number):
        metal_df = tm_data[tm_data['Metal'] == metal].sort_values('Concentration')
        concentrations = metal_df['Concentration'].values
        tms = metal_df['Tm'].values
        
        color = metal_colors[metal]
        ax.scatter(concentrations, tms, color=color, label=metal, s=50, alpha=0.7)
        
        coeffs = np.polyfit(concentrations, tms, 1)
        fit_line = np.poly1d(coeffs)
        
        conc_range = np.linspace(concentrations.min(), concentrations.max(), 100)
        ax.plot(conc_range, fit_line(conc_range), color=color, linestyle='-', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Concentration (µM)', fontsize=8)
    ax.set_ylabel('Tm (°C)', fontsize=8)
    ax.set_title('Tm vs Concentration', fontsize=8)
    # ax.set_xscale('log')
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(protein_name, fontsize=10, y=0.98)
    
    plt.tight_layout()
    protein_lower = protein_name.lower()
    plt.savefig(f'{protein_lower}_tm_vs_concentration.png', dpi=300)
    plt.show()

def binding_curve_hill(conc, kd, ymin, ymax, n):
    """Hill equation with Hill coefficient n for cooperativity"""
    return ymin + (ymax - ymin) * conc**n / (kd**n + conc**n)

def binding_curve_two_site(conc, kd1, kd2, ymin, ymax):
    """Two independent binding sites with different affinities"""
    site1 = conc / (kd1 + conc)
    site2 = conc / (kd2 + conc)
    return ymin + (ymax - ymin) * 0.5 * (site1 + site2)

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
            
            # Calculate R²
            y_pred = binding_curve_hill(concentrations, *popt)
            ss_res = np.sum((values - y_pred)**2)
            ss_tot = np.sum((values - np.mean(values))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return {'Kd': kd, 'Kd_Error': kd_err, 'Hill_n': n, 'Hill_n_Error': n_err, 
                    'R_squared': r_squared, 'Fit_Params': popt, 'Model': 'hill'}
        
        elif model == 'two-site':
            # Two independent binding sites
            kd1_guess = kd_guess * 0.1  
            kd2_guess = kd_guess * 10 
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
            
            if kd1 > kd2:
                kd1, kd2 = kd2, kd1
                kd1_err, kd2_err = kd2_err, kd1_err
            
            # Calculate R²
            y_pred = binding_curve_two_site(concentrations, *popt)
            ss_res = np.sum((values - y_pred)**2)
            ss_tot = np.sum((values - np.mean(values))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return {'Kd1': kd1, 'Kd1_Error': kd1_err, 'Kd2': kd2, 'Kd2_Error': kd2_err,
                    'R_squared': r_squared, 'Fit_Params': popt, 'Model': 'two-site'}
    except:
        if model == 'hill':
            return {'Kd': np.nan, 'Kd_Error': np.nan, 'Hill_n': np.nan, 'Hill_n_Error': np.nan, 
                    'R_squared': np.nan, 'Fit_Params': None, 'Model': 'hill'}
        else:
            return {'Kd1': np.nan, 'Kd1_Error': np.nan, 'Kd2': np.nan, 'Kd2_Error': np.nan,
                    'R_squared': np.nan, 'Fit_Params': None, 'Model': 'two-site'}

def find_kds(df, override=None, exclude_high=0, model='hill'):
    wt_tm = df[(df['Metal'] == 'EDTA') & (df['Concentration'] == 0)]['Tm'].iloc[0]
    kd_data = []
    
    for (metal, conc), group in df.groupby(['Metal', 'Concentration']):
        group = group.sort_values('Temperature')
        temps = group['Temperature'].values
        smoothed_fluor = group['Smoothed Fluorescence'].values
        std_err = group['Standard Error'].values
        
        wt_idx = np.argmin(np.abs(temps - wt_tm))
        wt_fluor = smoothed_fluor[wt_idx]
        wt_se = std_err[wt_idx]
        
        data_dict = {
            'Metal': metal,
            'Concentration': conc,
            'WT Tm Temperature': wt_tm,
            'WT Tm': wt_fluor,
            'WT Tm Standard Error': wt_se
        }
        
        if override is not None:
            override_idx = np.argmin(np.abs(temps - override))
            override_fluor = smoothed_fluor[override_idx]
            override_se = std_err[override_idx]
            data_dict['Override Temperature'] = override
            data_dict['Override Tm'] = override_fluor
            data_dict['Override Tm Standard Error'] = override_se
        
        kd_data.append(data_dict)
    
    kd_df = pd.DataFrame(kd_data)
    kd_list = []
    
    for metal in kd_df['Metal'].unique():
        metal_data = kd_df[kd_df['Metal'] == metal].sort_values('Concentration')
        concs = metal_data['Concentration'].values
        
        if exclude_high > 0:
            non_zero_mask = concs > 0
            sorted_indices = np.argsort(concs)
            exclude_indices = sorted_indices[-(exclude_high):] if np.sum(non_zero_mask) > exclude_high else []
            fit_mask = np.ones(len(concs), dtype=bool)
            fit_mask[exclude_indices] = False
            
            concs_fit = concs[fit_mask]
            wt_vals_fit = (1 - metal_data['WT Tm'].values)[fit_mask]
            wt_errs_fit = metal_data['WT Tm Standard Error'].values[fit_mask]
        else:
            concs_fit = concs
            wt_vals_fit = 1 - metal_data['WT Tm'].values
            wt_errs_fit = metal_data['WT Tm Standard Error'].values
        
        fit_result = fit_binding_curve(concs_fit, wt_vals_fit, wt_errs_fit, model=model)
        
        # Quality control: reject fits with poor R² or low amplitude (non-binding)
        r2_threshold = 0.7
        amplitude_threshold = 0.1
        
        if not np.isnan(fit_result['R_squared']):
            if model == 'hill':
                ymin, ymax = fit_result['Fit_Params'][1], fit_result['Fit_Params'][2]
            else:  # two-site
                ymin, ymax = fit_result['Fit_Params'][2], fit_result['Fit_Params'][3]
            
            amplitude = abs(ymax - ymin)
            
            # Mark as N/A if R² is too low or amplitude is too small
            if fit_result['R_squared'] < r2_threshold or amplitude < amplitude_threshold:
                if model == 'hill':
                    fit_result = {'Kd': np.nan, 'Kd_Error': np.nan, 'Hill_n': np.nan, 
                                 'Hill_n_Error': np.nan, 'R_squared': fit_result['R_squared'],
                                 'Fit_Params': None, 'Model': 'hill'}
                else:
                    fit_result = {'Kd1': np.nan, 'Kd1_Error': np.nan, 'Kd2': np.nan,
                                 'Kd2_Error': np.nan, 'R_squared': fit_result['R_squared'],
                                 'Fit_Params': None, 'Model': 'two-site'}
        
        fit_result['Metal'] = metal
        fit_result['Temperature'] = 'WT'
        kd_list.append(fit_result)
        
        if override is not None:
            if exclude_high > 0:
                override_vals_fit = (1 - metal_data['Override Tm'].values)[fit_mask]
                override_errs_fit = metal_data['Override Tm Standard Error'].values[fit_mask]
            else:
                override_vals_fit = 1 - metal_data['Override Tm'].values
                override_errs_fit = metal_data['Override Tm Standard Error'].values
            fit_result = fit_binding_curve(concs_fit, override_vals_fit, override_errs_fit, model=model)
            
            # Quality control: reject fits with poor R² or low amplitude (non-binding)
            if not np.isnan(fit_result['R_squared']):
                if model == 'hill':
                    ymin, ymax = fit_result['Fit_Params'][1], fit_result['Fit_Params'][2]
                else:  # two-site
                    ymin, ymax = fit_result['Fit_Params'][2], fit_result['Fit_Params'][3]
                
                amplitude = abs(ymax - ymin)
                
                # Mark as N/A if R² is too low or amplitude is too small
                if fit_result['R_squared'] < r2_threshold or amplitude < amplitude_threshold:
                    if model == 'hill':
                        fit_result = {'Kd': np.nan, 'Kd_Error': np.nan, 'Hill_n': np.nan, 
                                     'Hill_n_Error': np.nan, 'R_squared': fit_result['R_squared'],
                                     'Fit_Params': None, 'Model': 'hill'}
                    else:
                        fit_result = {'Kd1': np.nan, 'Kd1_Error': np.nan, 'Kd2': np.nan,
                                     'Kd2_Error': np.nan, 'R_squared': fit_result['R_squared'],
                                     'Fit_Params': None, 'Model': 'two-site'}
            
            fit_result['Metal'] = metal
            fit_result['Temperature'] = 'Override'
            kd_list.append(fit_result)
    
    kd_results = pd.DataFrame(kd_list)
    return kd_df, kd_results

def plot_kds(df, kd_results, protein_name, model='hill'):
    has_override = 'Override Temperature' in df.columns
    
    plot_configs = [('WT Tm', 'WT Tm Temperature', 'WT')]
    if has_override:
        plot_configs.append(('Override Tm', 'Override Temperature', 'Override'))
    
    n_plots = len(plot_configs)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    conc_min = df[df['Concentration'] > 0]['Concentration'].min()
    conc_max = df['Concentration'].max()
    conc_fit = np.logspace(np.log10(conc_min), np.log10(conc_max), 200)

    for ax, (data_col, temp_col, kd_key) in zip(axes, plot_configs):
        temp_value = df[temp_col].iloc[0]
        
        for metal in sorted(df['Metal'].unique(), key=get_atomic_number):
            metal_data = df[df['Metal'] == metal].sort_values('Concentration')
            concentrations = metal_data['Concentration'].values
            values = 1 - metal_data[data_col].values  # Convert to % folded
            errors = metal_data[f'{data_col} Standard Error'].values
            
            color = metal_colors[metal]
            
            kd_row = kd_results[(kd_results['Metal'] == metal) & (kd_results['Temperature'] == kd_key)]
            if not kd_row.empty:
                popt = kd_row['Fit_Params'].values[0]
                
                if model == 'hill':
                    kd = kd_row['Kd'].values[0]
                    kd_err = kd_row['Kd_Error'].values[0]
                    n = kd_row['Hill_n'].values[0]
                    n_err = kd_row['Hill_n_Error'].values[0]
                    r2 = kd_row['R_squared'].values[0]
                    
                    if not np.isnan(kd):
                        if kd < 1.0:
                            kd_nm = kd * 1000
                            kd_err_nm = kd_err * 1000
                            label = f"{metal}: Kd={kd_nm:.0f}±{kd_err_nm:.0f} nM, n={n:.2f}±{n_err:.2f}, R²={r2:.3f}"
                        else:
                            label = f"{metal}: Kd={kd:.1f}±{kd_err:.1f} µM, n={n:.2f}±{n_err:.2f}, R²={r2:.3f}"
                    else:
                        label = f"{metal}: Kd=N/A"
                        
                elif model == 'two-site':
                    kd1 = kd_row['Kd1'].values[0]
                    kd1_err = kd_row['Kd1_Error'].values[0]
                    kd2 = kd_row['Kd2'].values[0]
                    kd2_err = kd_row['Kd2_Error'].values[0]
                    r2 = kd_row['R_squared'].values[0]
                    
                    if not np.isnan(kd1):
                        if kd1 < 1.0:
                            kd1_str = f"{kd1*1000:.0f}±{kd1_err*1000:.1f} nM"
                        else:
                            kd1_str = f"{kd1:.1f}±{kd1_err:.1f} µM"
                        if kd2 < 1.0:
                            kd2_str = f"{kd2*1000:.0f}±{kd2_err*1000:.1f} nM"
                        else:
                            kd2_str = f"{kd2:.1f}±{kd2_err:.1f} µM"
                        label = f"{metal}: Kd1={kd1_str}, Kd2={kd2_str}, R²={r2:.3f}"
                    else:
                        label = f"{metal}: Kd=N/A"
            else:
                popt = None
                label = f"{metal}: Kd=N/A"
            
            ax.errorbar(concentrations, values, yerr=errors, 
                       color=color, marker='o', linestyle='', 
                       capsize=3, alpha=0.7, markersize=6)
            
            if popt is not None:
                if model == 'hill':
                    fit_vals = binding_curve_hill(conc_fit, *popt)
                else:
                    fit_vals = binding_curve_two_site(conc_fit, *popt)
                ax.plot(conc_fit, fit_vals, color=color, linestyle='-', 
                       linewidth=2, alpha=0.8, label=label)
            else:
                ax.plot([], [], color=color, label=label)
        
        ax.set_xlabel('Concentration (µM)', fontsize=10)
        ax.set_ylabel('% Folded', fontsize=10)
        title_name = 'WT Tm' if kd_key == 'WT' else 'Override Temp'
        ax.set_title(f'Titration at {title_name} ({temp_value:.1f}°C)', fontsize=11)
        ax.set_xscale('log')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(protein_name, fontsize=12, y=0.995)
    
    plt.tight_layout()
    protein_lower = protein_name.lower()
    plt.savefig(f'{protein_lower}_metal_titrations.png', dpi=300)
    plt.show()

def plot_kd_bars(df, kd_results, protein_name, model='hill'):
    """Plot bar graph of inverse Kds for each metal"""
    # Determine which temperature to use (Override if present, otherwise WT)
    has_override = 'Override Temperature' in df.columns
    if has_override:
        temp_value = df['Override Temperature'].iloc[0]
        temp_key = 'Override'
    else:
        temp_value = df['WT Tm Temperature'].iloc[0]
        temp_key = 'WT'
    
    # Filter kd_results for the selected temperature
    kd_results_filtered = kd_results[kd_results['Temperature'] == temp_key].copy()
    
    # Get all metals and sort by atomic number
    all_metals = [m for m in kd_results_filtered['Metal'].unique() if m not in ['EDTA', 'Mix']]
    all_metals = sorted(all_metals, key=get_atomic_number)

    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_positions = np.arange(len(all_metals))
    bar_width = 0.8
    
    if model == 'two-site':
        bar_width = 0.4
    
    for metal_idx, metal in enumerate(all_metals):
        metal_data = kd_results_filtered[kd_results_filtered['Metal'] == metal]
        
        if metal_data.empty:
            continue
        
        base_color = metal_colors[metal]
        x_pos = x_positions[metal_idx]
        
        if model == 'hill':
            kd = metal_data['Kd'].values[0]
            kd_err = metal_data['Kd_Error'].values[0]
            
            # Set N.B. (no binding) to 10^-4 if Kd is NA
            if np.isnan(kd):
                ax.bar(x_pos, 1e-4, bar_width,
                       color=base_color, edgecolor='black', linewidth=1,
                       alpha=0.5, hatch='//')
                ax.text(x_pos, 1e-4 * 1.5, 'N.B.', ha='center', va='bottom',
                       fontsize=10, fontweight='bold', color='black')
                continue
            
            inverse_kd = 1 / kd if kd > 0 else 0
            
            # Asymmetric error bars for inverse (appropriate for log scale)
            if kd > 0 and kd_err > 0:
                lower_kd = max(kd - kd_err, kd * 0.01)  # Prevent negative or zero Kd
                upper_kd = kd + kd_err
                inverse_upper = 1 / lower_kd  # Higher Kd gives lower inverse
                inverse_lower = 1 / upper_kd  # Lower Kd gives higher inverse
                yerr_lower = inverse_kd - inverse_lower
                yerr_upper = inverse_upper - inverse_kd
            else:
                yerr_lower = 0
                yerr_upper = 0
            
            ax.bar(x_pos, inverse_kd, bar_width,
                   color=base_color, edgecolor='black', linewidth=1)
            ax.errorbar(x_pos, inverse_kd, yerr=[[yerr_lower], [yerr_upper]],
                       fmt='none', ecolor='black', capsize=5, capthick=2)
        
        else:  # two-site
            kd1 = metal_data['Kd1'].values[0]
            kd1_err = metal_data['Kd1_Error'].values[0]
            kd2 = metal_data['Kd2'].values[0]
            kd2_err = metal_data['Kd2_Error'].values[0]
            
            if np.isnan(kd1):
                ax.bar(x_pos, 1e-4, bar_width,
                       color=base_color, edgecolor='black', linewidth=1,
                       alpha=0.5, hatch='//')
                ax.text(x_pos, 1e-4 * 1.1, 'N.B.', ha='center', va='bottom',
                       fontsize=10, fontweight='bold', color='black')
                continue
            
            inverse_kd1 = 1 / kd1 if kd1 > 0 else 0
            inverse_kd2 = 1 / kd2 if kd2 > 0 else 0
            
            # Asymmetric error bars for inverse Kd1
            if kd1 > 0 and kd1_err > 0:
                lower_kd1 = max(kd1 - kd1_err, kd1 * 0.01)
                upper_kd1 = kd1 + kd1_err
                inverse_upper1 = 1 / lower_kd1
                inverse_lower1 = 1 / upper_kd1
                yerr_lower1 = inverse_kd1 - inverse_lower1
                yerr_upper1 = inverse_upper1 - inverse_kd1
            else:
                yerr_lower1 = 0
                yerr_upper1 = 0
            
            # Asymmetric error bars for inverse Kd2
            if kd2 > 0 and kd2_err > 0:
                lower_kd2 = max(kd2 - kd2_err, kd2 * 0.01)
                upper_kd2 = kd2 + kd2_err
                inverse_upper2 = 1 / lower_kd2
                inverse_lower2 = 1 / upper_kd2
                yerr_lower2 = inverse_kd2 - inverse_lower2
                yerr_upper2 = inverse_upper2 - inverse_kd2
            else:
                yerr_lower2 = 0
                yerr_upper2 = 0
            
            base_rgb = matplotlib.colors.to_rgba(base_color)
            lighter_color = tuple(c * 0.5 + 0.5 for c in base_rgb[:3]) + (base_rgb[3],)
            
            offset = bar_width * 0.55
            ax.bar(x_pos - offset, inverse_kd1, bar_width,
                   color=base_color, edgecolor='black', linewidth=1,
                   label='Site 1' if metal_idx == 0 else '')
            ax.errorbar(x_pos - offset, inverse_kd1, yerr=[[yerr_lower1], [yerr_upper1]],
                       fmt='none', ecolor='black', capsize=5, capthick=2)
            ax.bar(x_pos + offset, inverse_kd2, bar_width,
                   color=lighter_color, edgecolor='black', linewidth=1,
                   label='Site 2' if metal_idx == 0 else '')
            ax.errorbar(x_pos + offset, inverse_kd2, yerr=[[yerr_lower2], [yerr_upper2]],
                       fmt='none', ecolor='black', capsize=5, capthick=2)
    
    ax.set_ylabel('Kd (M)', fontsize=12)
    ax.set_yscale('log')
    ax.set_ylim(10e-5,1000)
    ax.set_xlabel('Metal', fontsize=12)
    ax.set_title(f'DSF Binding Affinity for {protein_name} at {temp_value:.1f}°C', pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(all_metals, rotation=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Replace y-tick labels with Kd values in log10 format (convert µM to M)
    yticks = ax.get_yticks()
    kd_labels = []
    for tick in yticks:
        if tick > 0:
            kd_val_um = 1/tick  # Kd in µM
            kd_val_m = kd_val_um / 1e6  # Convert to M
            log_kd = np.log10(kd_val_m)
            # Format as 10^x with superscript
            if log_kd == int(log_kd):
                kd_labels.append(f'10$^{{{int(log_kd)}}}$')
            else:
                kd_labels.append(f'10$^{{{log_kd:.1f}}}$')
        else:
            kd_labels.append('∞')
    ax.set_yticklabels(kd_labels)
    
    if model == 'two-site':
        ax.legend(fontsize=9, loc='best')
    
    plt.tight_layout()
    protein_lower = protein_name.lower()
    plt.savefig(f'{protein_lower}_kd_bar_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results_csv(titration_df, kd_results, protein_name):
    """Save titration and Kd results to CSV files"""
    protein_lower = protein_name.lower()
    titration_df.to_csv(f'{protein_lower}_titration_data.csv', index=False)
    
    # Reorder kd_results columns and convert superscripts
    kd_results_export = kd_results.copy()
    kd_results_export['Metal'] = kd_results_export['Metal'].str.replace('²⁺', '2+').str.replace('³⁺', '3+')
    
    # Convert 'WT' and 'Override' labels to actual temperature values
    wt_temp = titration_df['WT Tm Temperature'].iloc[0]
    kd_results_export.loc[kd_results_export['Temperature'] == 'WT', 'Temperature'] = wt_temp
    if 'Override Temperature' in titration_df.columns:
        override_temp = titration_df['Override Temperature'].iloc[0]
        kd_results_export.loc[kd_results_export['Temperature'] == 'Override', 'Temperature'] = override_temp
    
    cols = kd_results_export.columns.tolist()
    cols.remove('Metal')
    cols.remove('Temperature')
    kd_results_export = kd_results_export[['Metal', 'Temperature'] + cols]
    kd_results_export.to_csv(f'{protein_lower}_kd_results.csv', index=False)

if __name__ == '__main__':
    main()