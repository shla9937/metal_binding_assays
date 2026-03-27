#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42  # Embed fonts as vectors
matplotlib.rcParams['pdf.use14corefonts'] = False
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
    parser.add_argument('-ht', '--high_temp', type=float, required=False, help="Exclude temps over this value")
    parser.add_argument('-lt', '--low_temp', type=float, required=False, help="Exclude temps under this value")
    parser.add_argument('-o', '--override', type=float, required=False, help="Override analysis temperature")
    parser.add_argument('-x', '--exclude_high', type=int, default=0, help="Number of highest concentrations to exclude from fitting")
    parser.add_argument('-l', '--exclude_low', type=int, default=0, help="Number of lowest concentrations to exclude from fitting")
    parser.add_argument('-m', '--model', type=str, default='hill', choices=['hill', 'two-site', 'quadratic'], 
                        help="Binding model: 'hill' (Hill equation with n), 'two-site' (two independent sites), or 'quadratic' (quadratic binding, accounts for ligand depletion)")
    parser.add_argument('-w', '--exclude_wells', type=str, nargs='+', default=[],
                        help="Well positions to exclude from analysis (e.g. A1 B3 C12)")
    parser.add_argument('-T', '--tm_threshold', type=float, default=1.0,
                        help="Minimum Tm change (°C) required to consider binding (default: 3.0)")
    parser.add_argument('-r', '--r2_threshold', type=float, default=0.7,
                        help="Minimum R² required to accept a fit (default: 0.7)")
    args = parser.parse_args()
    
    df = parse_csv_file(args.csv)
    if args.high_temp:
        df = exclude_high_temps(df, args.high_temp)
    if args.low_temp:
        df = exclude_low_temps(df, args.low_temp)
    # df = exclude_gap_temps(df, x, y)
    raw_df = assign_conc(df)
    if args.exclude_wells:
        raw_df = exclude_wells(raw_df, args.exclude_wells)
    if args.exclude_high > 0:
        raw_df = exclude_high_conc(raw_df, args.exclude_high)
    if args.exclude_low > 0:
        raw_df = exclude_low_conc(raw_df, args.exclude_low)
    smoothed_df = smooth_wells(raw_df)
    norm_df = normalize(smoothed_df)
    avg_df = average(norm_df)
    avg_tm_df = find_tms(avg_df)
    titration_df, kd_results = find_kds(avg_tm_df, override=args.override, model=args.model, tm_threshold=args.tm_threshold, r2_threshold=args.r2_threshold)
    plot_df(raw_df, 'Fluorescence', args.protein)
    plot_df(avg_tm_df, 'Smoothed Fluorescence', args.protein, error_column='Standard Error', override=args.override)
    plot_tms(avg_tm_df, args.protein)
    plot_kds(titration_df, kd_results, args.protein, model=args.model)
    plot_kd_bars(titration_df, kd_results, args.protein, model=args.model)
    save_results_csv(titration_df, kd_results, args.protein)
    save_command_txt(args.protein)

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

def exclude_high_temps(df, exclude):
    return df[df['Temperature'] <= exclude]

def exclude_low_temps(df, exclude):
    return df[df['Temperature'] >= exclude]

def exclude_gap_temps(df, gap_low, gap_high):
    """Exclude temperatures within a gap range (keep temps outside the gap)"""
    return df[(df['Temperature'] < gap_low) | (df['Temperature'] > gap_high)]

def exclude_high_conc(df, exclude_high):
    """Remove the N highest concentrations from the dataframe (excluding 0 concentration)"""
    # Get all unique concentrations (excluding 0)
    unique_concs = sorted([c for c in df['Concentration'].unique() if c > 0], reverse=True)
    
    # Get the N highest concentrations to exclude
    if exclude_high >= len(unique_concs):
        # If trying to exclude all or more, exclude all except the lowest
        concs_to_exclude = unique_concs[:-1]
    else:
        concs_to_exclude = unique_concs[:exclude_high]
    
    # Remove rows with those concentrations
    df_filtered = df[~df['Concentration'].isin(concs_to_exclude)]
    
    return df_filtered

def exclude_wells(df, wells):
    """Remove specific well positions from the dataframe"""
    wells_normalised = [w.strip().upper() for w in wells]
    return df[~df['Well Position'].str.upper().isin(wells_normalised)]

def exclude_low_conc(df, exclude_low):
    unique_concs = sorted([c for c in df['Concentration'].unique() if c > 0], reverse=True)
    # Get the N highest concentrations to exclude
    if exclude_low >= len(unique_concs):
        # If trying to exclude all or more, exclude all except the lowest
        concs_to_exclude = unique_concs[:-1]
    else:
        concs_to_exclude = unique_concs[-exclude_low:]
    # Remove rows with those concentrations
    df_filtered = df[~df['Concentration'].isin(concs_to_exclude)]
    
    return df_filtered

def assign_conc(df):
    global metals_left, metals_right, rows, metal_colors, protein_conc
    df['Metal'] = None
    df['Concentration'] = np.nan
    protein_conc = 5  # µM
    # which metal is being titrated in columns 1-12, A-P)
    metals_left = ["Mn²⁺", "Mn²⁺", "Co²⁺", "Co²⁺", "Ni²⁺", "Ni²⁺", "Cu²⁺", "Cu²⁺",
                   "Nd³⁺", "Nd³⁺", "Dy³⁺", "Dy³⁺", "EDTA", "EDTA", "Apo", "Apo"]
    # which metal is being titrated in columns 13-24, A-P)
    metals_right = ["Mn²⁺", "Mn²⁺", "Co²⁺", "Co²⁺", "Ni²⁺", "Ni²⁺", "Cu²⁺", "Cu²⁺",
                   "Nd³⁺", "Nd³⁺", "Dy³⁺", "Dy³⁺", "EDTA", "EDTA", "Apo", "Apo"]
    concentrations = [100, 50.0, 25.0, 12.5, 6.25, 3.13, 1.56, 0.781, 0.391, 0.195, 0.0977, 0.0488]
    rows = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P"]
    metal_colors = {
        # Colored aqueous chloride solutions
        "Mn²⁺": "#FF1493",    # deep pink (MnCl2 aq is faint pink)
        "Co²⁺": "#8B008B",    # dark magenta (CoCl2 aq is deep pink/magenta)
        "Ni²⁺": "#00C853",    # bright green (NiCl2 aq is green)
        "Cu²⁺": "#1E90FF",    # dodger blue (CuCl2 aq is blue)
        "Nd³⁺": "#8A2BE2",    # blue violet (NdCl3 aq is lilac/purple)
        "Dy³⁺": "#FFD700",    # gold (DyCl3 aq is pale yellow)
        "Pr³⁺": "#80FF20",    # vibrant lime green (PrCl3 aq is yellow-green)
        "Er³⁺": "#FF4D80",    # vibrant pink (ErCl3 aq is distinctly pink)
        "Ho³⁺": "#FF9D00",    # vibrant amber (HoCl3 aq is distinctly yellow)
        # Non-metal controls
        "EDTA": "#FF6600",    # bright orange
        "Apo":  "#808080",    # medium gray
        # Colorless aqueous solutions — unique gray shades
        # Alkali metals
        "Li⁺":  "#DCE8F0",    # very light blue-gray
        "K⁺":   "#E8E8D8",    # very light warm gray
        "Rb⁺":  "#9090A8",    # medium blue-gray
        "Cs⁺":  "#707060",    # medium olive-gray
        # Alkaline earth metals
        "Mg²⁺": "#D8D8D0",    # very light warm gray
        "Ca²⁺": "#C8D0D8",    # light cool gray
        "Sr²⁺": "#A8B0A8",    # medium green-gray
        "Ba²⁺": "#585858",    # dark gray
        # Group 3 / early transition
        "Sc³⁺": "#989898",    # medium gray
        "Y³⁺":  "#686868",    # medium-dark gray
        "Zn²⁺": "#C0C0C0",    # silver
        "La³⁺": "#484858",    # dark blue-gray
        # Colorless lanthanides
        "Ce³⁺": "#A8C0B0",    # light teal-gray
        "Sm³⁺": "#C8C0B0",    # light warm tan-gray
        "Eu³⁺": "#C0B0C0",    # light mauve-gray
        "Gd³⁺": "#686060",    # dark warm gray
        "Tb³⁺": "#606868",    # dark teal-gray
        "Tm³⁺": "#A0A898",    # medium warm olive-gray
        "Yb³⁺": "#787070",    # medium dark warm gray
        "Lu³⁺": "#404040",    # very dark gray
    }

    for row in rows:
        for well in range(1,13):
            well_pos = row+str(well)
            df.loc[df['Well Position'] == well_pos, 'Metal'] = metals_left[rows.index(row)]
            df.loc[df['Well Position'] == well_pos, 'Concentration'] = concentrations[well-1]            
        for well in range(13,25):
            well_pos = row+str(well)
            df.loc[df['Well Position'] == well_pos, 'Metal'] = metals_right[rows.index(row)]
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

def plot_df(df, y_column, protein_name, error_column=None, override=None):    
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
                color = tuple(c * (1 - conc_idx / max(len(concentrations)-1, 1)) for c in color[:3]) + (color[3],)
                
                ax.errorbar(conc_data['Temperature'], conc_data[y_column], 
                        yerr=conc_data[error_column], label=f"{conc:.3g} µM", 
                        color=color, alpha=0.3)
            
            apo_tm = df[df['Metal'] == 'Apo'].groupby('Concentration')['Tm'].first().mean()
            
            # Use override temperature if provided, otherwise use Apo Tm
            if override:
                plot_temp = float(override)
                temp_label = f'Override Tm ({plot_temp:.1f}°C)'
            else:
                plot_temp = apo_tm
                temp_label = f'Apo Tm ({apo_tm:.1f}°C)'
            
            ax.axvline(x=plot_temp, color='black', linestyle='--', linewidth=1.5, alpha=0.8, label=temp_label)
            
            ax.set_xlabel('Temperature (°C)', fontsize=10)
            ax.set_ylabel(y_column, fontsize=10)
            ax.set_title(f"{metal}", fontsize=11)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        for idx in range(n_metals, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(protein_name, fontsize=12, y=0.995)
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
                    
                    # Skip wells with no data (e.g., excluded by exclude_high)
                    if len(well_data) == 0:
                        continue
                    
                    conc = well_data['Concentration'].iloc[0]
                    conc_idx = concentrations.index(conc) if conc in concentrations else 0
                    
                    color = matplotlib.colors.to_rgba(metal_colors[metal])
                    color = tuple(c * (1 - conc_idx / max(len(concentrations)-1, 1)) for c in color[:3]) + (color[3],)
                    
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
    save_vector_pdf(f"{protein_lower}_{y_column.lower().replace(' ', '_')}_melt_curves.pdf")
    plt.show()

def smooth_wells(df):
    """Apply Savitzky-Golay smoothing per well to raw fluorescence."""
    df = df.copy()
    def _smooth(x):
        wl = min(len(x), 51)
        if wl % 2 == 0:
            wl -= 1
        if wl >= 5:
            return savgol_filter(x.values, wl, 3)
        return x.values
    df['Smoothed Fluorescence'] = df.groupby('Well')['Fluorescence'].transform(_smooth)
    return df

def normalize(df):
    df = df.copy()
    scaler = MinMaxScaler()
    df['Normalized Fluorescence'] = df.groupby('Well')['Smoothed Fluorescence'].transform(
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
        
        # Data is already smoothed per well before normalization; compute derivative directly
        derivative = np.gradient(fluor, temps)
        
        # Find peak of derivative (steepest slope = Tm)
        peak_idx = np.argmax(derivative)
        tm = temps[peak_idx]
        
        tm_values.append(tm)
        smoothed_values.append(fluor)
        group_indices.append(group.index)
    
    # Add Tm and smoothed fluorescence columns to dataframe
    df['Tm'] = np.nan
    df['Smoothed Fluorescence'] = np.nan
    for tm, smoothed, indices in zip(tm_values, smoothed_values, group_indices):
        df.loc[indices, 'Tm'] = tm
        df.loc[indices, 'Smoothed Fluorescence'] = smoothed

    return df

def plot_tms(df, protein_name):
    fig, ax = plt.subplots(figsize=(4, 5))
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
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), fontsize=6, ncol=6,
              frameon=True, borderaxespad=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.suptitle(protein_name, fontsize=10, y=0.98)
    
    plt.tight_layout()
    protein_lower = protein_name.lower()
    save_vector_pdf(f'{protein_lower}_tm_vs_concentration.pdf')
    plt.show()

def binding_curve_hill(conc, kd, ymin, ymax, n):
    """Hill equation with Hill coefficient n for cooperativity"""
    return ymin + (ymax - ymin) * conc**n / (kd**n + conc**n)

def binding_curve_two_site(conc, kd1, kd2, ymin, ymax):
    """Two independent binding sites with different affinities"""
    site1 = conc / (kd1 + conc)
    site2 = conc / (kd2 + conc)
    return ymin + (ymax - ymin) * 0.5 * (site1 + site2)

def binding_curve_quadratic(conc, kd, ymin, ymax):
    """Quadratic binding equation (Bai et al. 2018) accounting for ligand depletion"""
    global protein_conc
    Pt = protein_conc  # Total protein concentration in µM
    Lt = conc  # Total ligand (metal) concentration in µM
    fraction_bound = ((Pt + Lt + kd) - np.sqrt((Pt + Lt + kd)**2 - 4*Pt*Lt)) / (2*Pt)
    return ymin + (ymax - ymin) * fraction_bound

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
        
        elif model == 'quadratic':
            popt, pcov = curve_fit(
                binding_curve_quadratic,
                concentrations,
                values,
                p0=[kd_guess, ymin_guess, ymax_guess],
                sigma=errors,
                absolute_sigma=True,
                maxfev=10000,
                bounds=([0, 0, 0], [np.inf, 1, 1])
            )
            kd, ymin, ymax = popt
            perr = np.sqrt(np.diag(pcov))
            kd_err = perr[0] * 1.96
            
            y_pred = binding_curve_quadratic(concentrations, *popt)
            ss_res = np.sum((values - y_pred)**2)
            ss_tot = np.sum((values - np.mean(values))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return {'Kd': kd, 'Kd_Error': kd_err, 'R_squared': r_squared, 'Fit_Params': popt, 'Model': 'quadratic'}
    except:
        if model == 'hill':
            return {'Kd': np.nan, 'Kd_Error': np.nan, 'Hill_n': np.nan, 'Hill_n_Error': np.nan, 
                    'R_squared': np.nan, 'Fit_Params': None, 'Model': 'hill'}
        elif model == 'quadratic':
            return {'Kd': np.nan, 'Kd_Error': np.nan,
                    'R_squared': np.nan, 'Fit_Params': None, 'Model': 'quadratic'}
        else:
            return {'Kd1': np.nan, 'Kd1_Error': np.nan, 'Kd2': np.nan, 'Kd2_Error': np.nan,
                    'R_squared': np.nan, 'Fit_Params': None, 'Model': 'two-site'}

def find_kds(df, override=None, model='hill', tm_threshold=3.0, r2_threshold=0.7):
    apo_tm = df[df['Metal'] == 'Apo'].groupby('Concentration')['Tm'].first().mean()
    kd_data = []
    
    for (metal, conc), group in df.groupby(['Metal', 'Concentration']):
        group = group.sort_values('Temperature')
        temps = group['Temperature'].values
        smoothed_fluor = group['Smoothed Fluorescence'].values
        std_err = group['Standard Error'].values
        
        apo_idx = np.argmin(np.abs(temps - apo_tm))
        apo_fluor = smoothed_fluor[apo_idx]
        apo_se = std_err[apo_idx]
        
        data_dict = {
            'Metal': metal,
            'Concentration': conc,
            'Apo Tm Temperature': apo_tm,
            'Apo Tm': apo_fluor,
            'Apo Tm Standard Error': apo_se
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

        # Compute Tm change as |Tm at highest included concentration - Apo Tm|
        included_concs = metal_data['Concentration'].values
        metal_tms = df[(df['Metal'] == metal) & (df['Concentration'].isin(included_concs))].groupby('Concentration')['Tm'].first()
        highest_conc = metal_tms.idxmax()
        tm_change = abs(metal_tms[highest_conc] - apo_tm)

        concs = metal_data['Concentration'].values
        apo_vals = 1 - metal_data['Apo Tm'].values
        apo_errs = metal_data['Apo Tm Standard Error'].values
        
        fit_result = fit_binding_curve(concs, apo_vals, apo_errs, model=model)
        
        # Quality control: reject fits with poor R² or insufficient Tm change
        if not np.isnan(fit_result['R_squared']):
            if fit_result['R_squared'] < r2_threshold or tm_change < tm_threshold:
                if model == 'hill':
                    fit_result = {'Kd': np.nan, 'Kd_Error': np.nan, 'Hill_n': np.nan,
                                 'Hill_n_Error': np.nan, 'R_squared': fit_result['R_squared'],
                                 'Fit_Params': None, 'Model': 'hill'}
                elif model == 'quadratic':
                    fit_result = {'Kd': np.nan, 'Kd_Error': np.nan,
                                 'R_squared': fit_result['R_squared'],
                                 'Fit_Params': None, 'Model': 'quadratic'}
                else:
                    fit_result = {'Kd1': np.nan, 'Kd1_Error': np.nan, 'Kd2': np.nan,
                                 'Kd2_Error': np.nan, 'R_squared': fit_result['R_squared'],
                                 'Fit_Params': None, 'Model': 'two-site'}
        
        fit_result['Metal'] = metal
        fit_result['Temperature'] = 'Apo'
        kd_list.append(fit_result)
        
        if override is not None:
            override_vals = 1 - metal_data['Override Tm'].values
            override_errs = metal_data['Override Tm Standard Error'].values
            fit_result = fit_binding_curve(concs, override_vals, override_errs, model=model)
            
            # Quality control: reject fits with poor R² or insufficient Tm change
            if not np.isnan(fit_result['R_squared']):
                if fit_result['R_squared'] < r2_threshold or tm_change < tm_threshold:
                    if model == 'hill':
                        fit_result = {'Kd': np.nan, 'Kd_Error': np.nan, 'Hill_n': np.nan, 
                                     'Hill_n_Error': np.nan, 'R_squared': fit_result['R_squared'],
                                     'Fit_Params': None, 'Model': 'hill'}
                    elif model == 'quadratic':
                        fit_result = {'Kd': np.nan, 'Kd_Error': np.nan,
                                     'R_squared': fit_result['R_squared'],
                                     'Fit_Params': None, 'Model': 'quadratic'}
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
    
    plot_configs = [('Apo Tm', 'Apo Tm Temperature', 'Apo')]
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
                            label = f"{metal}: Kd={kd*1000:.0f}nM, n={n:.2f}, R²={r2:.3f}"
                        else:
                            label = f"{metal}: Kd={kd:.1f}µM, n={n:.2f}, R²={r2:.3f}"
                    else:
                        label = f"{metal}: Kd=N/A"
                
                elif model == 'quadratic':
                    kd = kd_row['Kd'].values[0]
                    kd_err = kd_row['Kd_Error'].values[0]
                    r2 = kd_row['R_squared'].values[0]
                    
                    if not np.isnan(kd):
                        if kd < 1.0:
                            label = f"{metal}: Kd={kd*1000:.0f}nM (quad), R²={r2:.3f}"
                        else:
                            label = f"{metal}: Kd={kd:.1f}µM (quad), R²={r2:.3f}"
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
                            kd1_str = f"{kd1*1000:.0f}nM"
                        else:
                            kd1_str = f"{kd1:.1f}µM"
                        if kd2 < 1.0:
                            kd2_str = f"{kd2*1000:.0f}nM"
                        else:
                            kd2_str = f"{kd2:.1f}µM"
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
                elif model == 'quadratic':
                    fit_vals = binding_curve_quadratic(conc_fit, *popt)
                else:
                    fit_vals = binding_curve_two_site(conc_fit, *popt)
                ax.plot(conc_fit, fit_vals, color=color, linestyle='-', 
                       linewidth=2, alpha=0.8, label=label)
            else:
                ax.plot([], [], color=color, label=label)
        
        ax.set_xlabel('Concentration (µM)', fontsize=10)
        ax.set_ylabel('% Folded', fontsize=10)
        title_name = 'Apo Tm' if kd_key == 'Apo' else 'Override Temp'
        ax.set_title(f'{protein_name} - Titration at {title_name} ({temp_value:.1f}°C)', fontsize=10)
        ax.set_xscale('log')
        ax.legend(fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.2),
                  ncol=3, frameon=True, borderaxespad=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    protein_lower = protein_name.lower()
    save_vector_pdf(f'{protein_lower}_metal_titrations.pdf')
    plt.show()

def plot_kd_bars(df, kd_results, protein_name, model='hill'):
    """Plot bar graph of inverse Kds for each metal"""
    # Determine which temperature to use (Override if present, otherwise Apo)
    has_override = 'Override Temperature' in df.columns
    if has_override:
        temp_value = df['Override Temperature'].iloc[0]
        temp_key = 'Override'
    else:
        temp_value = df['Apo Tm Temperature'].iloc[0]
        temp_key = 'Apo'
    
    # Filter kd_results for the selected temperature
    kd_results_filtered = kd_results[kd_results['Temperature'] == temp_key].copy()
    
    # Get all metals and sort by atomic number
    all_metals = [m for m in kd_results_filtered['Metal'].unique() if m not in ['EDTA', 'Apo']]
    all_metals = sorted(all_metals, key=get_atomic_number)

    fig, ax = plt.subplots(figsize=(4, 3))
    
    x_positions = np.arange(len(all_metals))
    bar_width = 0.6
    
    if model == 'two-site':
        bar_width = 0.4
    
    # On a log y-axis, bars default to bottom=0 which maps to -infinity in the PDF
    # geometry, making objects huge when opened in Illustrator. Using a finite bottom
    # value just below the visible axis floor prevents this.
    bar_bottom = 5e-5  # half-decade below the ylim floor of 1e-4
    
    for metal_idx, metal in enumerate(all_metals):
        metal_data = kd_results_filtered[kd_results_filtered['Metal'] == metal]
        
        if metal_data.empty:
            continue
        
        base_color = metal_colors[metal]
        x_pos = x_positions[metal_idx]
        
        if model in ('hill', 'quadratic'):
            kd = metal_data['Kd'].values[0]
            kd_err = metal_data['Kd_Error'].values[0]
            
            # Set NB (no binding) to 10^-4 if Kd is NA
            if np.isnan(kd):
                ax.bar(x_pos, 1e-4 - bar_bottom, bar_width,
                       bottom=bar_bottom, color=base_color, edgecolor='black', linewidth=1,
                       alpha=0.5, hatch='//')
                ax.text(x_pos, 1e-4 * 1.5, 'NB', ha='center', va='bottom',
                       fontsize=6, fontweight='bold', color='black')
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
            
            ax.bar(x_pos, inverse_kd - bar_bottom, bar_width,
                   bottom=bar_bottom, color=base_color, edgecolor='black', linewidth=1)
            ax.errorbar(x_pos, inverse_kd, yerr=[[yerr_lower], [yerr_upper]],
                       fmt='none', ecolor='black', capsize=4, capthick=1)
        
        else:  # two-site
            kd1 = metal_data['Kd1'].values[0]
            kd1_err = metal_data['Kd1_Error'].values[0]
            kd2 = metal_data['Kd2'].values[0]
            kd2_err = metal_data['Kd2_Error'].values[0]
            
            if np.isnan(kd1):
                ax.bar(x_pos, 1e-4 - bar_bottom, bar_width,
                       bottom=bar_bottom, color=base_color, edgecolor='black', linewidth=1,
                       alpha=0.5, hatch='//')
                ax.text(x_pos, 1e-4 * 1.1, 'NB', ha='center', va='bottom',
                       fontsize=6, fontweight='bold', color='black')
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
            ax.bar(x_pos - offset, inverse_kd1 - bar_bottom, bar_width,
                   bottom=bar_bottom, color=base_color, edgecolor='black', linewidth=1,
                   label='Site 1' if metal_idx == 0 else '')
            ax.errorbar(x_pos - offset, inverse_kd1, yerr=[[yerr_lower1], [yerr_upper1]],
                       fmt='none', ecolor='black', capsize=4, capthick=1)
            ax.bar(x_pos + offset, inverse_kd2 - bar_bottom, bar_width,
                   bottom=bar_bottom, color=lighter_color, edgecolor='black', linewidth=1,
                   label='Site 2' if metal_idx == 0 else '')
            ax.errorbar(x_pos + offset, inverse_kd2, yerr=[[yerr_lower2], [yerr_upper2]],
                       fmt='none', ecolor='black', capsize=4, capthick=1)
    
    ax.set_ylabel('Kd (M)', fontsize=10)
    ax.set_yscale('log')
    ax.set_ylim(10e-5, 1000)
    ax.set_title(f'{protein_name} - DSF Binding Affinity at {temp_value:.1f}°C', pad=20)
    ax.set_xlim(-0.5, len(all_metals) - 0.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(all_metals, rotation=0, fontsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set explicit ticks at clean Kd decade values in M, converted to inverse-Kd (µM⁻¹) plot units
    kd_decades_m = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    tick_positions = [1 / (kd_m * 1e6) for kd_m in kd_decades_m]  # convert M → µM, then invert
    ymin, ymax = ax.get_ylim()
    tick_positions = [t for t in tick_positions if ymin <= t <= ymax]
    kd_labels = [f'10$^{{{int(np.log10(kd))}}}$' for kd in kd_decades_m
                 if ymin <= 1 / (kd * 1e6) <= ymax]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(kd_labels)
    
    if model == 'two-site':
        ax.legend(fontsize=8, loc='best')
    
    plt.tight_layout()
    protein_lower = protein_name.lower()
    save_vector_pdf(f'{protein_lower}_kd_bar_chart.pdf')
    plt.show()

def save_vector_pdf(filename):
    """Save current figure as a fully vector PDF by disabling per-artist rasterization."""
    fig = plt.gcf()
    for ax in fig.get_axes():
        for artist in ax.get_children():
            if hasattr(artist, 'set_rasterized'):
                artist.set_rasterized(False)
    fig.savefig(filename, bbox_inches='tight', backend='pdf')

def save_command_txt(protein_name):
    """Save the command used to run this analysis"""
    import sys
    protein_lower = protein_name.lower()
    command = ' '.join(sys.argv)
    with open(f'{protein_lower}_command.txt', 'w') as f:
        f.write(command + '\n')

def save_results_csv(titration_df, kd_results, protein_name):
    """Save titration and Kd results to CSV files"""
    protein_lower = protein_name.lower()
    titration_df.to_csv(f'{protein_lower}_titration_data.csv', index=False)
    
    # Reorder kd_results columns and convert superscripts
    kd_results_export = kd_results.copy()
    kd_results_export['Metal'] = kd_results_export['Metal'].str.replace('²⁺', '2+').str.replace('³⁺', '3+')
    
    # Convert 'Apo' and 'Override' labels to actual temperature values
    apo_temp = titration_df['Apo Tm Temperature'].iloc[0]
    kd_results_export.loc[kd_results_export['Temperature'] == 'Apo', 'Temperature'] = apo_temp
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