#!/usr/bin/env python
"""
DSF Binding Analysis Script
- Import raw fluorescence data from 384-well DSF plate
- Calculate smoothed first derivative (melt curves)
- Find WT Tm from averaged WT wells
- Normalize raw data to WT
- For each metal titration: extract fluorescence at WT Tm and fit binding curves
- Plot all binding curves and generate Kd values
- Export results for Tm bar graph visualization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import argparse


def get_plate_configuration():
    """
    Hardcoded plate configuration for metal binding titrations.
    
    Returns:
        dict: Configuration with structure:
        {
            'concentrations': [array of concentrations in M],
            'metals': {
                'row_letter': {
                    'cols_1_12': 'metal_name',  # Metal for columns 1-12
                    'cols_13_24': 'metal_name'  # Metal for columns 13-24
                }
            }
        }
    """
    config = {
        # Concentrations for each titration (same for all metals)
        # Example: 10-fold dilution series from 1 mM down to 100 pM
        'concentrations': np.array([
            1e-3,      # 1 mM
            1e-4,      # 100 µM
            1e-5,      # 10 µM
            1e-6,      # 1 µM
            1e-7,      # 100 nM
            1e-8,      # 10 nM
            1e-9,      # 1 nM
            1e-10,     # 100 pM
            1e-11,     # 10 pM
            1e-12,     # 1 pM
            0.0,       # Blank/control (no ligand)
            0.0,       # Blank/control (no ligand)
        ]),
        
        # Metal assignments per row and well range
        'metals': {
            'A': {'cols_1_12': 'Al³⁺', 'cols_13_24': 'K⁺'},
            'B': {'cols_1_12': 'Ce³⁺', 'cols_13_24': 'Ca²⁺'},
            'C': {'cols_1_12': 'Co²⁺', 'cols_13_24': 'Mg²⁺'},
            'D': {'cols_1_12': 'Cu²⁺', 'cols_13_24': 'Mn²⁺'},
            'E': {'cols_1_12': 'Fe³⁺', 'cols_13_24': 'Mo⁶⁺'},
            'F': {'cols_1_12': 'Pt⁴⁺', 'cols_13_24': 'Ni²⁺'},
            'G': {'cols_1_12': 'Zn²⁺', 'cols_13_24': 'VO²⁺'},
            'H': {'cols_1_12': 'Cu⁺', 'cols_13_24': None},
            'I': {'cols_1_12': None, 'cols_13_24': None},
            'J': {'cols_1_12': None, 'cols_13_24': None},
            'K': {'cols_1_12': None, 'cols_13_24': None},
            'L': {'cols_1_12': None, 'cols_13_24': None},
            'M': {'cols_1_12': None, 'cols_13_24': None},
            'N': {'cols_1_12': None, 'cols_13_24': None},
            'O': {'cols_1_12': None, 'cols_13_24': None},
            'P': {'cols_1_12': None, 'cols_13_24': None},
        }
    }
    return config

def main():
    parser = argparse.ArgumentParser(description="DSF Binding Affinity Analysis")
    parser.add_argument('-f', '--file', type=str, required=True, help="Path to DSF .csv file")
    parser.add_argument('-o', '--output', type=str, default='.', help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Parse data
    print("Parsing DSF file...")
    df = parse_dsf_file(args.file)
    well_data = organize_by_well(df)
    
    # Get plate configuration
    config = get_plate_configuration()
    
    # Calculate WT Tm
    print("Calculating WT Tm...")
    wt_tm, wt_tms_list = get_wt_tm(well_data)
    print(f"WT Tm: {wt_tm:.2f}°C (from wells: {wt_tms_list})")
    
    # Get fluorescence at WT Tm for normalization
    wt_fluor_at_tm = extract_fluorescence_at_tm(well_data, wt_tm, 'A12')
    
    # Plot melt curves for all wells
    print("Plotting melt curves...")
    fig, axes = plt.subplots(4, 6, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (well_pos, data) in enumerate(sorted(well_data.items())):
        if idx >= 24:
            break
        ax = axes[idx]
        temp = data['temp']
        deriv = data['deriv']  # Use pre-calculated derivative
        
        ax.plot(temp, deriv, color='blue', alpha=0.7, label='Derivative')
        ax.axvline(wt_tm, color='red', linestyle='--', alpha=0.5, label='WT Tm')
        ax.set_title(well_pos, fontsize=8)
        ax.set_xlabel('Temperature (°C)', fontsize=7)
        ax.set_ylabel('Derivative', fontsize=7)
        ax.legend(fontsize=6)
    
    plt.suptitle('All Wells - Melt Curves (Derivative)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'melt_curves.png'), dpi=300)
    plt.close()
    
    # Extract binding data for each metal
    print("Extracting binding curves...")
    metal_data = {}
    binding_results = []
    
    for row_letter in config['metals'].keys():
        for col_range in ['cols_1_12', 'cols_13_24']:
            metal = config['metals'][row_letter][col_range]
            if metal is None:
                continue
            
            # Determine column range (1-12 or 13-24)
            if col_range == 'cols_1_12':
                cols = list(range(1, 13))
            else:
                cols = list(range(13, 25))
            
            # Extract fluorescence values at WT Tm for this metal titration
            fluorescence_values = []
            for col in cols:
                well_pos = f"{row_letter}{col}"
                fluor_at_tm = extract_fluorescence_at_tm(well_data, wt_tm, well_pos)
                fluorescence_values.append(fluor_at_tm)
            
            fluorescence_values = np.array(fluorescence_values)
            concentrations = config['concentrations']
            
            # Fit binding curve
            kd, h, r_squared, popt = fit_binding_curve(concentrations, fluorescence_values)
            
            if not np.isnan(kd):
                metal_data[metal] = (concentrations, fluorescence_values, kd, r_squared)
                binding_results.append({
                    'Metal': metal,
                    'Row': row_letter,
                    'Columns': f"{cols[0]}-{cols[-1]}",
                    'Kd (M)': kd,
                    'Hill_Coefficient': h,
                    'R_squared': r_squared
                })
                print(f"  {metal}: Kd = {kd:.2e} M, R² = {r_squared:.3f}")
            else:
                print(f"  {metal}: Fitting failed")
    
    # Plot all binding curves
    print("Plotting binding curves...")
    plot_binding_curves(metal_data, args.output)
    
    # Export results
    if binding_results:
        results_df = pd.DataFrame(binding_results)
        results_df.to_csv(os.path.join(args.output, 'binding_analysis_results.csv'), index=False)
        print(f"\nResults saved to {os.path.join(args.output, 'binding_analysis_results.csv')}")
        print(results_df.to_string())
    else:
        print("No binding curves successfully fitted.")

def parse_dsf_file(filepath):
    """
    Parse DSF export CSV file, skipping comment header lines.
    Expected columns: Well Position (or Well), Temperature, Fluorescence, Derivative
    Handles both .csv and .eds formats with comment headers.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    header_idx = None
    for i, line in enumerate(lines):
        if not line.startswith('#') and line.strip():
            header_idx = i
            break
    
    if header_idx is None:
        raise ValueError("Could not find data table header in file.")
    
    # Determine delimiter (tab or comma)
    if header_idx < len(lines):
        header_line = lines[header_idx]
        delimiter = '\t' if '\t' in header_line else ','
    else:
        delimiter = '\t'
    
    df = pd.read_csv(filepath, sep=delimiter, skiprows=header_idx)
    df.columns = [c.strip() for c in df.columns]
    
    # Handle different column name variations
    if 'Well' in df.columns and 'Well Position' not in df.columns:
        df.rename(columns={'Well': 'Well Position'}, inplace=True)
    
    # Ensure required columns exist
    required_cols = ['Well Position', 'Temperature', 'Fluorescence']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {list(df.columns)}")
    
    # If Derivative not in file, we can calculate it, but for .csv format it should be present
    if 'Derivative' not in df.columns:
        print("Warning: Derivative column not found. Calculation not implemented for CSV format.")
    
    return df


def organize_by_well(df):
    """
    Organize fluorescence and derivative data by well position.
    Returns dictionary: {well_position: {'temp': temps, 'fluor': fluor, 'deriv': deriv}}
    """
    well_data = {}
    for well_pos in df['Well Position'].unique():
        well_df = df[df['Well Position'] == well_pos].sort_values('Temperature')
        temps = well_df['Temperature'].values
        fluor = well_df['Fluorescence'].values
        deriv = well_df['Derivative'].values  # Use pre-calculated derivative
        well_data[well_pos] = {'temp': temps, 'fluor': fluor, 'deriv': deriv}
    return well_data


def find_tm_from_derivative(temp, derivative):
    """Find Tm as the peak of the negative derivative."""
    peak_idx = np.argmax(derivative)
    return temp[peak_idx]


def get_wt_tm(well_data, wt_wells=['A12', 'B12', 'C12']):
    """
    Calculate average Tm from WT control wells using the pre-calculated derivative.
    Returns average WT Tm and list of individual WT Tms.
    """
    wt_tms = []
    for well in wt_wells:
        if well in well_data:
            temp = well_data[well]['temp']
            deriv = well_data[well]['deriv']
            tm = find_tm_from_derivative(temp, deriv)
            wt_tms.append(tm)
    
    avg_wt_tm = np.mean(wt_tms)
    return avg_wt_tm, wt_tms


def normalize_fluorescence(fluor, wt_fluor_at_tm):
    """Normalize fluorescence to WT level at Tm."""
    return fluor / wt_fluor_at_tm if wt_fluor_at_tm > 0 else fluor


def extract_fluorescence_at_tm(well_data, tm, well_pos):
    """Extract fluorescence value at a specific temperature using interpolation."""
    if well_pos not in well_data:
        return np.nan
    
    temp = well_data[well_pos]['temp']
    fluor = well_data[well_pos]['fluor']
    
    # Linear interpolation to find fluorescence at tm
    f_interp = interp1d(temp, fluor, kind='linear', bounds_error=False, fill_value='extrapolate')
    return f_interp(tm)


def hill_equation(x, kd, h, bottom, top):
    """Hill equation for binding: y = bottom + (top - bottom) / (1 + (Kd/x)^h)"""
    return bottom + (top - bottom) / (1 + (kd / (x + 1e-12))**h)


def fit_binding_curve(concentrations, fluorescence):
    """
    Fit binding curve using Hill equation.
    Returns Kd, hill coefficient, R^2, and fit parameters.
    """
    # Remove NaN and zero concentrations
    mask = (~np.isnan(fluorescence)) & (concentrations > 0)
    conc_clean = concentrations[mask]
    fluor_clean = fluorescence[mask]
    
    if len(conc_clean) < 3:
        return np.nan, np.nan, np.nan, None
    
    try:
        # Initial parameter guesses
        bottom = np.min(fluor_clean)
        top = np.max(fluor_clean)
        kd_guess = np.median(conc_clean)
        
        p0 = [kd_guess, 1.0, bottom, top]
        bounds = ([1e-9, 0.1, 0, 0], [1e3, 4, np.inf, np.inf])
        
        popt, _ = curve_fit(hill_equation, conc_clean, fluor_clean, p0=p0, bounds=bounds, maxfev=5000)
        kd, h, bottom, top = popt
        
        # Calculate R^2
        predicted = hill_equation(conc_clean, kd, h, bottom, top)
        ss_res = np.sum((fluor_clean - predicted)**2)
        ss_tot = np.sum((fluor_clean - np.mean(fluor_clean))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return kd, h, r_squared, (kd, h, bottom, top)
    
    except Exception as e:
        print(f"Fitting error: {e}")
        return np.nan, np.nan, np.nan, None


def plot_binding_curves(metal_data, output_dir, title="Metal Binding Curves"):
    """
    Plot all metal binding curves on a single graph.
    metal_data: dict with metal names as keys and (concentrations, fluorescence, kd, r2) as values
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for metal, (conc, fluor, kd, r2) in metal_data.items():
        if np.isnan(kd) or r2 < 0.3:
            continue
        
        # Plot data points
        mask = (~np.isnan(fluor)) & (conc > 0)
        ax.scatter(conc[mask], fluor[mask], label=f'{metal} (Kd={kd:.2e}, R²={r2:.2f})', s=50)
        
        # Plot fit curve
        conc_fit = np.logspace(np.log10(np.min(conc[mask])), np.log10(np.max(conc[mask])), 100)
        fluor_fit = hill_equation(conc_fit, kd, 1.0, np.min(fluor[mask]), np.max(fluor[mask]))
        ax.plot(conc_fit, fluor_fit, linewidth=2)
    
    ax.set_xscale('log')
    ax.set_xlabel('Concentration (M)', fontsize=12)
    ax.set_ylabel('Normalized Fluorescence', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'binding_curves.png'), dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
