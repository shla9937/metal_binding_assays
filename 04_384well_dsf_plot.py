#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.signal import find_peaks
from scipy.optimize import curve_fit 
from matplotlib.colors import Normalize
import matplotlib.colors

def main():
    parser = argparse.ArgumentParser(
        description="Plot raw fluorescence data and negative derivative data from CSV files in a directory, with Tm detection."
    )
    parser.add_argument('-c', '--csvs', type=str, nargs='+', required=True, help="List of csvs from 01_dsf_import.py")
    parser.add_argument('-e', '--exclude', type=int, default=0, help="Exclude highests point")
    parser.add_argument('-t', '--title', type=str, required=True, help="Title")
    args = parser.parse_args()

    output_dir = os.path.join('..', 'outputs', args.title.replace(' ', '_'))
    os.makedirs(output_dir, exist_ok=True)
    metal_df = load_data(args.csvs)
    metal_df = plot_tm_scatter(metal_df, args.exclude, args.title, output_dir)
    plot_tm_bar(metal_df, args.title, output_dir)
    metal_df.to_csv(os.path.join(output_dir, f"{args.title.replace(' ', '_')}_tm_values.csv"), index=True)
    
def load_data(csv_files):
    # Load and concatenate files
    df_list = [pd.read_csv(csv_file, index_col=0) for csv_file in csv_files]
    metal_df = pd.concat(df_list, ignore_index=False, axis=1)

    # Get tab20 colors
    tab20_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    tab20_hex = [matplotlib.colors.rgb2hex(c) for c in tab20_colors]

    # Define explicit colors for each element using tab20 colors for each period
    element_colors = {
        # Period 1 (first two colors of tab20)
        'H': tab20_hex[0], 'He': tab20_hex[1],
        
        # Period 2 (next 8 colors of tab20)
        'Li': tab20_hex[2], 'Be': tab20_hex[3], 'B': tab20_hex[4], 'C': tab20_hex[5],
        'N': tab20_hex[6], 'O': tab20_hex[7], 'F': tab20_hex[8], 'Ne': tab20_hex[9],
        
        # Period 3 (next 8 colors of tab20, wrapping around if needed)
        'Na': tab20_hex[10], 'Mg': tab20_hex[11], 'Al': tab20_hex[12], 'Si': tab20_hex[13],
        'P': tab20_hex[14], 'S': tab20_hex[15], 'Cl': tab20_hex[16], 'Ar': tab20_hex[17],
        
        # Period 4 (cycle through tab20 colors)
        'K': tab20_hex[0], 'Ca': tab20_hex[1], 'Sc': tab20_hex[2], 'Ti': tab20_hex[3],
        'V': tab20_hex[4], 'Cr': tab20_hex[5], 'Mn': tab20_hex[6], 'Fe': tab20_hex[7],
        'Co': tab20_hex[8], 'Ni': tab20_hex[9], 'Cu': tab20_hex[10], 'Zn': tab20_hex[11],
        'Ga': tab20_hex[12], 'Ge': tab20_hex[13], 'As': tab20_hex[14], 'Se': tab20_hex[15],
        'Br': tab20_hex[16], 'Kr': tab20_hex[17],
        
        # Period 5 (cycle through tab20 colors again)
        'Rb': tab20_hex[0], 'Sr': tab20_hex[1], 'Y': tab20_hex[2], 'Zr': tab20_hex[3],
        'Nb': tab20_hex[4], 'Mo': tab20_hex[5], 'Tc': tab20_hex[6], 'Ru': tab20_hex[7],
        'Rh': tab20_hex[8], 'Pd': tab20_hex[9], 'Ag': tab20_hex[10], 'Cd': tab20_hex[11],
        'In': tab20_hex[12], 'Sn': tab20_hex[13], 'Sb': tab20_hex[14], 'Te': tab20_hex[15],
        'I': tab20_hex[16], 'Xe': tab20_hex[17],
        
        # Period 6 (including lanthanides)
        'Cs': tab20_hex[0], 'Ba': tab20_hex[1], 'La': tab20_hex[2], 'Ce': tab20_hex[3],
        'Pr': tab20_hex[4], 'Nd': tab20_hex[5], 'Pm': tab20_hex[6], 'Sm': tab20_hex[7],
        'Eu': tab20_hex[8], 'Gd': tab20_hex[9], 'Tb': tab20_hex[10], 'Dy': tab20_hex[11],
        'Ho': tab20_hex[12], 'Er': tab20_hex[13], 'Tm': tab20_hex[14], 'Yb': tab20_hex[15],
        'Lu': tab20_hex[16], 'Hf': tab20_hex[17], 'Ta': tab20_hex[18], 'W': tab20_hex[19],
        'Re': tab20_hex[0], 'Os': tab20_hex[1], 'Ir': tab20_hex[2], 'Pt': tab20_hex[3],
        'Au': tab20_hex[4], 'Hg': tab20_hex[5], 'Tl': tab20_hex[6], 'Pb': tab20_hex[7],
        'Bi': tab20_hex[8], 'Po': tab20_hex[9], 'At': tab20_hex[10], 'Rn': tab20_hex[11],
        
        # Period 7 (including actinides)
        'Fr': tab20_hex[0], 'Ra': tab20_hex[1], 'Ac': tab20_hex[2], 'Th': tab20_hex[3],
        'Pa': tab20_hex[4], 'U': tab20_hex[5], 'Np': tab20_hex[6], 'Pu': tab20_hex[7],
        'Am': tab20_hex[8], 'Cm': tab20_hex[9], 'Bk': tab20_hex[10], 'Cf': tab20_hex[11],
        'Es': tab20_hex[12], 'Fm': tab20_hex[13], 'Md': tab20_hex[14], 'No': tab20_hex[15],
        'Lr': tab20_hex[16], 'Rf': tab20_hex[17], 'Db': tab20_hex[18], 'Sg': tab20_hex[19],
        'Bh': tab20_hex[0], 'Hs': tab20_hex[1], 'Mt': tab20_hex[2], 'Ds': tab20_hex[3],
        'Rg': tab20_hex[4], 'Cn': tab20_hex[5], 'Nh': tab20_hex[6], 'Fl': tab20_hex[7],
        'Mc': tab20_hex[8], 'Lv': tab20_hex[9], 'Ts': tab20_hex[10], 'Og': tab20_hex[11]
    }

    def get_metal_symbol(col):
        return ''.join(c for c in col if c.isalpha())

    def get_metal_color(col):
        metal = get_metal_symbol(col)
        return element_colors.get(metal, '#333333')  # Default gray
    
    # Store colors in DataFrame for later use
    metal_df.attrs['colors'] = {col: get_metal_color(col) for col in metal_df.columns}
    
    print(metal_df)

    def get_oxidation_state(col):
        superscript_map = {'¹': 1, '²': 2, '³': 3, '⁴': 4, '⁵': 5, '⁶': 6, '⁷': 7}
        for char in col:
            if (char in superscript_map):
                return superscript_map[char]
        return 1 if '⁺' in col else 0
    
    # Define periodic table structure for atomic number lookup
    period_elements = {
        1: ['H', 'He'],
        2: ['Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne'],
        3: ['Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar'],
        4: ['K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'],
        5: ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
            'In', 'Sn', 'Sb', 'Te', 'I', 'Xe'],
        6: ['Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
            'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
            'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'],
        7: ['Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
            'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
            'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']}

    # Create atomic numbers dictionary
    atomic_numbers = {}
    z = 1
    for period in sorted(period_elements.keys()):
        for element in period_elements[period]:
            atomic_numbers[element] = z
            z += 1

    def get_sort_key(col):
        if '⁺' not in col:
            return (float('inf'), float('inf'))
        oxidation_state = get_oxidation_state(col)
        metal = get_metal_symbol(col)
        atomic_num = atomic_numbers.get(metal, float('inf'))
        return (oxidation_state, atomic_num)
    
    # Sort columns
    metal_cols = [col for col in metal_df.columns if '⁺' in col]
    sorted_cols = sorted(metal_cols, key=get_sort_key)
    other_cols = [col for col in metal_df.columns if col not in sorted_cols]
    sorted_cols.extend(other_cols)
    metal_df = metal_df[sorted_cols]
    
    return metal_df

def hill_eq(concentration, ymin, ymax, K, n):
    return ymin + (ymax - ymin) * (concentration**n) / (K**n + concentration**n)

def hcl_effect(tm_df, metals, concentrations, exclude):
    concentrations = concentrations[exclude:]
    for metal, wells in metals.items():
        if metal not in ["WT", "EDTA", "HCl", "Blank"]:
            metals[metal] = wells[exclude:]
    return metals, concentrations

def plot_tm_scatter(metal_df, exclude, title, output_dir):
    # Get WT average and EDTA values
    wt_avg_tm = metal_df.loc['WT'].mean(skipna=True)
    edta_avg_tm = metal_df.loc['EDTA'].mean(skipna=True)
    wt_edta = metal_df.iloc[-3:]
    metal_df = metal_df.iloc[:-3]
    compare_tm = edta_avg_tm
    
    metal_df = pd.concat([metal_df, wt_edta], axis=0)
    print(metal_df)

    # Initialize parameter rows
    metal_df.loc['Kd'] = np.nan
    metal_df.loc['Delta_Tm'] = np.nan
    metal_df.loc['R2'] = np.nan

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    ax = fig.add_subplot(gs[0])
    ax_text = fig.add_subplot(gs[1])
    
    kd_summary = []
    concentrations = metal_df.index[:-6].astype(float) 
    concentrations = concentrations[exclude:]

    for metal in metal_df.columns:
        if metal == "Acid":
            continue
        tm_values = metal_df[metal].iloc[:-6] 
        tm_values = tm_values[exclude:]
        mask = (tm_values != 0) & tm_values.notna()
        filtered_concentrations = concentrations[mask]
        filtered_tm_values = tm_values[mask]
        print(metal)
        # Determine if stabilizing or destabilizing
        try:
            if (filtered_tm_values.iloc[-1] - compare_tm) < 0: # destabilizing
                p0 = [compare_tm, max(filtered_tm_values), np.median(filtered_concentrations), 0]
                bounds = ([compare_tm - 5, 0, 0, -1], [compare_tm + 5, 125, np.inf, 1])
            else: # stabilizing
                p0 = [min(filtered_tm_values), compare_tm, np.median(filtered_concentrations), 0]
                bounds = ([0, compare_tm - 5, 0, -1], [125, compare_tm + 5, np.inf, 1])
            
            # Fit Hill equation
            popt, _ = curve_fit(lambda x, ymin, ymax, K, n: hill_eq(x, ymin, ymax, K, n),
                                filtered_concentrations,
                                filtered_tm_values,
                                p0=p0,
                                bounds=bounds)
            ymin, ymax, K, n = popt
            
            # Calculate parameters
            if (filtered_tm_values.iloc[-1] - compare_tm) < 0:
                delta_tm = ymax - compare_tm
            else: 
                delta_tm = ymin - compare_tm
            
            # Generate fit curve and calculate R²
            fit_x = np.logspace(np.log10(min(filtered_concentrations)), np.log10(max(filtered_concentrations)), 100)
            fit_y = hill_eq(fit_x, ymin, ymax, K, n)
            predicted_y = hill_eq(filtered_concentrations, ymin, ymax, K, n)
            r_squared = 1 - (np.sum((filtered_tm_values - predicted_y) ** 2) / 
                            np.sum((filtered_tm_values - filtered_tm_values.mean()) ** 2))
            
            # Plot data and fit with metal-specific colors
            color = metal_df.attrs['colors'].get(metal, '#333333')
            ax.scatter(filtered_concentrations, filtered_tm_values, 
                      label=metal, color=color)
            if r_squared > 0.5:
                ax.plot(fit_x, fit_y, color=color, alpha=0.5)
            
            # Store parameters in DataFrame
            metal_df.loc['Kd', metal] = K
            metal_df.loc['Delta_Tm', metal] = delta_tm
            metal_df.loc['R2', metal] = r_squared
            
            # Add to summary
            kd_summary.append(f"{metal}: Kd={K:.2f}µM, ΔTm={delta_tm:.2f}°C, R²={r_squared:.2f}")
        except RuntimeError:
            print(f"Could not fit Hill equation for {metal}")
            kd_summary.append(f"{metal}: N.B.")
            color = metal_df.attrs['colors'].get(metal, '#333333')
            ax.scatter(filtered_concentrations, filtered_tm_values, 
                      label=metal, color=color)

    # Add WT and EDTA lines
    ax.axhline(wt_avg_tm, color='red', linestyle='-', label="WT")
    ax.axhline(edta_avg_tm, color='black', linestyle='-', label="EDTA")
    
    # Customize plot
    ax.set_xscale('log')
    ax.set_xlabel("Concentration (µM)")
    ax.set_ylabel("Tm (°C)")
    ax.set_title("DSF Hill equation fit for "+title)
    
    # Add summary text in its own subplot
    summary_text = "\n".join(kd_summary)
    ax_text.text(0.5, 0.5, summary_text, 
                ha='center', va='center', 
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="black"))
    ax_text.axis('off')  # Hide axes of text subplot
    
    # Adjust layout
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir+"/"+title.replace(' ', '_')+"_hill_fit.png"), dpi=300, bbox_inches='tight')
    return metal_df

def plot_tm_bar(metal_df, title, output_dir):
    # Filter out control columns and get values
    metals = [col for col in metal_df.columns if col not in ["WT", "EDTA", "Acid", "Blank"]]
    Kd_values = metal_df.loc['Kd', metals].values.astype(float)
    delta_tm = metal_df.loc['Delta_Tm', metals].values.astype(float)
    r2_values = metal_df.loc['R2', metals].values.astype(float)
    
    # Set high Kd for poor fits
    poor_fits = (r2_values < 0.5) | (np.abs(delta_tm) < 3)
    Kd_values[poor_fits] = 10000
    
    # Calculate inverse Kd and create color mapping
    inv_Kd = [1/kd for kd in Kd_values]
    norm = Normalize(vmin=-10, vmax=10)
    colors = [plt.cm.coolwarm(norm(tm)) for tm in delta_tm]

    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(metals, inv_Kd, color=colors, edgecolor='black', width=0.8)
    
    # Customize plot
    ax.set_ylabel('Binding Affinity (1/Kd)', fontsize=12)
    ax.set_yscale('log')
    ax.set_xlabel('Metal Ion', fontsize=12)
    ax.set_title('DSF Kd and Tm shift for '+title, pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add ΔTm color legend
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('ΔTm (°C)', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir+"/"+title.replace(' ', '_')+"_kd_tm_bar.png", dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()