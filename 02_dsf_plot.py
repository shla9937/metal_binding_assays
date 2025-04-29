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

    output_dir = os.path.join('..', 'outputs', args.title.replace(' ', '_'))
    os.makedirs(output_dir, exist_ok=True)
    metal_df = load_data(args.csvs)
    metal_df = plot_tm_scatter(metal_df, args.exclude, args.title)
    plot_tm_bar(metal_df, args.title)
    metal_df.to_csv(os.path.join(output_dir, f"{args.title}_tm_values.csv"), index=True)
    
def load_data(csv_files):
    # Load and concatenate files
    df_list = [pd.read_csv(csv_file, index_col=0) for csv_file in csv_files]
    metal_df = pd.concat(df_list, ignore_index=False, axis=1)
    
    # Complete periodic table with atomic numbers
    atomic_numbers = {
        'H': 1, 'He': 2,
        'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
        'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
        'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48,
        'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
        'Cs': 55, 'Ba': 56,
        'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63,
        'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
        'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
        'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86,
        'Fr': 87, 'Ra': 88,
        'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95,
        'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103,
        'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112,
        'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
    }
    
    def get_oxidation_state(col):
        """Extract oxidation state from superscript number."""
        superscript_map = {'¹': 1, '²': 2, '³': 3, '⁴': 4, '⁵': 5, '⁶': 6, '⁷': 7}
        for char in col:
            if char in superscript_map:
                return superscript_map[char]
        return 1 if '⁺' in col else 0
    
    def get_metal_symbol(col):
        """Extract metal symbol from column name."""
        return ''.join(c for c in col if c.isalpha())
    
    def get_sort_key(col):
        """Create sort key (oxidation_state, atomic_number)."""
        if '⁺' not in col:
            return (float('inf'), float('inf'))
        oxidation_state = get_oxidation_state(col)
        metal = get_metal_symbol(col)
        atomic_num = atomic_numbers.get(metal, float('inf'))
        return (oxidation_state, atomic_num)
    
    # Sort metal columns by oxidation state and atomic number
    metal_cols = [col for col in metal_df.columns if '⁺' in col]
    sorted_cols = sorted(metal_cols, key=get_sort_key)
    
    # Add remaining columns at the end
    other_cols = [col for col in metal_df.columns if col not in sorted_cols]
    sorted_cols.extend(other_cols)
    
    # Reorder DataFrame columns
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

def plot_tm_scatter(metal_df, exclude, title):
    wt_avg_tm = metal_df.loc['WT'].mean(skipna=True)
    edta_avg_tm = metal_df.loc['EDTA'].mean(skipna=True)
    metal_df.loc['Kd'] = np.nan
    metal_df.loc['Delta_Tm'] = np.nan
    metal_df.loc['R2'] = np.nan

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    ax = fig.add_subplot(gs[0])
    ax_text = fig.add_subplot(gs[1])
    
    kd_summary = []
    concentrations = metal_df.index[:-5].astype(float) 
    concentrations = concentrations[exclude:]

    for metal in metal_df.columns:
        tm_values = metal_df[metal].iloc[:-5] 
        tm_values = tm_values[exclude:]
        mask = (tm_values != 0) & tm_values.notna()
        filtered_concentrations = concentrations[mask]
        filtered_tm_values = tm_values[mask]
        print(metal)
        # Determine if stabilizing or destabilizing
        try:
            if (filtered_tm_values.iloc[-1] - wt_avg_tm) < 0: # destabilizing
                p0 = [wt_avg_tm, max(filtered_tm_values), np.median(filtered_concentrations), 0]
                bounds = ([wt_avg_tm - 5, 0, 0, -1], [wt_avg_tm + 5, 125, np.inf, 1])
            else: # stabilizing
                p0 = [min(filtered_tm_values), wt_avg_tm, np.median(filtered_concentrations), 0]
                bounds = ([0, wt_avg_tm - 5, 0, -1], [125, wt_avg_tm + 5, np.inf, 1])
            
            # Fit Hill equation
            popt, _ = curve_fit(lambda x, ymin, ymax, K, n: hill_eq(x, ymin, ymax, K, n),
                                filtered_concentrations,
                                filtered_tm_values,
                                p0=p0,
                                bounds=bounds)
            ymin, ymax, K, n = popt
            
            # Calculate parameters
            if (filtered_tm_values.iloc[-1] - wt_avg_tm) < 0:
                delta_tm = ymax - wt_avg_tm
            else: 
                delta_tm = ymin - wt_avg_tm
            
            # Generate fit curve and calculate R²
            fit_x = np.logspace(np.log10(min(filtered_concentrations)), np.log10(max(filtered_concentrations)), 100)
            fit_y = hill_eq(fit_x, ymin, ymax, K, n)
            predicted_y = hill_eq(filtered_concentrations, ymin, ymax, K, n)
            r_squared = 1 - (np.sum((filtered_tm_values - predicted_y) ** 2) / 
                            np.sum((filtered_tm_values - filtered_tm_values.mean()) ** 2))
            
            # Plot data and fit
            ax.scatter(filtered_concentrations, filtered_tm_values, label=metal)
            if r_squared > 0.5:
                ax.plot(fit_x, fit_y, color='gray', alpha=0.5)
            
            # Store parameters in DataFrame
            metal_df.loc['Kd', metal] = K
            metal_df.loc['Delta_Tm', metal] = delta_tm
            metal_df.loc['R2', metal] = r_squared
            
            # Add to summary
            kd_summary.append(f"{metal}: Kd={K:.2f}µM, ΔTm={delta_tm:.2f}°C, R²={r_squared:.2f}")
        except RuntimeError:
            print(f"Could not fit Hill equation for {metal}")
            kd_summary.append(f"{metal}: N.B.")

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
    plt.savefig("../outputs/"+str(title)+"/"+str(title)+"_hill_fit.png", dpi=300, bbox_inches='tight')
    return metal_df

def plot_tm_bar(metal_df, title):
    # Filter out control columns and get values
    metals = [col for col in metal_df.columns if col not in ["WT", "EDTA", "HCl", "Blank"]]
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
    
    # Add Kd labels to bars
    # for bar, kd in zip(bars, Kd_values):
    #     if kd < 10000:
    #         height = bar.get_height()
    #         if height < 0.02:
    #             new_height = 0.05
    #         elif height < 0.5:
    #             new_height = height + 0.05
    #         else:
    #             new_height = height - 0.4
    #         ax.text(bar.get_x() + bar.get_width()/2, new_height,
    #                f'{kd:.1f} μM', ha='center', va='bottom', 
    #                fontsize=10, rotation=90)

    # Add ΔTm color legend
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('ΔTm (°C)', fontsize=12)

    plt.tight_layout()
    plt.savefig("../outputs/"+str(title)+"/"+str(title)+"_kd_tm_bar.png", dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()