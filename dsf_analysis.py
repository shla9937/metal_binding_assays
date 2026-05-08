#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42  # Embed fonts as vectors
matplotlib.rcParams['pdf.use14corefonts'] = False
matplotlib.rcParams['savefig.transparent'] = True
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['font.size'] = 8        # default for all text
matplotlib.rcParams['axes.titlesize'] = 8
matplotlib.rcParams['axes.labelsize'] = 8
matplotlib.rcParams['xtick.labelsize'] = 7
matplotlib.rcParams['ytick.labelsize'] = 7
matplotlib.rcParams['legend.fontsize'] = 7
matplotlib.rcParams['figure.titlesize'] = 8
import matplotlib.pyplot as plt
import argparse
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import Normalize
import matplotlib.colors


def main():
    parser = argparse.ArgumentParser(description="Analyze 384 well DSF up to 32 metals.")
    parser.add_argument('-c','--csv',type=str,nargs='+',required=True,help="Raw DSF csvs from DA2 (one or more files; multiple files are averaged together)")
    parser.add_argument('-p','--protein',type=str,required=True,help="Name of protein")
    parser.add_argument('-ht','--high_temp',type=float,required=False,help="Exclude temps over this value")
    parser.add_argument('-lt','--low_temp',type=float,required=False,help="Exclude temps under this value")
    parser.add_argument('-o','--override',type=float,required=False,help="Override analysis temperature")
    parser.add_argument('-m','--model',type=str,default='hill',choices=['hill','two-site','quadratic'],help="Binding model: 'hill' , 'two-site' , or 'quadratic'")
    parser.add_argument('-w','--exclude_wells',type=str,nargs='+',default=[],help="Well positions to exclude from analysis (e.g. A1 B3 C12)")
    parser.add_argument('-ms','--metal_set',type=int,default=29,choices=[6, 29],help="Metal set being used in plate, either 29x1 or 6x4.")
    parser.add_argument('-s', '--show_all',action='store_true',help="Show all figures interactively, including legend figures. By default only the first raw fluorescence figure and bar chart are shown.")
    args = parser.parse_args()
    
    metal_setup(args.metal_set)
    raw_dfs, avg_tm_df = load_data(args)
    titration_df, kd_results = analyze(avg_tm_df, args)
    plot_all(raw_dfs, avg_tm_df, titration_df, kd_results, args, show_all=args.show_all)
    save_results_csv(titration_df, kd_results, args.protein)
    save_command_txt(args.protein)

def load_data(args):
    """Load, filter, smooth, normalize, average, and find Tms."""
    raw_dfs = []
    norm_dfs = []
    for i, csv_file in enumerate(args.csv):
        df = parse_csv_file(csv_file)
        if args.high_temp:
            df = exclude_high_temps(df, args.high_temp)
        if args.low_temp:
            df = exclude_low_temps(df, args.low_temp)
        df = assign_conc(df)
        if args.exclude_wells:
            df = exclude_wells(df, args.exclude_wells)
        raw_dfs.append(df)
        df_rep = df.copy()
        df_rep['Well'] = df_rep['Well'].astype(str) + f'_rep{i}'
        norm_dfs.append(normalize(smooth_wells(df_rep)))
    avg_tm_df = find_tms(average(pd.concat(norm_dfs, ignore_index=True)))
    return raw_dfs, avg_tm_df

def analyze(avg_tm_df, args):
    return find_kds(avg_tm_df,args.override,args.model,tm_threshold,r2_threshold,signal_threshold)

def plot_all(raw_dfs, avg_tm_df, titration_df, kd_results, args, show_all=False):
    snr_per_plate = [calc_snr(df) for df in raw_dfs]
    snr_df = pd.concat(snr_per_plate).groupby('Well Position')['SNR'].mean().reset_index()

    def apo_snr(raw_df, snr_plate_df):
        apo_wells = raw_df[raw_df['Metal'] == 'Apo']['Well Position'].unique()
        return snr_plate_df[snr_plate_df['Well Position'].isin(apo_wells)]['SNR'].dropna()

    all_apo_snr = pd.concat([apo_snr(raw_dfs[i], snr_per_plate[i]) for i in range(len(raw_dfs))])
    global_snr_str = f"apo SNR median={all_apo_snr.median():.1f}"
    raw_ylim = (pd.concat(raw_dfs)['Fluorescence'].min(), pd.concat(raw_dfs)['Fluorescence'].max())
    for i, raw_df in enumerate(raw_dfs):
        plate_apo_snr = apo_snr(raw_df, snr_per_plate[i])
        plate_snr_str = f"apo SNR median={plate_apo_snr.median():.1f}"
        rep_label = f"{args.protein} (rep {i+1})" if len(raw_dfs) > 1 else args.protein
        show_this = (i == 0) or show_all  # always show first raw figure
        plot_df(raw_df, 'Fluorescence', rep_label, snr_df=snr_df, snr_title=plate_snr_str, ylim=raw_ylim, show=show_this)
    plot_df(avg_tm_df, 'Smoothed Fluorescence', args.protein, error_column='Standard Error', override=args.override, snr_df=snr_df, snr_title=global_snr_str, kd_results=kd_results, show=True)
    plot_tms(avg_tm_df, args.protein, show=show_all)
    plot_kds(titration_df, kd_results, args.protein, model=args.model, show=show_all)
    plot_kd_bars(titration_df, kd_results, args.protein, model=args.model, show=True)

def metal_setup(metal_set):
    global metals_left, metals_right, concentrations, protein_conc, tm_threshold, r2_threshold, signal_threshold
    # which metal is being titrated in columns 1-12, A-P (left) or columns 13-24, A-P (right))
    if metal_set == 29:
        metals_left = ["Li⁺", "Cu²⁺", "Mg²⁺", "Zn²⁺", "K⁺", "Rb⁺", "Ca²⁺", "Sr²⁺",
                    "Sc³⁺", "Y³⁺", "Mn²⁺", "Cs⁺", "Co²⁺", "Ba²⁺", "Ni²⁺", "La³⁺"]
        metals_right = ["Ce³⁺", "Ho³⁺", "Pr³⁺", "Er³⁺", "Nd³⁺", "Tm³⁺", "Sm³⁺", "Yb³⁺",
                    "Eu³⁺", "Lu³⁺", "Gd³⁺", "EDTA", "Tb³⁺", "EDTA", "Dy³⁺", "Apo"]
    elif metal_set == 6:
        metals_left = ["Mn²⁺", "Mn²⁺", "Co²⁺", "Co²⁺", "Ni²⁺", "Ni²⁺", "Cu²⁺", "Cu²⁺",
                   "Nd³⁺", "Nd³⁺", "Dy³⁺", "Dy³⁺", "EDTA", "EDTA", "Apo", "Apo"]
        metals_right = ["Mn²⁺", "Mn²⁺", "Co²⁺", "Co²⁺", "Ni²⁺", "Ni²⁺", "Cu²⁺", "Cu²⁺",
                   "Nd³⁺", "Nd³⁺", "Dy³⁺", "Dy³⁺", "EDTA", "EDTA", "Apo", "Apo"]
    
    # metal concentrations in µM
    concentrations = [100, 50.0, 25.0, 12.5, 6.25, 3.13, 1.56, 0.781, 0.391, 0.195, 0.0977, 0.0488] 
    protein_conc = 5  # µM      
    tm_threshold = 2.0
    r2_threshold = 0.7
    signal_threshold = 0.15
    return True

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

def exclude_high_conc(df, exclude_high):
    unique_concs = sorted([c for c in df['Concentration'].unique() if c > 0], reverse=True)
    if exclude_high >= len(unique_concs):
        concs_to_exclude = unique_concs[:-1]
    else:
        concs_to_exclude = unique_concs[:exclude_high]
    
    df_filtered = df[~df['Concentration'].isin(concs_to_exclude)]
    return df_filtered

def exclude_wells(df, wells):
    wells_normalised = [w.strip().upper() for w in wells]
    return df[~df['Well Position'].str.upper().isin(wells_normalised)]

def exclude_low_conc(df, exclude_low):
    unique_concs = sorted([c for c in df['Concentration'].unique() if c > 0], reverse=True)
    if exclude_low >= len(unique_concs):
        concs_to_exclude = unique_concs[:-1]
    else:
        concs_to_exclude = unique_concs[-exclude_low:]
    df_filtered = df[~df['Concentration'].isin(concs_to_exclude)] 
    return df_filtered

def assign_conc(df):
    global rows, metal_colors
    df['Metal'] = None
    df['Concentration'] = np.nan
    rows = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P"]
    metal_colors = {
        # Colored aqueous chloride solutions
        "Mn²⁺": "#FF1493",    
        "Co²⁺": "#8B008B",    
        "Ni²⁺": "#00C853",    
        "Cu²⁺": "#1E90FF",    
        "Nd³⁺": "#8A2BE2",    
        "Dy³⁺": "#FFD700",   
        "Pr³⁺": "#80FF20",   
        "Er³⁺": "#FF4D80",    
        "Ho³⁺": "#FF9D00",    
        # Non-metal controls
        "EDTA": "#FF6600",    
        "Apo":  "#808080",    
        # Colorless aqueous solutions — unique gray shades
        # Alkali metals
        "Li⁺":  "#DCE8F0",    
        "K⁺":   "#E8E8D8",    
        "Rb⁺":  "#9090A8",    
        "Cs⁺":  "#707060",   
        # Alkaline earth metals
        "Mg²⁺": "#D8D8D0",   
        "Ca²⁺": "#C8D0D8",    
        "Sr²⁺": "#A8B0A8",   
        "Ba²⁺": "#585858",   
        # Group 3 / early transition
        "Sc³⁺": "#989898",   
        "Y³⁺":  "#686868",    
        "Zn²⁺": "#C0C0C0",   
        "La³⁺": "#484858",    
        # Colorless lanthanides
        "Ce³⁺": "#A8C0B0",    
        "Sm³⁺": "#C8C0B0",   
        "Eu³⁺": "#C0B0C0",    
        "Gd³⁺": "#686060",   
        "Tb³⁺": "#606868",    
        "Tm³⁺": "#A0A898",   
        "Yb³⁺": "#787070",    
        "Lu³⁺": "#404040"}

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

def get_atomic_number(metal_name):
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
        'Pa': 91, 'U': 92}
    element = ''.join(c for c in metal_name if c.isalpha())
    return atomic_numbers.get(element, 999)

def plot_df(df, y_column, protein_name, error_column=None, override=None, snr_df=None, snr_title=None, kd_results=None, ylim=None, show=True):    
    # If error_column provided, use averaged data mode
    if error_column:
        unique_metals = sorted(df['Metal'].unique(), key=get_atomic_number)
        n_metals = len(unique_metals)
        n_cols = 5
        n_rows = (n_metals + n_cols - 1) // n_cols
        fig_h = max(2.0, min(6.0, n_rows * 1.4))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.9, fig_h))
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
            
            ax.axvline(x=plot_temp, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
            ax.tick_params(labelsize=5)
            # Add binding SNR to subplot title if kd_results provided
            if kd_results is not None:
                temp_key = 'Override' if override else 'Apo'
                kd_row = kd_results[(kd_results['Metal'] == metal) & (kd_results['Temperature'] == temp_key)]
                if not kd_row.empty and 'Binding_SNR' in kd_row.columns:
                    bsnr = kd_row['Binding_SNR'].values[0]
                    bsnr_str = f'{bsnr:.1f}' if not np.isnan(bsnr) else 'N/A'
                    ax.set_title(f"{metal}  (SNR={bsnr_str})", fontsize=7)
                else:
                    ax.set_title(f"{metal}", fontsize=7)
            else:
                ax.set_title(f"{metal}", fontsize=7)
            ax.grid(True, alpha=0.3)

        for idx in range(n_metals, len(axes)):
            axes[idx].axis('off')
        fig.supxlabel('Temperature (°C)', fontsize=7)
        fig.supylabel(y_column, fontsize=7)
        suptitle = f'{protein_name}  |  {snr_title}' if snr_title else protein_name
        fig.suptitle(suptitle, fontsize=8, y=0.998)
    else:
        # Build panel list: one per (row, half-plate), sorted by metal atomic number, Apo/EDTA last
        special_metals = {'Apo', 'EDTA'}
        panel_list = []
        for row in rows:
            for well_range, metal_list in [(range(1, 13), metals_left), (range(13, 25), metals_right)]:
                metal = metal_list[rows.index(row)]
                panel_list.append((metal, row, well_range))
        panel_list.sort(key=lambda x: (x[0] in special_metals, get_atomic_number(x[0]), x[1], x[2].start))

        n_cols = 5
        n_panels = len(panel_list)
        n_rows_grid = (n_panels + n_cols - 1) // n_cols
        fig_h = max(2.0, min(6.0, n_rows_grid * 1.4))
        fig, axes = plt.subplots(n_rows_grid, n_cols, figsize=(6.9, fig_h))
        axes = axes.flatten()

        for i in range(n_panels, len(axes)):
            axes[i].axis('off')

        for plot_idx, (metal, row, well_range) in enumerate(panel_list):
            ax = axes[plot_idx]
            well_positions = [row + str(w) for w in well_range]
            well_data_subset = df[df['Well Position'].isin(well_positions)]
            concentrations_in_panel = sorted(well_data_subset['Concentration'].unique())

            for well in well_range:
                well_pos = row + str(well)
                well_data = df[df['Well Position'] == well_pos].sort_values('Temperature')
                if len(well_data) == 0:
                    continue
                conc = well_data['Concentration'].iloc[0]
                conc_idx = concentrations_in_panel.index(conc) if conc in concentrations_in_panel else 0
                color = matplotlib.colors.to_rgba(metal_colors[metal])
                color = tuple(c * (1 - conc_idx / max(len(concentrations_in_panel) - 1, 1)) for c in color[:3]) + (color[3],)
                ax.plot(well_data['Temperature'], well_data[y_column], alpha=0.8, color=color)

            ax.tick_params(labelsize=5)
            ax.set_title(f"{metal}", fontsize=6)
            ax.grid(True, alpha=0.3)

        fig.supxlabel('Temperature (°C)', fontsize=7)
        fig.supylabel(y_column, fontsize=7)
        suptitle = f'{protein_name}  |  {snr_title}' if snr_title else protein_name
        fig.suptitle(suptitle, fontsize=8, y=0.998)
    
    plt.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.4)
    plt.subplots_adjust(left=0.12)
    file_stem = protein_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    plt.savefig(f"{file_stem}_{y_column.lower().replace(' ', '_')}_melt_curves.pdf", bbox_inches='tight', backend='pdf')
    if show:
        plt.show()
    else:
        plt.close()

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
    df['Smoothed Derivative'] = -df.groupby('Well')['Derivative'].transform(_smooth) #flip derv sign
    return df

def normalize(df):
    df = df.copy()
    scaler = MinMaxScaler()
    df['Normalized Fluorescence'] = df.groupby('Well')['Smoothed Fluorescence'].transform(
        lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten())
    df['Normalized Derivative'] = df.groupby('Well')['Smoothed Derivative'].transform(
        lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten())
    return df

def average(df):
    df_copy = df.copy()
    df_copy['Temperature'] = df_copy['Temperature'].round(1)
    grp = df_copy.groupby(['Metal', 'Concentration', 'Temperature'])
    fluor_df = grp['Normalized Fluorescence'].apply(list).reset_index()
    fluor_df['Average Normalized Fluorescence'] = fluor_df['Normalized Fluorescence'].apply(np.mean)
    fluor_df['Standard Error'] = fluor_df['Normalized Fluorescence'].apply(lambda x: np.std(x) / np.sqrt(len(x)))
    deriv_df = grp['Normalized Derivative'].apply(list).reset_index()
    deriv_df['Average Normalized Derivative'] = deriv_df['Normalized Derivative'].apply(np.mean)
    avg_df = fluor_df.merge(deriv_df[['Metal', 'Concentration', 'Temperature', 'Average Normalized Derivative']], on=['Metal', 'Concentration', 'Temperature'])
    return avg_df

def find_tms(df):
    tm_values = []
    smoothed_values = []
    group_indices = []

    for (metal, conc), group in df.groupby(['Metal', 'Concentration']):
        group = group.sort_values('Temperature')
        temps = group['Temperature'].values
        fluor = group['Average Normalized Fluorescence'].values
        derivative = group['Average Normalized Derivative'].values
        peak_idx = np.argmax(derivative)
        tm = temps[peak_idx]
        tm_values.append(tm)
        smoothed_values.append(fluor)
        group_indices.append(group.index)

    df['Tm'] = np.nan
    df['Smoothed Fluorescence'] = np.nan
    for tm, smoothed, indices in zip(tm_values, smoothed_values, group_indices):
        df.loc[indices, 'Tm'] = tm
        df.loc[indices, 'Smoothed Fluorescence'] = smoothed

    return df

def plot_tms(df, protein_name, show=True):
    fig, ax = plt.subplots(figsize=(3.3, 3.3))
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

    ax.set_xlabel('Concentration (µM)', fontsize=10)
    ax.set_ylabel('Tm (°C)', fontsize=10)
    ax.set_title(f'{protein_name} - Tm vs Concentration', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    protein_lower = protein_name.lower()
    plt.savefig(f'{protein_lower}_tm_vs_concentration.pdf', bbox_inches='tight', backend='pdf')
    if show:
        plt.show()
    else:
        plt.close()

    handles, labels = ax.get_legend_handles_labels()
    fig_leg, ax_leg = plt.subplots(figsize=(3.5, 4.5))
    ax_leg.axis('off')
    ax_leg.legend(handles, labels, fontsize=6, loc='center', ncol=2, frameon=True)
    fig_leg.tight_layout()
    fig_leg.savefig(f'{protein_lower}_tm_vs_concentration_legend.pdf', bbox_inches='tight', backend='pdf')
    if show:
        plt.show()
    else:
        plt.close()

def binding_curve_hill(conc, kd, ymin, ymax, n):
    return ymin + (ymax - ymin) * conc**n / (kd**n + conc**n)

def binding_curve_two_site(conc, kd1, kd2, ymin, ymax):
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
    try:
        # Initial parameter guesses
        ymin_guess = np.min(values)
        ymax_guess = np.max(values)
        kd_guess = np.median(concentrations)
        errors = np.where(errors == 0, 1e-10, errors) # Replace zero errors with small value to avoid division by zero
        
        if model == 'hill':
            n_guess = 1.0  # Hill coefficient
            popt, pcov = curve_fit(
                binding_curve_hill, 
                concentrations, 
                values,
                p0=[kd_guess, ymin_guess, ymax_guess, n_guess],
                sigma=errors,
                absolute_sigma=True,
                maxfev=10000,
                bounds=([0, 0, 0, 0.1], [np.inf, 1, 1, 5]))
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
                bounds=([0, 0, 0, 0], [np.inf, np.inf, 1, 1]))
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
                bounds=([0, 0, 0], [np.inf, 1, 1]))
            kd, ymin, ymax = popt
            perr = np.sqrt(np.diag(pcov))
            kd_err = perr[0] * 1.96
            
            # Calculate R²
            y_pred = binding_curve_quadratic(concentrations, *popt)
            ss_res = np.sum((values - y_pred)**2)
            ss_tot = np.sum((values - np.mean(values))**2)
            r_squared = 1 - (ss_res / ss_tot)
            return {'Kd': kd, 'Kd_Error': kd_err, 'R_squared': r_squared, 'Fit_Params': popt, 'Model': 'quadratic'}
    except:
        if model == 'hill':
            return {'Kd': np.nan, 'Kd_Error': np.nan, 'Hill_n': np.nan, 'Hill_n_Error': np.nan, 'R_squared': np.nan, 'Fit_Params': None, 'Model': 'hill'}
        elif model == 'quadratic':
            return {'Kd': np.nan, 'Kd_Error': np.nan, 'R_squared': np.nan, 'Fit_Params': None, 'Model': 'quadratic'}
        else:
            return {'Kd1': np.nan, 'Kd1_Error': np.nan, 'Kd2': np.nan, 'Kd2_Error': np.nan, 'R_squared': np.nan, 'Fit_Params': None, 'Model': 'two-site'}

def null_fit(model, r_squared):
    base = {'R_squared': r_squared, 'Fit_Params': None, 'Model': model}
    if model == 'hill':
        return {**base, 'Kd': np.nan, 'Kd_Error': np.nan, 'Hill_n': np.nan, 'Hill_n_Error': np.nan}
    elif model == 'quadratic':
        return {**base, 'Kd': np.nan, 'Kd_Error': np.nan}
    else:
        return {**base, 'Kd1': np.nan, 'Kd1_Error': np.nan, 'Kd2': np.nan, 'Kd2_Error': np.nan}

def find_kds(df, override, model, tm_threshold, r2_threshold, signal_threshold):
    apo_tm = df[df['Metal'] == 'Apo'].groupby('Concentration')['Tm'].first().mean()
    kd_data = []

    for (metal, conc), group in df.groupby(['Metal', 'Concentration']):
        group = group.sort_values('Temperature')
        temps = group['Temperature'].values
        smoothed_fluor = group['Smoothed Fluorescence'].values
        std_err = group['Standard Error'].values
        data_dict = {'Metal': metal,'Concentration': conc,'Apo Tm Temperature': apo_tm,'Apo Tm': np.interp(apo_tm, temps, smoothed_fluor),'Apo Tm Standard Error': np.interp(apo_tm, temps, std_err)}
        if override is not None:
            data_dict.update({'Override Temperature': override,'Override Tm': np.interp(override, temps, smoothed_fluor),'Override Tm Standard Error': np.interp(override, temps, std_err)})
        kd_data.append(data_dict)

    kd_df = pd.DataFrame(kd_data)
    kd_list = []

    for metal in kd_df['Metal'].unique():
        metal_data = kd_df[kd_df['Metal'] == metal].sort_values('Concentration')
        included_concs = metal_data['Concentration'].values
        metal_tms = df[(df['Metal'] == metal) & (df['Concentration'].isin(included_concs))].groupby('Concentration')['Tm'].first()
        tm_change = metal_tms.max() - metal_tms.min()
        concs = metal_data['Concentration'].values

        configs = [('Apo Tm', 'Apo')]
        if override is not None:
            configs.append(('Override Tm', 'Override'))

        for val_col, temp_label in configs:
            vals = 1 - metal_data[val_col].values
            errs = metal_data[f'{val_col} Standard Error'].values
            signal_range = np.max(vals) - np.min(vals)
            mean_err = np.mean(errs)
            binding_snr = signal_range / mean_err if mean_err > 0 else np.nan
            fit_result = fit_binding_curve(concs, vals, errs, model=model)
            # Quality control: flat signal and Tm change always apply;
            # R² threshold only applies when the fit converged
            fails_qc = (signal_range < signal_threshold
                        or tm_change <= tm_threshold
                        or (not np.isnan(fit_result['R_squared'])
                            and fit_result['R_squared'] < r2_threshold))
            if fails_qc:
                fit_result = null_fit(model, fit_result['R_squared'])
            fit_result['Metal'] = metal
            fit_result['Temperature'] = temp_label
            fit_result['Binding_SNR'] = binding_snr
            kd_list.append(fit_result)

    return kd_df, pd.DataFrame(kd_list)

def fmt_kd(kd):
    """Return a human-readable Kd string (nM or µM)."""
    return f"{kd*1000:.0f}nM" if kd < 1.0 else f"{kd:.1f}µM"

def kd_label(metal, kd_row, model):
    r2 = kd_row['R_squared'].values[0]
    if model == 'two-site':
        kd1, kd2 = kd_row['Kd1'].values[0], kd_row['Kd2'].values[0]
        if np.isnan(kd1):
            return f"{metal}: Kd=N/A"
        return f"{metal}: Kd1={fmt_kd(kd1)}, Kd2={fmt_kd(kd2)}, R²={r2:.3f}"
    else:
        kd = kd_row['Kd'].values[0]
        if np.isnan(kd):
            return f"{metal}: Kd=N/A"
        suffix = ' (quad)' if model == 'quadratic' else f", n={kd_row['Hill_n'].values[0]:.2f}"
        return f"{metal}: Kd={fmt_kd(kd)}{suffix}, R²={r2:.3f}"

def eval_fit(model, conc_fit, popt):
    fn = {'hill': binding_curve_hill, 'quadratic': binding_curve_quadratic,
          'two-site': binding_curve_two_site}[model]
    return fn(conc_fit, *popt)

def plot_kds(df, kd_results, protein_name, model='hill', show=True):
    has_override = 'Override Temperature' in df.columns
    plot_configs = [('Apo Tm', 'Apo Tm Temperature', 'Apo')]
    if has_override:
        plot_configs.append(('Override Tm', 'Override Temperature', 'Override'))

    conc_min = df[df['Concentration'] > 0]['Concentration'].min()
    conc_max = df['Concentration'].max()
    conc_fit = np.logspace(np.log10(conc_min), np.log10(conc_max), 200)
    protein_lower = protein_name.lower()

    for data_col, temp_col, kd_key in plot_configs:
        fig, ax = plt.subplots(figsize=(3.3, 3.3))

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
                label = kd_label(metal, kd_row, model)
            else:
                popt = None
                label = f"{metal}: Kd=N/A"

            ax.errorbar(concentrations, values, yerr=errors, color=color, marker='o', linestyle='', capsize=3, alpha=0.7, markersize=6)
            if popt is not None:
                ax.plot(conc_fit, eval_fit(model, conc_fit, popt), color=color,
                        linestyle='-', linewidth=2, alpha=0.8, label=label)
            else:
                ax.plot([], [], color=color, label=label)

        ax.set_xlabel('Concentration (µM)', fontsize=10)
        ax.set_ylabel('% Folded', fontsize=10)
        title_name = 'Apo Tm' if kd_key == 'Apo' else 'Override Temp'
        ax.set_title(f'{protein_name} - Titration at {title_name} ({temp_value:.1f}°C)', fontsize=10)
        ax.set_xscale('log')
        ax.set_ylim(0, 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        suffix = '_override' if kd_key == 'Override' else ''
        plt.savefig(f'{protein_lower}_metal_titrations{suffix}.pdf', bbox_inches='tight', backend='pdf')
        if show:
            plt.show()
        else:
            plt.close()

        handles, labels = ax.get_legend_handles_labels()
        fig_leg, ax_leg = plt.subplots(figsize=(3.5, 4.5))
        ax_leg.axis('off')
        ax_leg.legend(handles, labels, fontsize=6, loc='center', ncol=2, frameon=True)
        fig_leg.tight_layout()
        fig_leg.savefig(f'{protein_lower}_metal_titrations{suffix}_legend.pdf', bbox_inches='tight', backend='pdf')
        if show:
            plt.show()
        else:
            plt.close()

def plot_kd_bars(df, kd_results, protein_name, model, show=True):
    has_override = 'Override Temperature' in df.columns
    if has_override:
        temp_value = df['Override Temperature'].iloc[0]
        temp_key = 'Override'
    else:
        temp_value = df['Apo Tm Temperature'].iloc[0]
        temp_key = 'Apo'
    
    kd_results_filtered = kd_results[kd_results['Temperature'] == temp_key].copy()
    all_metals = [m for m in kd_results_filtered['Metal'].unique() if m not in ['EDTA', 'Apo']]
    all_metals = sorted(all_metals, key=get_atomic_number)
    n_metals = len(all_metals)
    bar_w_inch = 6.9 / 32 
    fig_w = max(1.5, min(6.9, n_metals * bar_w_inch))
    fig, ax = plt.subplots(figsize=(fig_w, 2.5))
    x_positions = np.arange(len(all_metals))
    bar_width = 0.6
    
    if model == 'two-site':
        bar_width = 0.4
    
    # Auto-scale ylim from actual Kd values so whitespace is minimised
    _inv_kds = []
    for _, _r in kd_results_filtered.iterrows():
        for _col in (['Kd'] if model in ('hill', 'quadratic') else ['Kd1', 'Kd2']):
            _k = _r.get(_col, np.nan)
            if not np.isnan(_k) and _k > 0:
                _inv_kds.append(1 / _k)
    if _inv_kds:
        ylim_top = 10 ** (np.ceil(np.log10(max(_inv_kds))) + 1)
        ylim_bot = 10 ** (np.floor(np.log10(min(_inv_kds))) - 1)
    else:
        ylim_top, ylim_bot = 1e4, 1e-5
    bar_bottom = ylim_bot * 0.5
    nb_label_y = ylim_bot * 1.5  # just inside the visible bottom for the NB text
    
    for metal_idx, metal in enumerate(all_metals):
        metal_data = kd_results_filtered[kd_results_filtered['Metal'] == metal]
        
        if metal_data.empty:
            continue
        
        base_color = metal_colors[metal]
        x_pos = x_positions[metal_idx]
        
        if model in ('hill', 'quadratic'):
            kd = metal_data['Kd'].values[0]
            kd_err = metal_data['Kd_Error'].values[0]
            
            if np.isnan(kd):
                ax.text(x_pos, nb_label_y, 'NB', ha='center', va='bottom', fontsize=6, fontweight='bold', color='black')
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
        
        else: 
            kd1 = metal_data['Kd1'].values[0]
            kd1_err = metal_data['Kd1_Error'].values[0]
            kd2 = metal_data['Kd2'].values[0]
            kd2_err = metal_data['Kd2_Error'].values[0]
            if np.isnan(kd1):
                ax.text(x_pos, nb_label_y, 'NB', ha='center', va='bottom', fontsize=6, fontweight='bold', color='black')
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
    ax.set_ylim(ylim_bot, ylim_top)
    ax.set_title(f'{protein_name} - DSF Binding Affinity at {temp_value:.1f}°C', pad=20)
    ax.set_xlim(-0.5, len(all_metals) - 0.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(all_metals, rotation=90, ha='right', fontsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set explicit ticks at clean Kd decade values in M, converted to inverse-Kd (µM⁻¹) plot units
    kd_decades_m = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    tick_positions = [1 / (kd_m * 1e6) for kd_m in kd_decades_m]  # convert M → µM, then invert
    ymin, ymax = ax.get_ylim()
    tick_positions = [t for t in tick_positions if ymin <= t <= ymax]
    kd_labels = [f'10$^{{{int(np.log10(kd))}}}$' for kd in kd_decades_m if ymin <= 1 / (kd * 1e6) <= ymax]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(kd_labels)
    
    if model == 'two-site':
        ax.legend(fontsize=8, loc='best')
    
    plt.tight_layout()
    plt.savefig(f'{protein_name.lower()}_kd_bar_chart.pdf', bbox_inches='tight', backend='pdf')
    if show:
        plt.show()
    else:
        plt.close()

def calc_snr(df):
    """Per-well SNR on raw fluorescence: (peak - baseline_mean) / baseline_std.
    Baseline = first 10% of temperature points (minimum 5)."""
    records = []
    for well_pos, group in df.groupby('Well Position'):
        group = group.sort_values('Temperature')
        fluor = group['Fluorescence'].values
        n_base = max(5, len(fluor) // 10)
        baseline = fluor[:n_base]
        baseline_std = np.std(baseline)
        snr = (np.max(fluor) - np.mean(baseline)) / baseline_std if baseline_std > 0 else np.nan
        records.append({'Well Position': well_pos, 'SNR': snr})
    return pd.DataFrame(records)

def save_command_txt(protein_name):
    protein_lower = protein_name.lower()
    command = ' '.join(sys.argv)
    with open(f'{protein_lower}_command.txt', 'w') as f:
        f.write(command + '\n')

def save_results_csv(titration_df, kd_results, protein_name):
    protein_lower = protein_name.lower()
    titration_df.to_csv(f'{protein_lower}_titration_data.csv', index=False)
    kd_results_export = kd_results.copy()
    kd_results_export['Metal'] = kd_results_export['Metal'].str.replace('²⁺', '2+').str.replace('³⁺', '3+')
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