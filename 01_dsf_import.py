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
    parser.add_argument('-d', '--directory', type=str, required=True, help="Path to the directory containing the data files")
    parser.add_argument('-m', '--metals', type=str, required=True, help="Space-separated list of metal symbols (e.g., 'K+ Ca2+ La3+')")
    args = parser.parse_args()

    raw_file, derivative_file = find_files(args.directory)
    raw_data = load_data(raw_file)
    derivative_data = load_data(derivative_file)
    metal_df = metals_and_concentrations(args.metals)
    metal_df = plot_data(raw_data, derivative_data, metal_df, args.directory)
    metal_df.to_csv(args.directory + "/tm_values.csv", index=True)

def metals_and_concentrations(metal_symbols):
    superscript_map = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
    metal_list = [metal.translate(superscript_map) for metal in metal_symbols.split()]
    concentrations = [1000, 500, 250, 125, 62.5, 12.5, 6.25, 3.25, 1.625, 0.8125, 0.40625]
    index = [str(c) for c in concentrations] + ['WT', 'EDTA']
    metal_df = pd.DataFrame(index=index, columns=metal_list)
    for i, metal in enumerate(metal_list):
        row = chr(65 + i)
        for j, conc in enumerate(concentrations):
            metal_df.loc[str(conc), metal] = f"{row}{j+1}"

    metal_df.loc['WT', metal_list[0:3]] = ['A12', 'B12', 'C12']
    metal_df.loc['EDTA', metal_list[0:3]] = ['D12', 'E12', 'F12']
    
    print(metal_df)
    return metal_df

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

def plot_data(raw_data, derivative_data, metal_df, directory):
    fluorescence_columns = [f"{chr(i)}{j}" for i in range(65, 73) for j in range(1, 13)]
    tm_dict = {"Well": [], "Tm": []}
    fig, axes = plt.subplots(8, 12, figsize=(12, 8))
    axes = axes.flatten()

    for i, col in enumerate(fluorescence_columns):
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
        id = chr(65 + row_label)+str(col_label+1)
        row_idx, col_idx = np.where(metal_df == id)

        if col_label == 0:
            ax.set_ylabel(f"{chr(65 + row_label)}", rotation=0, labelpad=10)
        if row_label == 7:
            ax.set_xlabel(f"{col_label + 1}", labelpad=10)
        
        if len(row_idx) and len(col_idx) > 0:
            try:
                text_content = f'{metal_df.columns[col_idx[0]]}\n{float(metal_df.index[row_idx[0]]):.1f}µM'
            except ValueError:
                text_content = f'{metal_df.index[row_idx[0]]}'
            metal_df.iloc[row_idx[0], col_idx[0]] = tms
        else:
            text_content = 'Blank'
        if tms == "Blank":
            text_content += '\nBlank'
        elif tms is not None:
            text_content += f'\n{tms:.0f}'
            
        ax.text(0.95, 0.95, text_content, transform=ax.transAxes, ha='right', va='top', fontsize=8, color='black')
        
    plt.tight_layout()
    plt.savefig(directory+"/raw_and_derivate.png", dpi=300)
    print(metal_df)
    return metal_df

if __name__ == '__main__':
    main()