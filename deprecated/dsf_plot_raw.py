#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.signal import find_peaks

def main():
    baseline_dict = {
        ('A12', 'B12', 'C12', 'D12', 'E12', 'F12', 'G12', 'H12'): [
            'A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12',
            'B1','B2','B3','B4','B5','B6','B7','B8','B9','B10','B11','B12',
            'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12',
            'D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12',
            'E1','E2','E3','E4','E5','E6','E7','E8','E9','E10','E11','E12',
            'F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12',
            'G1','G2','G3','G4','G5','G6','G7','G8','G9','G10','G11','G12',
            'H1','H2','H3','H4','H5','H6','H7','H8','H9','H10','H11','H12']
            }

    parser = argparse.ArgumentParser(
        description="Analyze DSF Data, baseline correct multiple columns using specified baseline wells"
    )
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to the DSF data CSV file")
    args = parser.parse_args()

    data = load_data(args.input)

    fluorescence_columns = [f"{chr(i)}{j}" for i in range(65, 73) for j in range(1, 13)]

    baseline_corrected_data = data.copy()
    for baseline_wells, columns in baseline_dict.items():
        baseline_fluorescence = get_smoothed_baseline_fluorescence(data, baseline_wells)
        for col in columns:
            if col in baseline_corrected_data.columns:
                baseline_corrected_data[col] = data[col] - baseline_fluorescence

    fig, axes = plt.subplots(8, 12, figsize=(12, 8))
    axes = axes.flatten()

    all_fluorescence = []
    all_derivatives = []
    for col in fluorescence_columns:
        if col not in baseline_corrected_data.columns:
            print(f"Warning: Column {col} not found in data.")
            continue
        fluorescence_data = baseline_corrected_data[col]
        smoothed_fluorescence = smooth_data(fluorescence_data)
        derivative = compute_derivative(data['Temperature'], smoothed_fluorescence)
        
        valid_fluoro = smoothed_fluorescence[np.isfinite(smoothed_fluorescence)]
        valid_deriv = derivative[np.isfinite(derivative)]
        
        all_fluorescence.extend(valid_fluoro)
        all_derivatives.extend(valid_deriv)
    
    all_fluo = np.array(all_fluorescence)
    all_deriv = np.array(all_derivatives)
    
    y_min = np.percentile(all_fluo, 5) * 1.25
    y_max = np.percentile(all_fluo, 90) * 1.25

    derivative_y_min = np.percentile(all_deriv, 2.5) * 1.25
    derivative_y_max = np.percentile(all_deriv, 100) * 1.25

    for i, col in enumerate(fluorescence_columns):
        if col not in baseline_corrected_data.columns:
            print(f"Warning: Column {col} not found in data.")
            continue

        fluorescence_data = baseline_corrected_data[col]
        smoothed_fluorescence = smooth_data(fluorescence_data)
        derivative = compute_derivative(data['Temperature'], smoothed_fluorescence)
        tms = find_tm(data['Temperature'], derivative)  # Get all peaks
        
        ax = axes[i]
        ax2 = ax.twinx()

        ax.plot(data['Temperature'], smoothed_fluorescence, color='blue', label='Raw Data')
        ax2.plot(data['Temperature'], derivative, color='orange', label='Derivative')

        # Plot vertical lines for all Tm values
        if tms is not None:
            for tm in tms:
                ax.axvline(tm, color='red', linestyle='--', label=f'Tm = {tm:.2f}')
        
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax2.grid(False)
        ax2.set_xticks([])
        ax2.set_yticks([])

        row_label = i // 12
        col_label = i % 12

        if col_label == 0:
            ax.set_ylabel(f"{chr(65 + row_label)}", rotation=0, labelpad=10)
        if row_label == 7:
            ax.set_xlabel(f"{col_label + 1}", labelpad=10)
        
        text_content = f'{col}'
        if tms is not None:
            # Display the Tms of all peaks, ordered from highest to lowest
            tms_sorted = sorted(tms, reverse=True)
            text_content += '\n' + '\n'.join([f'{tm:.0f}' for tm in tms_sorted])
        ax.text(0.95, 0.95, text_content, transform=ax.transAxes, ha='right', va='top', fontsize=8, color='black')

    plt.tight_layout()
    plt.show()

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns=['Unnamed: 0'])
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

def get_smoothed_baseline_fluorescence(data, baseline_wells, window_size=5):
    well1_data = data[baseline_wells[0]]
    well2_data = data[baseline_wells[1]]
    baseline = (well1_data + well2_data) / 2
    smoothed_baseline = smooth_data(baseline, window_size)
    return smoothed_baseline

def smooth_data(fluorescence_data, window_size=5):
    return fluorescence_data.rolling(window=window_size).mean()

def compute_derivative(temperature, fluorescence_data):
    return np.gradient(fluorescence_data, temperature)

def find_tm(temperature, derivative):
    peak_height_mult = 0.5
    peak_distance = 1
    derivative_std = np.nanstd(derivative)
    height = peak_height_mult * derivative_std
    peaks, _ = find_peaks(derivative, height=height, distance=peak_distance, prominence=None)
    if len(peaks) == 0:
        return None
    tms = temperature[peaks]
    return tms


if __name__ == '__main__':
    main()
