#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.signal import find_peaks

def main():
    parser = argparse.ArgumentParser(
        description="Analyze DSF Data, baseline correct using a baseline plate"
    )
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to the experimental DSF data CSV file")
    parser.add_argument('-b', '--baseline', type=str, required=True, help="Path to the baseline DSF data CSV file")
    args = parser.parse_args()

    # Load experimental and baseline data
    experimental_data = load_data(args.input)
    baseline_data = load_data(args.baseline)

    fluorescence_columns = [f"{chr(i)}{j}" for i in range(65, 73) for j in range(1, 13)]

    # Subtract smoothed baseline plate signals from experimental plate signals
    baseline_corrected_data = experimental_data.copy()
    for col in fluorescence_columns:
        if col in experimental_data.columns and col in baseline_data.columns:
            smoothed_baseline = smooth_data(baseline_data[col])
            baseline_corrected_data[col] = experimental_data[col] - smoothed_baseline
        else:
            print(f"Warning: Column {col} not found in both experimental and baseline data.")

    # Plotting and analysis
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
        derivative = compute_derivative(experimental_data['Temperature'], smoothed_fluorescence)
        
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
        derivative = compute_derivative(experimental_data['Temperature'], smoothed_fluorescence)
        tms = find_tm(experimental_data['Temperature'], derivative)  # Get all peaks
        
        ax = axes[i]
        ax2 = ax.twinx()

        ax.plot(experimental_data['Temperature'], smoothed_fluorescence, color='blue', label='Raw Data')
        ax2.plot(experimental_data['Temperature'], derivative, color='orange', label='Derivative')

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

        ax.set_ylim(y_min, y_max)
        ax2.set_ylim(derivative_y_min, derivative_y_max)

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

def smooth_data(fluorescence_data, window_size=5):
    return fluorescence_data.rolling(window=window_size).mean()

def compute_derivative(temperature, fluorescence_data):
    return np.gradient(fluorescence_data, temperature)

def find_tm(temperature, derivative):
    peak_height_mult = 0.25
    peak_distance = 1
    derivative_std = np.nanstd(derivative)
    height = peak_height_mult * derivative_std
    peaks, _ = find_peaks(derivative, height=None, distance=peak_distance, prominence=height)
    if len(peaks) == 0:
        return None
    tms = temperature[peaks]
    return tms


if __name__ == '__main__':
    main()