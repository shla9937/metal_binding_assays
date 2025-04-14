#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def main():
    baseline_dict = {
        ('D11', 'D11'): ['A1','A2','A3','A4','A5','B1','B2','B3','B4','B5','C1','C2','C3','C4','C5', 
                         'D1','D2','D3','D4','D5','E1','E2','E3','E4','E5','F1','F2','F3','F4','G5','G1',
                         'G2','G3','G4','G5','A11','A12', 'D11', 'D12'],
        ('E11', 'E11'): ['H1','H2','H3','H4','H5','A6','A7','A8','A9','A10','B6','B7','B8','B9','B10', 
                         'C6','C7','C8','C9','C10','D6','D7','D8','D9','D10','B11','B12', 'E11','E12'],
        ('F11', 'F11'): ['E6','E7','E8','E9','E10','F6','F7','F8','F9','F10','G6','G7','G8','G9','G10', 
                         'H6','H7','H8','H9','H10','C11','C12', 'F11','F12']
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

    # Compute dynamic y-limits based on the central 95% of valid values
    all_fluorescence = []
    all_derivatives = []
    for col in fluorescence_columns:
        if col not in baseline_corrected_data.columns:
            print(f"Warning: Column {col} not found in data.")
            continue
        fluorescence_data = baseline_corrected_data[col]
        smoothed_fluorescence = smooth_data(fluorescence_data)
        derivative = compute_derivative(data['Temperature'], smoothed_fluorescence)
        
        # Filter out NaN or infinite values
        valid_fluoro = smoothed_fluorescence[np.isfinite(smoothed_fluorescence)]
        valid_deriv = derivative[np.isfinite(derivative)]
        
        all_fluorescence.extend(valid_fluoro)
        all_derivatives.extend(valid_deriv)
    
    # Convert to numpy arrays
    all_fluo = np.array(all_fluorescence)
    all_deriv = np.array(all_derivatives)
    
    # Set limits to the 2.5th and 97.5th percentiles to capture only the central 95% of data
    y_min = np.percentile(all_fluo, 5) * 1.25
    y_max = np.percentile(all_fluo, 90) * 1.25

    derivative_y_min = np.percentile(all_deriv, 2.5) * 1.25
    derivative_y_max = np.percentile(all_deriv, 100) * 1.25

    # Plot each column using the computed dynamic limits
    for i, col in enumerate(fluorescence_columns):
        if col not in baseline_corrected_data.columns:
            print(f"Warning: Column {col} not found in data.")
            continue

        fluorescence_data = baseline_corrected_data[col]
        smoothed_fluorescence = smooth_data(fluorescence_data)
        derivative = compute_derivative(data['Temperature'], smoothed_fluorescence)
        tm = find_tm(data['Temperature'], derivative)
        
        ax = axes[i]
        ax2 = ax.twinx()

        ax.plot(data['Temperature'], smoothed_fluorescence, color='blue', label='Raw Data')
        ax2.plot(data['Temperature'], derivative, color='orange', label='Derivative')
        
        if tm is not None:
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
        if tm is not None:
            text_content += f'\n{tm:.0f}'
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
    tm_candidates = []
    num_std_dev_threshold = 2  # Adjust to tune sensitivity

    derivative_mean = np.nanmean(derivative)
    derivative_std = np.nanstd(derivative)
    derivative_max = np.nanmax(derivative)
    derivative_min = np.nanmin(derivative)
    derivative_range = derivative_max - derivative_min
    increase_threshold = 0.2 * derivative_range
    std_dev_threshold = derivative_mean + num_std_dev_threshold * derivative_std

    for i in range(1, len(temperature) - 5):
        left_idx = i - 5
        right_idx = i + 5

        if np.isnan(derivative[left_idx]) or np.isnan(derivative[i]) or np.isnan(derivative[right_idx]):
            continue

        if derivative[left_idx] < derivative[i] and derivative[right_idx] < derivative[i]:
            if (temperature[right_idx] - temperature[left_idx] >= 1):
                if (derivative[i] - derivative[left_idx] >= increase_threshold) or \
                   (derivative[right_idx] - derivative[i] >= increase_threshold):
                    if derivative[i] > std_dev_threshold:
                        tm_candidates.append((temperature[i], derivative[i]))

    if tm_candidates:
        tm_candidates.sort(key=lambda x: x[1], reverse=True)
        return tm_candidates[0][0]
    else:
        return None

if __name__ == '__main__':
    main()
