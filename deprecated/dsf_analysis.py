#!/Users/slaursen/anaconda3/bin/python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy.optimize as opt
from matplotlib import cm
from matplotlib.colors import Normalize

def main():
    parser = argparse.ArgumentParser(description="Process and analyze RFU vs Temperature data")
    parser.add_argument('-f', '--file', required=True, help="Input CSV file")
    parser.add_argument('--low', type=float, default=25.0, help="Low temperature cut (default: 25.0C)")
    parser.add_argument('--high', type=float, default=30.0, help="High temperature cut (default: 30.0C)")
    parser.add_argument('-cm', '--colormap', type=str, default='Blues',
                        help="Choose the colormap (default: 'Blues')")
    parser.add_argument('-t', '--title', type=str, default="Melting Temperature Analysis", help="Overarching title for the plots")
    args = parser.parse_args()

    global df
    df = pd.read_csv(args.file, sep=",", encoding="utf-8", header=0, on_bad_lines='skip')
    df.columns = df.columns.str.replace('¬∞', '°').str.replace('¬µ', 'µ')

    temp_col = "Temperature"
    concentration_cols = [col for col in df.columns if col != temp_col]
    temp_cut_low = args.low
    temp_cut_high = args.high
    colormap_name = args.colormap
    title = args.title

    derivatives = find_first_derivative(df, temp_col, concentration_cols)
    smoothed_derivatives = smooth_data(derivatives)
    melting_temps = find_melting_temp(smoothed_derivatives, temp_col, temp_cut_low, temp_cut_high)

    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    plot_raw_data(axs[0], concentration_cols, colormap_name)
    plot_first_derivative(axs[1], smoothed_derivatives, concentration_cols, temp_cut_low, temp_cut_high, colormap_name)
    plot_melting_temperatures(axs[2], melting_temps, colormap_name)

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(bottom=0.2, right=0.9, top=0.75)  # Adjust top to bring title closer
    plt.show()

def plot_raw_data(ax, concentration_cols, colormap_name):
    cmap = getattr(cm, colormap_name + '_r')
    norm = Normalize(vmin=0, vmax=len(concentration_cols))
    for i, conc_col in enumerate(concentration_cols):
        color = cmap(norm(i))
        ax.plot(df['Temperature'], df[conc_col], label=conc_col, color=color)
    ax.set_title("Raw Data (RFU vs Temperature)")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("RFU")
    ax.legend(loc='center right', fontsize='small', markerscale=0.8)

def plot_first_derivative(ax, smoothed_derivatives, concentration_cols, temp_cut_low, temp_cut_high, colormap_name):
    cmap = getattr(cm, colormap_name + '_r')
    norm = Normalize(vmin=0, vmax=len(concentration_cols))
    for i, conc_col in enumerate(concentration_cols):
        valid_indices = (df['Temperature'] >= temp_cut_low) & (df['Temperature'] <= temp_cut_high)
        valid_temperatures = df['Temperature'][valid_indices].values
        valid_derivative = smoothed_derivatives[conc_col][valid_indices]
        color = cmap(norm(i))
        ax.plot(valid_temperatures, valid_derivative, label=conc_col, color=color)
    ax.set_title("Smoothed First Derivative")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("First Derivative (RFU/°C)")
    ax.legend(loc='center right', fontsize='small', markerscale=0.8)

def plot_melting_temperatures(ax, melting_temps, colormap_name):
    cmap = getattr(cm, colormap_name)
    concentrations = [float(conc_col.replace('uM', '').replace('µ', '')) for conc_col in melting_temps.keys()]
    melting_temps_values = list(melting_temps.values())
    plot_max = max(concentrations)
    norm = Normalize(vmin=min(concentrations), vmax=max(concentrations))
    
    for i, conc in enumerate(concentrations):
        color = cmap(norm(conc))
        ax.scatter(conc, melting_temps_values[i], color=color, label=f"{concentrations[i]} µM" if i == 0 else "", zorder=5, edgecolor='black')
    
    valid_indices = [i for i, conc in enumerate(concentrations) if conc > 0]
    fit_concentrations = np.array([concentrations[i] for i in valid_indices])
    fit_melting_temps = np.array([melting_temps_values[i] for i in valid_indices])
    
    def hill_eq(x, ymin, ymax, K, n):
        small_constant = 0
        return ymin + (ymax - ymin) / (1 + (x / (K + small_constant)) ** n)
    
    if len(fit_concentrations) > 1:
        initial_guess = [min(fit_melting_temps), max(fit_melting_temps), np.median(fit_concentrations), 1.0]
        popt, _ = opt.curve_fit(hill_eq, fit_concentrations, fit_melting_temps, p0=initial_guess, maxfev=100000)
        ymin, ymax, K, n = popt
        fit_x = np.linspace(min(fit_concentrations), max(fit_concentrations), 100)
        fit_y = hill_eq(fit_x, *popt)
        ax.plot(fit_x, fit_y, color='black', label="Hill Equation Fit")
        
        # Calculate R-squared
        residuals = fit_melting_temps - hill_eq(fit_concentrations, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((fit_melting_temps - np.mean(fit_melting_temps))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        midpoint_x = K
        midpoint_y = ymin + (ymax - ymin) / 2
        
        # Add Tm difference, Kd, and R-squared to the plot
        max_tm = max(melting_temps_values)
        min_tm = melting_temps['0']
        tm_diff = max_tm - min_tm
        ax.plot([plot_max, plot_max], [min_tm + 0.2, max_tm - 0.2], 'black', linestyle="-")
        ax.plot([plot_max - 0.5, plot_max + 0.5], [max_tm - 0.2, max_tm - 0.2], 'black', linestyle="-")
        ax.plot([plot_max - 0.5, plot_max + 0.5], [min_tm + 0.2, min_tm + 0.2], 'black', linestyle="-")
        ax.text(22, (max_tm + min_tm) / 2, f"ΔTm={tm_diff:.2f}°C \nKd={K:.2f}µM \nR²={r_squared:.2f}", va='center', fontsize=10, color='black')
    
    ax.set_title("Melting Temperatures (Tm) \n vs Concentration")
    ax.set_xlabel("Concentration (µM)")
    ax.set_ylabel("Melting Temperature (°C)")
    ax.set_ylim(50, 100)
    ax.legend().set_visible(False)

def find_first_derivative(df, temp_col, concentration_cols):
    temp = df[temp_col].values
    derivatives = {}
    for conc_col in concentration_cols:
        y = df[conc_col].values
        derivative = np.gradient(y, temp)
        derivatives[conc_col] = derivative
    return derivatives

def smooth_data(derivatives, order=0):
    smoothed = {}
    for conc_col, derivative in derivatives.items():
        smoothed[conc_col] = derivative
    return smoothed

def find_melting_temp(derivatives, temp_col, temp_cut_low, temp_cut_high):
    melting_temps = {}
    for conc_col, derivative in derivatives.items():
        valid_indices = (df[temp_col] >= temp_cut_low) & (df[temp_col] <= temp_cut_high)
        valid_temperatures = df[temp_col][valid_indices].values
        valid_derivative = derivative[valid_indices]
        min_derivative_index = np.argmin(valid_derivative)
        Tm = valid_temperatures[min_derivative_index]
        melting_temps[conc_col] = Tm
    return melting_temps

if __name__ == "__main__":
    main()
