#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.colors import Normalize

# Hypothetical data (replace with your actual values)
metals = ['Fe³⁺', 'Cu²⁺', 'Zn²⁺', 'Ni²⁺', 'Co²⁺']
Kd_values = [0.2, 0.5, 1.0, 2.0, 5.0]  # in μM
ΔTm = [+10.0, -3.0, +2.0, 0.0, -10]  # Thermal shift values

# Calculate 1/Kd for plotting
inv_Kd = [1/kd for kd in Kd_values]

# Normalize ΔTm values to the range [-10, +10]
norm = Normalize(vmin=-10, vmax=10)

# Create color gradient based on ΔTm using the coolwarm colormap
colors = [plt.cm.coolwarm(norm(tm)) for tm in ΔTm]

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(metals, inv_Kd, color=colors, edgecolor='black')

# Axis configuration
ax.set_ylabel('Binding Affinity (1/Kd)', fontsize=12)
ax.set_xlabel('Metal Ion', fontsize=12)
ax.set_title('Transition Metal Binding Affinities and Stability Effects', pad=20)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Annotate bars with Kd values
for bar, kd in zip(bars, Kd_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height * 1.05,  # Position above the bar
            f'{kd} μM', ha='center', va='bottom', fontsize=10)

# Add ΔTm color legend
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
sm.set_array([])  # No need to pass data; the norm handles the range
cbar = plt.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label('ΔTm (°C)', fontsize=12)

plt.show()
