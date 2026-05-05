# DSF Metal Specificity Pipeline

A pipeline for running and analyzing Differential Scanning Fluorimetry (DSF) metal-binding screens on a Bio-Rad CFX (via Opentrons OT-2 liquid handling) and extracting Kd values for up to 29 metals simultaneously.

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Opentrons Protocols](#opentrons-protocols)
   - [29-Metal Triplicate Screen](#ot2_dsf_30_metals_triplicatepy)
   - [8-Metal Quadruplicate Screen](#ot2_dsf_8_metals_quadruplicatepy)
   - [384-Well Plate Cleaning](#ot2_dsf_384well_cleaningpy)
3. [Running the Analysis](#running-the-analysis)
4. [Outputs](#outputs)
5. [Metal Specificity App](#metal-specificity-app)
6. [Plate Layout & Concentrations](#plate-layout--concentrations)

---

## Environment Setup

Dependencies are managed via conda. The environment file is `metal_env.yml`.

```bash
conda env create -f metal_env.yml
conda activate metal
```

Key packages installed:

| Package | Version | Role |
|---|---|---|
| Python | 3.10 | Runtime |
| numpy | 1.26 | Numerics |
| pandas | 2.2 | Data wrangling |
| matplotlib | 3.9 | Figure generation |
| scipy | (conda-forge) | Curve fitting, signal filtering |
| scikit-learn | (pip) | Min-Max normalization |
| dash / plotly | 3.x / 6.x | Specificity web app |

> **Note:** The analysis script (`dsf_analysis.py`) is self-contained and does not require the Opentrons package. The OT-2 scripts run on the robot's built-in Python environment and do **not** use this conda environment.

---

## Opentrons Protocols

All three scripts are uploaded to and executed from the **Opentrons App** on the OT-2. They are not run locally.

---

### `ot2_dsf_30_metals_triplicate.py`

**Purpose:** Screen up to 29 metals against a single protein in triplicate across 3 × 384-well plates in a single robot run.

**When to use:** First-pass screen to identify which metals bind your protein and to get triplicate Kd measurements with good statistics.

**Deck layout:**

| Slot | Labware |
|---|---|
| 1–3 | Applied Biosystems MicroAmp 384-well qPCR plates |
| 4 | Greiner 96-well plate (metal dilution staging) |
| 5 | 300 µL tip rack |
| 6 | NEST 12-well reservoir (buffer + per-plate buffer wells) |
| 7 | Opentrons 15-tube rack — Falcon 15 mL (metals 1–15) |
| 8 | Opentrons 15-tube rack — Falcon 15 mL (metals 16–29 + extra EDTA) |
| 10–11 | 20 µL tip racks |

**Pipettes:** p20 multi (right), p300 single (left)

**Stock preparation (prepare before run):**

| Reagent | Stock concentration | Final concentration | Volume needed |
|---|---|---|---|
| Each metal chloride | 5× (5 mM or 500 µM) | 1 mM or 100 µM | ~500 µL into Falcon |
| EDTA | 5× (500 mM or 500 µM) | 100 mM or 100 µM | ~500 µL into last Falcon |
| Protein + Sypro + ROX | 5× (25 µM, 50×, 250 nM) | 5 µM, 10×, 50 nM | 6 mL total → 250 µL into last 3 columns of staging plate |
| Buffer | ~100 mM buffer, 150 mM NaCl | — | ~10 mL in trough well 1; ~3 mL each in wells 2–4 |

**Dilution series:** 12-point 1:2 dilution — 100 µM → 48.8 nM (using `dilution_factor = 1`)

**Run sequence per plate:**
1. Dilutes metals from Falcons into 96-well staging plate
2. Adds buffer + protein/Sypro mix to all wells
3. Adds metal stock to column 1 (or 13) and performs serial dilution across 12 columns
4. Pauses and prompts operator to transfer plate to qPCR reader before starting the next replicate

---

### `ot2_dsf_8_metals_quadruplicate.py`

**Purpose:** Detailed titration of up to 8 metals (typically 6 metals + EDTA + Apo) in quadruplicate on a single 384-well plate.

**When to use:** Follow-up screen after identifying binders from the 29-metal screen, or when you need more replicates and tighter error bars for a smaller metal panel.

**Deck layout:**

| Slot | Labware |
|---|---|
| 2 | 20 µL tip rack |
| 4 | Greiner 96-well plate (metal stocks — one metal per column) |
| 5 | Corning 384-well flat-bottom plate |
| 6 | NEST 12-well reservoir (buffer in well 1) |

**Pipettes:** p20 multi only (right)

> Uses partial-column nozzle configuration to pick up only as many tips as rows needed — no wasted tips.

**Stock preparation:**

| Reagent | Stock concentration | Final concentration | Volume needed |
|---|---|---|---|
| Each metal | 5× (5 mM or 500 µM) | 1 mM or 100 µM | ~50 µL into staging well |
| EDTA | 5× | same as metals | ~50 µL |
| Buffer (Apo) | 5× or neat | — | ~50 µL |
| Protein + Sypro + ROX | 5× (25 µM, 50×, 250 nM) | 5 µM, 10×, 50 nM | ~2 mL → 250 µL into well H12 of staging plate |

**Layout:** All 16 rows of the 384-well plate are used, with 4 replicates of the same 8-metal panel (left half = metals 1–4 × 4 row-pairs; right half = metals 5–8 × 4 row-pairs). Analyze with `-ms 6`.

---

### `ot2_dsf_384well_cleaning.py`

**Purpose:** Wash used 384-well qPCR plates for reuse. Performs three water wash cycles.

**When to use:** After reading plates on the qPCR instrument.

**Runtime parameter:** Set the number of plates (1–8) in the Opentrons App before running.

**Deck layout:**

| Slot | Labware |
|---|---|
| 1–8 | 384-well plates to be washed (up to 8) |
| 7–9 | NEST 12-well reservoirs (wash water) |
| 10 | NEST 1-well 195 mL reservoir (liquid waste) |
| 11 | 300 µL tip rack |

**Pipette:** p300 multi (left)

**Wash cycle:** For each of 3 water reservoirs — dispense 30 µL/well → mix 3× → aspirate to waste.

---

## Running the Analysis

The analysis script reads Bio-Rad DA2 exported CSV files and produces Kd fits and publication-quality figures.

### Basic usage

```bash
# 29-metal screen, single replicate
dsf_analysis.py -c run1.csv -p MyProtein -ms 29

# 29-metal screen, three replicates averaged together
dsf_analysis.py -c rep1.csv rep2.csv rep3.csv -p MyProtein -ms 29

# 6-metal quadruplicate screen, trim temperature range
dsf_analysis.py -c screen.csv -p MyProtein -ms 6 -lt 60 -ht 95
```

### All arguments

| Flag | Long form | Required | Description |
|---|---|---|---|
| `-c` | `--csv` | ✅ | One or more Bio-Rad DA2 CSV files. Multiple files are averaged. |
| `-p` | `--protein` | ✅ | Protein name (used in titles and output filenames). |
| `-ms` | `--metal_set` | | `29` (default) or `6` — selects the plate metal assignment map. |
| `-lt` | `--low_temp` | | Exclude temperatures **below** this value (°C). |
| `-ht` | `--high_temp` | | Exclude temperatures **above** this value (°C). |
| `-o` | `--override` | | Use this temperature (°C) as the analysis point instead of the Apo Tm. |
| `-m` | `--model` | | Binding model: `hill` (default), `two-site`, or `quadratic`. |
| `-w` | `--exclude_wells` | | Space-separated well positions to drop (e.g. `-w A1 B3`). |

### Binding models

| Model | Description | Best for |
|---|---|---|
| `hill` | Standard Hill equation with cooperativity coefficient *n* | Most cases; allows for cooperative binding |
| `quadratic` | Quadratic binding (Bai et al. 2018); accounts for ligand depletion | When protein and metal concentrations are similar |
| `two-site` | Two independent sites with Kd1 and Kd2 | When titration curves show biphasic behavior |

### QC thresholds (hardcoded)

| Parameter | Value | Meaning |
|---|---|---|
| `tm_threshold` | 1.0 °C | Minimum Tm shift across the titration to be considered a real signal |
| `r2_threshold` | 0.7 | Minimum R² for a fit to be reported; below this the metal is marked NB (no binding) |
| `signal_threshold` | 0.15 | Minimum normalized fluorescence range across the titration |

---

## Outputs

All output files are written to the **current working directory**. Run the script from the experiment folder.

| File | Description |
|---|---|
| `{protein}_fluorescence_melt_curves.pdf` | Raw fluorescence traces, one panel per metal × row combination, color-coded by concentration |
| `{protein}_smoothed_fluorescence_melt_curves.pdf` | Averaged, normalized melt curves with error shading; Binding SNR shown per panel |
| `{protein}_tm_vs_concentration.pdf` | Scatter plot of Tm vs. metal concentration for all metals |
| `{protein}_metal_titrations.pdf` | Binding curves (% folded at Apo Tm) with fitted Kd lines; one plot per temperature point |
| `{protein}_kd_bar_chart.pdf` | Bar chart of 1/Kd for all metals sorted by atomic number; NB metals shown with hatching |
| `{protein}_titration_data.csv` | Per-metal, per-concentration fluorescence values at the analysis temperature |
| `{protein}_kd_results.csv` | Kd, error, Hill *n*, R², and Binding_SNR for every metal |
| `{protein}_command.txt` | Exact command used to generate the results (for reproducibility) |

All figures are saved as PDF with embedded vector fonts (DejaVu Sans) and transparent backgrounds, sized for journal submission (max 6.9" wide).

---

## Metal Specificity App

`metal_specificity_app.py` is a local Dash web app for comparing Kd values across multiple proteins side by side.

### Running the app

```bash
python metal_specificity_app.py
```

Then open `http://127.0.0.1:8050` in a browser.

### Input format

Upload one or more `*_kd_results.csv` files (output from `dsf_analysis.py`). The app expects:
- First column: metal names
- Remaining columns: one Kd value per protein (in µM)

### What it shows

- **Heatmap:** Per-protein, column-normalized log₁₀(Kd) grayscale heatmap. Darker = tighter binding.
- **Specificity row:** Best-binding metal per protein and the fold-selectivity over the second-best metal.
- **Hover text:** Raw Kd values with auto-scaled units (nM / µM / mM).

---

## Plate Layout & Concentrations

### 29-metal layout (columns 1–24, rows A–P)

```
Columns  1–12  (left half):   Li, Cu, Mg, Zn, K, Rb, Ca, Sr, Sc, Y, Mn, Cs, Co, Ba, Ni, La
Columns 13–24  (right half):  Ce, Ho, Pr, Er, Nd, Tm, Sm, Yb, Eu, Lu, Gd, EDTA, Tb, EDTA, Dy, Apo
```

Each row titrates its assigned metal across 12 wells in a 1:2 dilution series:

```
100 µM → 50 → 25 → 12.5 → 6.25 → 3.13 → 1.56 → 0.781 → 0.391 → 0.195 → 0.0977 → 0.0488 µM
```

### 6-metal layout (`-ms 6`)

All 16 rows use the same 8-slot panel (6 metals + EDTA + Apo), repeated in quadruplicate across row-pairs:

```
Mn²⁺, Co²⁺, Ni²⁺, Cu²⁺, Nd³⁺, Dy³⁺, EDTA, Apo  (× 2 rows each side)
```

### Apo and EDTA controls

- **Apo** — buffer-only wells; sets the reference Tm used for all Kd calculations
- **EDTA** — chelator control; expected to destabilize metal-loaded protein or show no shift for apo protein
