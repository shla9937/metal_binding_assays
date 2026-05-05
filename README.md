# DSF metal specificity pipeline

A pipeline for running and analyzing differential scanning fluorimetry (DSF) metal-binding screens with automated setup on an Opentrons OT-2 liquid handling robot and data acquisition on an Applied Biosystems QuantStudio 7 qPCR. This method is set up to extract Kd values for up to 32 metals simultaneously using 12 well titration series. **It could be adapted for any non-hydrophobic ligand.**

---

## Table of contents

1. [Opentrons protocols](#opentrons-protocols)
   - [`ot2_dsf_30_metals_triplicate.py`](#ot2_dsf_30_metals_triplicatepy)
   - [`ot2_dsf_6_metals_quadruplicate.py`](#ot2_dsf_6_metals_quadruplicatepy)
   - [`ot2_dsf_384well_cleaning.py`](#ot2_dsf_384well_cleaningpy)
2. [Python environment setup](#python-environment-setup)
3. [Running the analysis](#running-the-analysis)
4. [Outputs](#outputs)
5. [Metal specificity app](#metal-specificity-app)
6. [Plate layout & concentrations](#plate-layout--concentrations)

---

## Opentrons protocols

To be run with the **Opentrons App api level 2.26 or higher** on the OT-2 robot. 

---

### `ot2_dsf_30_metals_triplicate.py`

**Purpose:** Screen up to 32 metals (configured for 29) against a single protein in triplicate or 3 proteins in singlicate across 3 × 384-well plates in a single robot run.

> **Key difference from the 6-metal script:** Metal stocks are prepared in **Falcon 15 mL tubes** and loaded into two tube racks on the deck. The robot uses the p300 single to pre-dilute metals into a 96-well staging plate before the p20 multi performs the titration.

**Deck layout:**

| Slot | Labware |
|---|---|
| 1–3 | Applied Biosystems MicroAmp 384-well qPCR plates |
| 4 | Greiner 96-well plate (metal dilution staging) |
| 5 | 300 µL tip rack |
| 6 | NEST 12-well reservoir (buff) |
| 7 | Opentrons 15-tube rack — Falcon 15 mL (metals 1–15) |
| 8 | Opentrons 15-tube rack — Falcon 15 mL (metals 16–29 + EDTA) |
| 10–11 | 20 µL tip racks |

**Pipettes:** p20 multi (right), p300 single (left)

**Stock preparation (prepare before run):**

| Reagent | Stock concentration | Final concentration | Volume needed |
|---|---|---|---|
| Metal chlorides | 100x (10 mM) | 100 µM | >500 µL into Falcon |
| EDTA | 5× (500 µM) | 100 µM | >500 µL into last Falcon |
| Protein + Sypro + ROX | 5× (25 µM, 50×, 250 nM) | 5 µM, 10×, 50 nM | 6 mL total → 250 µL/well into last 3 columns of staging plate |
| Buff | 1x | 1x | 10 mL in trough well 1; 5 mL each in wells 2–4 |

*Buff should be 100 mM Good's buff (MES or similar), 150 mM NaCl, pH ≤ 6*

**Dilution series:** 12-point 1:2 dilution — 100 µM → 48.8 nM (using `dilution_factor = 1`)

**Run sequence per plate:**
1. Dilutes metals from Falcons into 96-well staging plate
2. Adds buff + protein/Sypro mix to all wells
3. Adds metal stock to column 1 (or 13) and performs serial dilution across 12 columns
4. Pauses and prompts operator to transfer plate to qPCR reader before starting the next replicate

---

### `ot2_dsf_6_metals_quadruplicate.py`

**Purpose:** Detailed titration of 6 metals + EDTA + Apo in quadruplicate on a single 384-well plate.

**When to use:** Used to bechmark original method, could be used if you only care about a smaller set of ligands.

> **Key difference from the 30-metal script:** Metal stocks are pipetted **directly into the Greiner 96-well staging plate by hand** (one metal per column, rows A–H). There are no Falcon tubes and no robot pre-dilution step. The robot reads directly from the staging plate wells.

**Deck layout:**

| Slot | Labware |
|---|---|
| 2 | 20 µL tip rack |
| 4 | Greiner 96-well plate (metal stocks loaded manually — one metal per column, protein/Sypro in column 12) |
| 5 | Corning 384-well flat-bottom plate |
| 6 | NEST 12-well reservoir (1× buff in well 1) |

**Pipettes:** p20 multi only (right)

**Stock preparation:**

| Reagent | Stock concentration | Final concentration | Volume needed |
|---|---|---|---|
| Each metal chloride | 5× (500 µM) | 100 µM | ~50 µL into staging well (one column per metal) |
| EDTA | 5× (500 µM) | 100 µM | ~50 µL into staging well G1 |
| Buff | 1× | 1x | ~50 µL into staging well H1 and 10 ml in to trough|
| Protein + Sypro + ROX | 5× (25 µM, 50×, 250 nM) | 5 µM, 10×, 50 nM | ~2 mL → 250 µL into column 12 of staging plate |

*Buff should be 100 mM Good's buff (MES or similar), 150 mM NaCl, pH ≤ 6*

**Dilution series:** 12-point 1:2 dilution — 100 µM → 48.8 nM (`dilution_factor = 1`)

**Layout:** All 16 rows of the 384-well plate are used, with 4 replicates of the same 6-metal + EDTA + Apo panel (left half = metals 1–4 × 4 row-pairs; right half = metals 5–8 × 4 row-pairs). Analyze with `-ms 6`.

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

## Python environment setup

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

---

## Running the analysis

The analysis script reads Bio-Rad DA2 exported CSV files and produces Kd fits and publication-quality figures.

### How the analysis works

The pipeline processes data in the following order:

1. **Load & filter** — reads one or more DA2 CSVs; temperature bounds (`-lt`, `-ht`) are applied here
2. **Smooth** — applies a Savitzky-Golay filter to each well's fluorescence and derivative traces
3. **Normalize** — Min-Max scales each well's smoothed fluorescence to [0, 1]
4. **Average** — if multiple CSV files are provided, replicates are averaged and a standard error is computed per (metal, concentration, temperature) point
5. **Find Tms** — the peak of the averaged derivative trace is taken as Tm for each well
6. **Fit Kds** — for each metal, fluorescence at the Apo Tm is extracted across the concentration series and fit to the chosen binding model
7. **Plot & save** — all figures and CSVs are written to the current directory

### All arguments

| Flag | Long form | Description |
|-----|---|---|
| `-c` | `--csv` | One or more Applied Bio DA2 CSV files. Multiple files are averaged. |
| `-p` | `--protein`  | Protein name (used in titles and output filenames). |
| `-ms` | `--metal_set` | `29` (default) or `6` — selects the plate metal assignment map. |
| `-lt` | `--low_temp` | Exclude temperatures **below** this value (°C). |
| `-ht` | `--high_temp` | Exclude temperatures **above** this value (°C). |
| `-o` | `--override` | Use this temperature (°C) as the analysis point instead of the Apo Tm. |
| `-m` | `--model` | Binding model: `hill` (default), `two-site`, or `quadratic`. |
| `-w` | `--exclude_wells` | Space-separated well positions to drop (e.g. `-w A1 B3`). |

### Basic usage

```bash
# 29-metal screen, single replicate
dsf_analysis.py -c run1.csv -p MyProtein -ms 29

# 29-metal screen, three replicates averaged together
dsf_analysis.py -c rep1.csv rep2.csv rep3.csv -p MyProtein -ms 29

# 6-metal quadruplicate screen, trim temperature range
dsf_analysis.py -c screen.csv -p MyProtein -ms 6 -lt 60 -ht 95
```

### Iterative temperature trimming workflow

**In almost all cases, run the script twice:**

**Step 1 — initial run (no temperature limits):**
```bash
dsf_analysis.py -c *.csv -p MyProtein -ms 29
```
Inspect the raw fluorescence PDF. Identify any pre-melting fluorescence rise at low temperature or post-melt aggregation artefacts at high temperature. These corrupt normalization because Min-Max scaling uses the global min and max of the trace — if a spurious peak or rise is present, the true melting transition will be compressed and Tms will be wrong.

**Step 2 — refined run with trimmed temperature range:**
```bash
dsf_analysis.py -c *.csv -p MyProtein -ms 29 -lt 45 -ht 95
```
Set `-lt` just below where the melt begins and `-ht` just above where it ends. Re-inspect the smoothed fluorescence PDF to confirm the sigmoid is well-resolved and normalization looks clean before trusting the Kd fits.


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

## Metal specificity app

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

## Plate layout & concentrations

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

- **Apo** — buff-only wells; sets the reference Tm used for all Kd calculations
- **EDTA** — chelator control; expected to destabilize metal-loaded protein or show no shift for apo protein
