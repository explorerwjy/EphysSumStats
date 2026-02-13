# NWB Electrophysiology Analysis Pipeline

This pipeline processes intracellular electrophysiology recordings stored in NWB (Neurodata Without Borders) format. It handles both **single protocol** (voltage-only or current-only) and **mixed protocol** (voltage + current) recordings.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Input Requirements](#input-requirements)
- [Pipeline Workflow](#pipeline-workflow)
- [Output Structure](#output-structure)
- [Configuration](#configuration)
- [Memory Optimization](#memory-optimization)

---

## Overview

### What This Pipeline Does
1. **Classifies sweeps** - Identifies valid current-clamp recordings vs artifacts
2. **Detects stimulus windows** - Finds baseline, stimulus, and response periods
3. **Analyzes action potentials** - Detects spikes and extracts morphology features
4. **Calculates RMP** - Measures resting membrane potential using Savitzky-Golay filtering
5. **Computes input resistance** - Calculates membrane properties from hyperpolarizing steps
6. **Generates visualizations** - Creates comprehensive plots and summary PDFs

### Supported Protocols
- **Single Protocol**: Voltage-only OR current-only recordings (e.g., Allen Brain Observatory)
- **Mixed Protocol**: Voltage AND current recordings in same file (e.g., Zuckerman Lab data)

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data
Place NWB files in the workspace directory:
```
Sneha_NWB_Allen_brain/
├── sub-1000610030_ses-1002181694_icephys.nwb  # Single protocol files
├── sub-1000610030_ses-1002208108_icephys.nwb
└── sub-1000610030/                             # Or organize in folders
    └── *.nwb
```

### 3. Run the Pipeline
```bash
python main.py
```

The script will:
- Auto-detect protocol type (single vs mixed)
- Process all `.nwb` files
- Create output folders with analysis results

---

## Input Requirements

### NWB File Structure

**Single Protocol** files must contain:
- `acquisition/` - Voltage or current timeseries data
- `stimulus/` - Stimulus waveforms
- `icephys_sequential_recordings` - Metadata about sweeps

**Mixed Protocol** files must contain:
- `acquisition/` - Voltage response timeseries
- `stimulus/` - Current stimulus timeseries
- Both stimulus and response indexed by sweep number

### Sampling Rate
- Typically 200 kHz for stimulus, 50 kHz for response (mixed protocol)
- Can vary; pipeline auto-detects from NWB metadata

---

## Pipeline Workflow

### Step 1: Sweep Classification (`sweep_classifier.py`)

**Purpose**: Identify valid current-clamp sweeps and filter out artifacts/malformed data

**What it does**:
1. Detects stimulus windows by finding stable current injection periods
2. Identifies baseline (no stimulus) and response windows
3. Filters out:
   - Voltage-clamp sweeps (not current-clamp)
   - Artifact sweeps (sharp corners, voltage jumps)
   - Malformed sweeps (insufficient baseline/stimulus)

**Key outputs**:
- `sweep_config.json` - Contains stimulus/baseline/response windows for each sweep
- Classification plots showing kept vs dropped sweeps

**Configuration** (see `analysis_config.py`):
- `BASELINE_THRESHOLD_PA = 0.01` - Current below this = no injection
- `STIMULUS_THRESHOLD_PA = 5.0` - Current above this = stimulus
- `MIN_STIMULUS_DURATION_S = 0.300` - Minimum 300ms stimulus
- `MIN_FLAT_RATIO = 0.70` - 70% of stimulus must be stable

---

### Step 2: Data Extraction

**Single Protocol** (`process_human_data.py`):
- Extracts voltage and current data from `acquisition/` and `stimulus/`
- Separates by unit type (mV vs pA)
- Creates unified tables: `voltage_all.parquet`, `current_all.parquet`

**Mixed Protocol** (`process_human_data_mixed_protocol.py`):
- Extracts stimulus (current) and response (voltage) separately
- Filters by unit type to handle mixed data
- Creates four tables:
  - `stimulus_all.parquet` - All stimulus data
  - `response_all.parquet` - All response data
  - `stimulus_currentclamp.parquet` - Current (pA) only
  - `response_voltageclamp.parquet` - Voltage (mV) only

**Memory optimization**: Uses save→delete→read-back pattern to avoid holding full dataset in RAM

---

### Step 3: Action Potential Detection (`spike_detection_new.py`)

**Purpose**: Detect and characterize action potential morphology

**What it does**:
1. Smooths voltage traces using adaptive Savitzky-Golay filter
2. Calculates dV/dt (voltage derivative)
3. Finds peaks using `scipy.signal.find_peaks`
4. For each spike, extracts:
   - **Peak** - Maximum voltage
   - **Threshold** - Where dV/dt exceeds 5% of max
   - **Fast trough** - Minimum voltage after peak
   - **Upstroke/downstroke velocity** - Max/min dV/dt
   - **Width** - Duration at half-max amplitude
   - **Latency** - Time from stimulus onset to first spike

**Configuration** (see `analysis_config.py`):
- `PEAK_HEIGHT_THRESHOLD = -10` mV
- `PEAK_PROMINENCE = 20` mV
- `THRESHOLD_PERCENT = 0.05` - 5% of max upstroke
- `MIN_PEAK_THRESHOLD_AMPLITUDE_MV = 15.0` mV

**Outputs**:
- `AP_analysis.csv` - Per-spike features (peak, threshold, width, etc.)
- `AP_Per_Sweep/` - Individual spike plots per sweep
- `Averaged_Peaks_Per_Sweep/` - Averaged AP waveforms

---

### Step 4: Resting Membrane Potential (`sav_gol_filter.py`)

**Purpose**: Calculate RMP and baseline stability during non-stimulus periods

**What it does**:
1. Filters to baseline periods only (no stimulus)
2. Applies Savitzky-Golay smoothing to remove noise
3. Downsamples to 25ms windows
4. Calculates:
   - **Filtered RMP** - Mean baseline voltage
   - **RMP derivative** - Rate of voltage drift (mV/s)
   - **Voltage range** - Max - min during baseline
   - **Std dev** - Baseline noise/stability

**Why this matters**:
- Detects unhealthy cells (drifting baseline)
- Identifies seal quality issues
- Flags sweeps with excessive noise

**Configuration** (see `analysis_config.py`):
- `BASELINE_WINDOW_MS = 25` - Window size for downsampling
- `REFERENCE_SMOOTH_MS = 200.05` - Smoothing window (scales with recording duration)

**Outputs**:
- RMP metrics appended to `analysis.csv`
- `Sav_Gol_Plots_Per_Sweep/` - Visualization of filtered baseline

---

### Step 5: Input Resistance (`input_resistance.py`)

**Purpose**: Calculate membrane input resistance from hyperpolarizing current steps

**What it does**:
1. Identifies hyperpolarizing sweeps (negative current injection)
2. Finds minimum voltage during stimulus (max hyperpolarization)
3. Calculates ΔV (voltage deflection from baseline)
4. Calculates R_input = ΔV / I_injected (Ohm's law)

**Why this matters**:
- Measures membrane integrity
- Indicates cell health
- Key parameter for computational models

**Outputs**:
- Input resistance values in `analysis.csv`
- `Input_Resistance/` - Voltage response plots for hyperpolarizing steps

---

### Step 6: Summary Outputs

**For each NWB file, creates**:
- `analysis.csv` - Per-sweep summary (RMP, input resistance, spike count, etc.)
- `AP_analysis.csv` - Per-spike features (all detected action potentials)
- `manifest.json` - Metadata about the analysis run
- `sweep_config.json` - Stimulus/baseline/response windows for each sweep
- `all_plots_summary.pdf` - Combined visualization of all plots
- `min_frequency.csv` - Row with minimum current that produces spikes
- `max_frequency.csv` - Row with maximum spike frequency
- `mean_frequency.csv` - Mean of all rows up to max frequency

**Grid plots**:
- `voltage_grid.png` / `current_grid.png` (single protocol)
- `stimulus_grid.png` / `response_grid.png` (mixed protocol)
- Show all valid sweeps in grid layout

---

## Output Structure

```
sub-{subject_id}_ses-{session_id}_icephys/
├── analysis.csv                    # Per-sweep summary
├── AP_analysis.csv                 # Per-spike features
├── manifest.json                   # Analysis metadata
├── sweep_config.json               # Window definitions
├── all_plots_summary.pdf           # Combined visualizations
├── min_frequency.csv               # Min current with spikes
├── max_frequency.csv               # Max spike frequency row
├── mean_frequency.csv              # Mean up to max frequency
│
├── voltage_all.parquet             # Raw voltage data (single protocol)
├── current_all.parquet             # Raw current data (single protocol)
│   OR
├── stimulus_all.parquet            # All stimulus data (mixed protocol)
├── response_all.parquet            # All response data (mixed protocol)
├── stimulus_currentclamp.parquet   # pA only (mixed protocol)
├── response_voltageclamp.parquet   # mV only (mixed protocol)
│
├── AP_Per_Sweep/                   # Individual spike plots
│   ├── sweep_0_spikes.png
│   ├── sweep_1_spikes.png
│   └── ...
│
├── Averaged_Peaks_Per_Sweep/       # Averaged AP waveforms
│   ├── sweep_0_avg_peak.png
│   └── ...
│
├── Input_Resistance/               # Hyperpolarizing step responses
│   ├── sweep_0_input_resistance.png
│   └── ...
│
├── Sav_Gol_Plots_Per_Sweep/       # Filtered baseline traces
│   ├── sweep_0_sav_gol.png
│   └── ...
│
└── Grid plots:
    ├── voltage_grid.png            # All voltage sweeps (single)
    ├── current_grid.png            # All current sweeps (single)
    OR
    ├── stimulus_grid.png           # All stimulus sweeps (mixed)
    └── response_grid.png           # All response sweeps (mixed)
```

---

## Configuration

### Central Configuration File: `analysis_config.py`

All analysis parameters are centralized for easy tuning:

```python
# Spike Detection
PEAK_HEIGHT_THRESHOLD = -10          # mV
PEAK_PROMINENCE = 20                 # mV
THRESHOLD_PERCENT = 0.05             # 5% of max dV/dt
MIN_PEAK_THRESHOLD_AMPLITUDE_MV = 15 # mV

# Sweep Classification
BASELINE_THRESHOLD_PA = 0.01         # No injection threshold
STIMULUS_THRESHOLD_PA = 5.0          # Injection threshold
MIN_STIMULUS_DURATION_S = 0.300      # 300ms minimum

# Artifact Detection
SECOND_DERIV_THRESHOLD = 10e9        # Sharp corner detection
VOLTAGE_JUMP_THRESHOLD = 10.0        # Voltage discontinuity (mV)

# Baseline Analysis
BASELINE_WINDOW_MS = 25              # RMP derivative window
```

**To modify thresholds**: Edit `analysis_config.py` - changes apply globally

---

## Memory Optimization

The pipeline includes aggressive memory management for processing large datasets:

### Key Optimizations (see `MEMORY_OPTIMIZATION.md`):

1. **Garbage collection** - `gc.collect()` after each file
2. **Explicit cleanup** - `del dataframe` after saving to disk
3. **Read-from-disk strategy** - For unified tables:
   - Save stimulus → delete → gc
   - Save response → delete → gc
   - Read back only needed columns for filtering
   - Reduces peak memory by ~70%

4. **Parquet format** - Compressed columnar storage for efficient I/O

### Typical Memory Usage:
- **Single file (small)**: 500 MB - 1 GB
- **Single file (large)**: 2-4 GB
- **Mixed protocol**: 3-6 GB (larger due to dual data streams)

**If you encounter memory issues**:
- Close other applications
- Process files one at a time
- See `MEMORY_OPTIMIZATION.md` for detailed troubleshooting

---

## Troubleshooting

### Common Issues

**1. "System has run out of application memory"**
- See `MEMORY_OPTIMIZATION.md`
- Try processing fewer files at once
- Restart Python kernel between runs

**2. "No valid sweeps found"**
- Check NWB file has current-clamp data (not voltage-clamp)
- Verify stimulus duration ≥ 300ms
- Review `sweep_config.json` to see why sweeps were dropped

**3. "Baseline duration < 25ms"**
- Sweeps need ≥50ms baseline for RMP derivative calculation
- Short baseline sweeps will have NaN for RMP derivative
- This is normal for some protocols - sweeps still analyzed for spikes

**4. Poor PDF quality**
- Individual plots saved as JPG at 100 DPI
- For higher quality, increase DPI in `sweep_classifier.py`
- Trade-off: larger file sizes

---

## File Descriptions

### Core Pipeline Files
- `main.py` - Entry point; orchestrates the entire pipeline
- `process_human_data.py` - Processes single protocol NWB files
- `process_human_data_mixed_protocol.py` - Processes mixed protocol NWB files
- `sweep_classifier.py` - Sweep classification and window detection
- `spike_detection_new.py` - Action potential detection and analysis
- `sav_gol_filter.py` - RMP calculation using Savitzky-Golay filtering
- `input_resistance.py` - Input resistance calculation
- `analysis.py` - Helper functions for data extraction

### Configuration
- `analysis_config.py` - Central parameter configuration
- `requirements.txt` - Python package dependencies

### Documentation
- `README.md` - This file
- `MEMORY_OPTIMIZATION.md` - Memory management strategies
- `CSV_PARAMETERS_GUIDE.txt` - Description of output CSV columns
- `Why_Sampling_Rate_Needed_SavGol.md` - Technical notes on filtering

---

## Citation & Data Sources

### Allen Brain Observatory
- Human intracellular electrophysiology data
- Single protocol recordings (voltage OR current)
- https://portal.brain-map.org/

### Zuckerman Institute
- Mixed protocol recordings (voltage AND current)
- Custom experimental protocols

---

## Support

For questions or issues:
1. Check `MEMORY_OPTIMIZATION.md` for memory-related problems
2. Review `CSV_PARAMETERS_GUIDE.txt` for output column definitions
3. Inspect `sweep_config.json` in output folders to understand sweep classification
4. Check `manifest.json` for analysis metadata and error logs

---

## Changelog

### February 7, 2026 (v01-07)
- **Improved plot quality**: Changed all image outputs from JPEG to PNG (lossless compression)
- **Higher resolution**: Increased DPI from 100/150 to 300 for all plots and PDF export
- **Better PDF rendering**: Added lanczos interpolation and auto-scaling for `all_plots_summary.pdf`
- **New output**: Added `mean_frequency.csv` - contains mean values of all columns from first row up to max frequency row
- **Sav-Gol filter fix**: Sweeps with no stimulus (0 pA) are now excluded from baseline filtering. These sweeps have no pre-stimulus baseline period, which caused erroneous filtering.
- **Debug logging**: Added diagnostic output for Sav-Gol filter to help troubleshoot filtering issues across different machines

### January 2026 (v01-06)
- Initial release of unified NWB analysis pipeline
- Support for single protocol (Allen Brain) and mixed protocol (Zuckerman) data
- Comprehensive spike detection and AP morphology analysis
- Savitzky-Golay baseline filtering
- Input resistance calculation
- Automatic sweep classification and artifact rejection

---

**Last Updated**: February 7, 2026
