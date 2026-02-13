# Python Requirements for Electrophysiology Analysis Pipeline

This document lists all required Python packages to run the electrophysiology data analysis pipeline end-to-end.

## Python Version

- **Python 3.8+** (tested with Python 3.10)

## Required Packages

### Core Scientific Computing
```
numpy>=1.20.0
pandas>=1.3.0
scipy>=1.7.0
```

### NWB (Neurodata Without Borders) Support
```
pynwb>=2.0.0
h5py>=3.0.0
```

### Data Storage & I/O
```
pyarrow>=6.0.0        # Required for parquet file support
openpyxl>=3.0.0       # Required for Excel template reading/writing
```

### Visualization
```
matplotlib>=3.4.0
```

## Installation

### Option 1: Install all at once (recommended)

Create a virtual environment and install all packages:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install all required packages
pip install numpy pandas scipy pynwb h5py pyarrow openpyxl matplotlib
```

### Option 2: Using requirements.txt

Create a `requirements.txt` file with the following content:

```
numpy>=1.20.0
pandas>=1.3.0
scipy>=1.7.0
pynwb>=2.0.0
h5py>=3.0.0
pyarrow>=6.0.0
openpyxl>=3.0.0
matplotlib>=3.4.0
```

Then install:

```bash
pip install -r requirements.txt
```

## Package Usage in Pipeline

| Package | Used For | Key Scripts |
|---------|----------|-------------|
| `numpy` | Array operations, numerical computations | All scripts |
| `pandas` | DataFrame operations, CSV/Parquet I/O | All data processing scripts |
| `scipy` | Signal processing (Savitzky-Golay filter, peak detection, linear regression) | `spike_detection_new.py`, `sav_gol_filter.py`, `input_resistance.py` |
| `pynwb` | Reading NWB files | `process_human_data.py`, `process_human_data_mixed_protocol.py`, `read_and_plot_nwb.py` |
| `h5py` | HDF5 file operations (NWB protocol detection) | `main.py` |
| `pyarrow` | Parquet file reading/writing (fast columnar storage) | All data extraction scripts |
| `openpyxl` | Excel template reading/writing for metadata logs | `process_human_data.py`, `process_human_data_mixed_protocol.py` |
| `matplotlib` | Plotting sweep data, spike detection visualization | `spike_detection_new.py`, `sav_gol_filter.py`, `input_resistance.py` |

## Verification

After installation, verify all packages are installed correctly:

```bash
python -c "import numpy, pandas, scipy, pynwb, h5py, pyarrow, openpyxl, matplotlib; print('✓ All packages installed successfully')"
```

## Troubleshooting

### Common Issues

1. **`pyarrow` installation fails**
   - On some systems, you may need to install build tools first
   - On macOS: `xcode-select --install`
   - On Linux: `sudo apt-get install build-essential`

2. **`h5py` installation issues**
   - May require HDF5 library
   - On macOS: `brew install hdf5`
   - On Linux: `sudo apt-get install libhdf5-dev`

3. **`pynwb` compatibility**
   - Ensure you have Python 3.8 or higher
   - `pynwb` requires `h5py` to be installed first

### Platform-Specific Notes

- **macOS**: Use `brew` to install system dependencies if needed
- **Linux**: Use `apt` or `yum` depending on your distribution
- **Windows**: Consider using Anaconda/Miniconda for easier scientific package management

## Optional Packages (Not Required)

These packages are imported in some scripts but not essential for core functionality:

- `warnings` (built-in)
- `argparse` (built-in)
- `subprocess` (built-in)
- `pathlib` (built-in)
- `json` (built-in)
- `datetime` (built-in)
- `typing` (built-in)

## Pipeline Execution Order

Once all packages are installed, run the pipeline:

```bash
# Start the interactive pipeline
python main.py

# Or directly process NWB files
python process_human_data.py <parent_dir> <output_dir> <template.xlsx>

# Or analyze existing bundles
python bundle_analyzer.py /path/to/bundle_dir
```

## Additional Resources

- [PyNWB Documentation](https://pynwb.readthedocs.io/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [SciPy Signal Processing](https://docs.scipy.org/doc/scipy/reference/signal.html)
- [PyArrow Documentation](https://arrow.apache.org/docs/python/)
