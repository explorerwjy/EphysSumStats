# This file runs a low pass filter through the data and calculates the input resistance
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import linregress
from pathlib import Path
from analysis_config import (
    get_analysis_window_bounds,
    MIN_PEAK_DISTANCE_MS,
    MIN_PEAK_DISTANCE_S,
    PEAK_HEIGHT_THRESHOLD,
    PEAK_PROMINENCE,
)

def get_input_resistance(df, df_pA, bundle_path, sweep_config=None):

    # Calculate input resistance
    print("CALCULATING INPUT RESISTANCE")
    print("NOTE: Input resistance is calculated during the stimulus period")
    print("      using sweeps with no action potentials (low current sweeps)")
    bundle_path = Path(bundle_path)
    
    # Load manifest to detect protocol type
    man = json.loads((Path(bundle_path) / "manifest.json").read_text())
    is_mixed = "stimulus" in man.get("tables", {}) and "response" in man.get("tables", {})
    print(f"Protocol type: {'MIXED' if is_mixed else 'SINGLE'}")
    
    # For input resistance, use JUST the stimulus window (no pre/post expansion)
    # Get the raw stimulus window from sweep_config
    if sweep_config is None:
        raise ValueError("sweep_config is required for input resistance calculation")
    
    try:
        # Find first valid sweep to get stimulus window
        valid_sweep = None
        t_stim_min = None
        t_stim_max = None
        
        for sweep_id, sweep_data in sweep_config.get("sweeps", {}).items():
            if sweep_data.get("valid", False):
                valid_sweep = sweep_id
                windows = sweep_data.get("windows", {})
                t_stim_min = windows.get("stimulus_start_s")
                t_stim_max = windows.get("stimulus_end_s")
                break
        
        if t_stim_min is None or t_stim_max is None:
            raise ValueError("Could not find stimulus window in sweep_config")
            
        print(f"Using stimulus window: [{t_stim_min:.6f}, {t_stim_max:.6f}] s")
    except (KeyError, TypeError) as e:
        raise ValueError(f"Failed to extract stimulus window from sweep_config: {e}")
    
    # For SINGLE protocol: filter using absolute times directly
    # For MIXED protocol: filter using absolute times (convert per-sweep below)
    if is_mixed:
        # For mixed protocol, we need to filter per-sweep with converted absolute times
        # Start with all sweeps and filter per-sweep inside the loop
        df_spikes_all = df
        df_pA_all = df_pA
    else:
        # For single protocol: use stimulus times directly as absolute times
        df_spikes_all = df[(df["t_s"]>=t_stim_min) & (df["t_s"]<=t_stim_max)]
        df_pA_all = df_pA[(df_pA["t_s"]>=t_stim_min) & (df_pA["t_s"]<=t_stim_max)]
    
    print(f"  mV data: {len(df_spikes_all)} rows, sweeps: {df_spikes_all['sweep'].unique()[:5]}...")
    print(f"  pA data: {len(df_pA_all)} rows, sweeps: {df_pA_all['sweep'].unique()[:5]}...")
    
    # Load analysis results to identify which sweeps have no spikes
    # This is more reliable than re-detecting peaks
    df_analysis = pd.read_parquet(bundle_path / "analysis.parquet")
    no_spike_sweeps = set(df_analysis[df_analysis['spike_frequency_Hz'] == 0]['sweep'].tolist())
    print(f"  Sweeps with NO spikes (from analysis): {len(no_spike_sweeps)} sweeps")
    print(f"  NOTE: Only using first 8 sweeps for input resistance calculation")
    
    # If mV data has multiple channels (can happen after hardware malfunction fix),
    # filter to keep only one channel (the one we selected as correct)
    if "channel_index" in df_spikes_all.columns:
        channels = df_spikes_all["channel_index"].unique()
        if len(channels) > 1:
            # Use only the first channel (or the most common one)
            primary_channel = df_spikes_all["channel_index"].value_counts().idxmax()
            print(f"  Note: Multiple mV channels detected. Using channel {primary_channel}")
            df_spikes_all = df_spikes_all[df_spikes_all["channel_index"] == primary_channel]

    current_avg_vals = []
    voltage_avg_vals = []
    valid_sweeps_found = 0
    
    # Only use first 8 sweeps that have no spikes for input resistance calculation
    MAX_NO_SPIKE_SWEEPS = 8
    sweeps_used_count = 0
    
    # For mixed protocol: filter per-sweep with absolute times converted from relative
    # For single protocol: already filtered above, just iterate
    for sweep_number, group in df_spikes_all.groupby("sweep"):
        # Skip sweeps that have spikes (use analysis results, not re-detection)
        if sweep_number not in no_spike_sweeps:
            continue
        
        # Only use first 8 sweeps that have no spikes
        if sweeps_used_count >= MAX_NO_SPIKE_SWEEPS:
            break
        
        if len(group) == 0:
            continue
        
        # Get the stimulus window for THIS specific sweep
        sweep_str = str(int(sweep_number))
        sweep_windows = sweep_config.get("sweeps", {}).get(sweep_str, {}).get("windows", {})
        sweep_t_stim_min = sweep_windows.get("stimulus_start_s", t_stim_min)
        sweep_t_stim_max = sweep_windows.get("stimulus_end_s", t_stim_max)
        
        # For mixed protocol: sweep_config already contains ABSOLUTE times, use directly
        if is_mixed:
            # Times from sweep_config are already absolute for mixed protocol
            group_filtered = group[(group["t_s"] >= sweep_t_stim_min) & (group["t_s"] <= sweep_t_stim_max)]
        else:
            group_filtered = group
        
        time = group_filtered["t_s"].values
        voltage = group_filtered["value"].values
        
        if len(time) == 0:
            continue
            
        peaks, props = find_peaks(voltage, height=PEAK_HEIGHT_THRESHOLD, prominence=PEAK_PROMINENCE)
        # Enforce minimum peak distance to remove duplicate/noise detections
        filtered_peaks_list = []
        for peak_idx in peaks:
            t_peak = time[int(peak_idx)]
            if not filtered_peaks_list or (t_peak - time[int(filtered_peaks_list[-1])]) >= MIN_PEAK_DISTANCE_S:
                filtered_peaks_list.append(peak_idx)
        peaks = np.array(filtered_peaks_list)
        
        # Look for sweeps with NO spikes (low current sweeps for I-V curve)
        if len(peaks) == 0:
            # For mixed protocol: use absolute times directly from sweep_config
            if is_mixed:
                df_pA_sweep_all = df_pA_all[df_pA_all["sweep"] == sweep_number]
                if len(df_pA_sweep_all) > 0:
                    # Times from sweep_config are already absolute for mixed protocol
                    df_pA_sweep = df_pA_sweep_all[(df_pA_sweep_all["t_s"] >= sweep_t_stim_min) & 
                                                  (df_pA_sweep_all["t_s"] <= sweep_t_stim_max)]
                else:
                    df_pA_sweep = pd.DataFrame()
            else:
                df_pA_sweep = df_pA_all[df_pA_all["sweep"] == sweep_number]
            
            if len(df_pA_sweep) == 0:
                print(f"  WARNING: No pA data found for sweep {sweep_number} (mV has {len(group_filtered)} rows)")
                continue
            
            current = df_pA_sweep["value"].values 
            current_avg = np.mean(current)
            current_avg_vals.append(current_avg)

            # get all the mV values in this time range for this sweep 
            voltage_avg = np.mean(voltage)
            voltage_avg_vals.append(voltage_avg)
            valid_sweeps_found += 1
            sweeps_used_count += 1
            print(f"  Sweep {sweep_number}: I={current_avg:.2f} pA, V={voltage_avg:.4f} mV")
    
    print(f"  Found {valid_sweeps_found} sweeps with no spikes for I-V curve")

    # Make sure current and voltage have the same number of values
    rin_mohm = np.nan
    
    if len(current_avg_vals) != len(voltage_avg_vals):
        print(f"Error: Length mismatch ({len(current_avg_vals)} current vs {len(voltage_avg_vals)} voltage)")
    elif len(current_avg_vals) == 0:
        print(f"ERROR: No valid sweeps found (no sweeps without spikes)")
    else:
        current_avg_vals = np.array(current_avg_vals)
        voltage_avg_vals = np.array(voltage_avg_vals)
        slope, intercept, r_value, p_value, std_err = linregress(current_avg_vals, voltage_avg_vals)
        rin_mohm = slope  # mV/pA = MΩ (input resistance)
        print(f"Rin = {rin_mohm:.2f} MΩ (R² = {r_value**2:.3f})")

        # Plot I-V curve with best fit 
        plot_dir = bundle_path / "Input_Resistance"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.scatter(current_avg_vals, voltage_avg_vals, s=8, alpha=0.6, label="Data")

        # Best-fit line
        fit_line = intercept + (slope * current_avg_vals)
        plt.plot(current_avg_vals, fit_line, 'r', label=f'Fit: V = {slope:.2f}*I + {intercept:.2f}')

        plt.xlabel("Current (pA)")
        plt.ylabel("Voltage (mV)")
        plt.title(f"I-V curve: First 8 Sweeps")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        #plt.show()
        plt.savefig(plot_dir / 'InputResistance.jpeg')
        plt.close()

        print("AVERAGED MΩ:",rin_mohm)

    # Update manifest
    # Path to your manifest file
    manifest_path = bundle_path /'manifest.json'

    # Load the existing manifest
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Add or update the key in the "analysis" section
    manifest.setdefault("analysis", {})["input_resistance"] = rin_mohm

    # Save the updated manifest
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)











