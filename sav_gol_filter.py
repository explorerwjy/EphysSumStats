# This file runs a low pass filter through the data and obtains some metrics
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import savgol_filter, find_peaks
from analysis import resting_vm_per_sweep
from analysis_config import (
    get_smoothing_proportion,
    SAV_GOL_POLY_ORDER,
    PEAK_HEIGHT_THRESHOLD,
    PEAK_PROMINENCE,
    MIN_PEAK_DISTANCE_MS,
    MIN_PEAK_DISTANCE_S,
    BASELINE_WINDOW_MS,
    BASELINE_WINDOW_S,
)

def run_sav_gol(df, df_analysis, fs, bundle_path, sweep_config=None):
    print("RUNNING SAV GOL FILTER")
    print("NOTE: Sav-Gol filter runs on baseline periods (no stimulus) in every sweep")
    
    # Detect protocol type (mixed vs single)
    bundle_path = Path(bundle_path)
    man = json.loads((bundle_path / "manifest.json").read_text())
    is_mixed = "stimulus" in man.get("tables", {}) and "response" in man.get("tables", {})
    print(f"Protocol type: {'MIXED' if is_mixed else 'SINGLE'}")
    
    # Handle sampling rate for mixed vs single protocol
    # For mixed protocol: fs might be a list like ['200000.0', '50000.0']
    # For single protocol: fs is a single number
    sweep_rates = {}  # Dictionary to store per-sweep sampling rates
    
    if isinstance(fs, list):
        # Mixed protocol: load per-sweep sampling rates from manifest
        print(f"  ⚠ Multiple sampling rates detected: {fs}")
        
        if is_mixed:
            # Load per-sweep rates from manifest protocols
            protocols = man.get("protocols", {})
            for sweep_id, protocol_data in protocols.items():
                sweep_rates[int(sweep_id)] = float(protocol_data.get("rate", max([float(f) for f in fs])))
            print(f"  ✓ Loaded per-sweep sampling rates for {len(sweep_rates)} sweeps")
        
        # Use maximum rate as default (fallback if sweep not in protocols)
        fs_default = max([float(f) for f in fs])
    else:
        # Single protocol: use the single rate
        fs_default = float(fs)
    
    #Must be odd - The filter needs a "center point" in the window to align the fitted curve with the sign
    #1001 - 50 ms at 20 kHz
    #201 - 10 ms at 20 kHz  
    #this cannot exceed 40000 samples or 2000 ms given our time range
    
    # Define smoothing proportion from config
    smoothing_proportion = get_smoothing_proportion()
    
    #How curvy the fit is inside the window
    poly_order = SAV_GOL_POLY_ORDER
    
    # STEP 1: Extract baseline windows from sweep_config for each sweep
    baseline_windows = {}  # sweep_id -> (baseline_start_s, baseline_end_s)
    
    if sweep_config is None:
        print("ERROR: sweep_config required to identify baseline periods")
        return
    
    try:
        # Get unique sweeps in the input dataframe
        available_sweeps = set(df["sweep"].unique())
        
        # Track sweeps with no stimulus (will be excluded from baseline analysis)
        no_stimulus_sweeps = []
        
        for sweep_id, sweep_data in sweep_config.get("sweeps", {}).items():
            sweep_id_int = int(sweep_id)
            # Only include sweeps that are both valid AND present in the dataframe
            if sweep_data.get("valid", False) and sweep_id_int in available_sweeps:
                # Check if this sweep has actual stimulus (not 0 pA)
                stimulus_level = sweep_data.get("stimulus_level_pA", None)
                if stimulus_level is not None and stimulus_level == 0:
                    no_stimulus_sweeps.append(sweep_id_int)
                    continue  # Skip sweeps with no stimulus
                
                windows = sweep_data.get("windows", {})
                baseline_start = windows.get("baseline_start_s")
                baseline_end = windows.get("baseline_end_s")
                if baseline_start is not None and baseline_end is not None:
                    baseline_windows[sweep_id_int] = (baseline_start, baseline_end)
        
        # Report sweeps with no stimulus (excluded from analysis)
        if no_stimulus_sweeps:
            print(f"\n⚠️  Excluding {len(no_stimulus_sweeps)} sweep(s) with NO STIMULUS (0 pA):")
            for sweep_id in sorted(no_stimulus_sweeps):
                print(f"   Sweep {sweep_id}: stimulus_level_pA = 0 (entire sweep is baseline, no pre-stimulus period)")
            print(f"   → These sweeps cannot be used for baseline filtering\n")
        
        print(f"Found baseline windows for {len(baseline_windows)} sweeps (out of {len(available_sweeps)} sweeps in data)")
        if len(baseline_windows) == 0:
            print("ERROR: No baseline windows found in sweep_config")
            return
        
        # Report on sweeps excluded from baseline analysis
        excluded_sweeps = available_sweeps - set(baseline_windows.keys())
        if excluded_sweeps:
            print(f"\nℹ️  {len(excluded_sweeps)} sweep(s) excluded from baseline analysis:")
            for sweep_id in sorted(excluded_sweeps):
                sweep_data = sweep_config.get("sweeps", {}).get(str(sweep_id), {})
                if not sweep_data.get("valid", False):
                    print(f"   Sweep {sweep_id}: marked as invalid in sweep_config")
                elif "windows" not in sweep_data or sweep_data["windows"].get("baseline_start_s") is None:
                    print(f"   Sweep {sweep_id}: missing baseline timing in sweep_config")
                else:
                    print(f"   Sweep {sweep_id}: unknown reason")
            print(f"   → These sweeps will have NaN for filtered_RMP_derivative\n")
    except (KeyError, TypeError) as e:
        print(f"ERROR: Failed to extract baseline windows from sweep_config: {e}")
        return
    
    # STEP 2: Extract baseline data for all sweeps and calculate time range
    # For MIXED protocol: convert relative times in sweep_config to absolute times in parquet
    # For SINGLE protocol: use relative times directly
    df_baseline_all = []
    debug_first_sweep = True
    for sweep_id, (baseline_start_rel, baseline_end_rel) in baseline_windows.items():
        # Get this sweep's data
        df_sweep_all = df[df["sweep"] == sweep_id]
        
        if len(df_sweep_all) == 0:
            continue
        
        if is_mixed:
            # For mixed protocol: sweep_config already contains ABSOLUTE times
            # Parquet also has ABSOLUTE times - use them directly
            baseline_start_abs = baseline_start_rel  # Actually absolute, misnamed
            baseline_end_abs = baseline_end_rel      # Actually absolute, misnamed
            
            # Debug first sweep
            if debug_first_sweep:
                print(f"\n[DEBUG] First sweep {sweep_id}:")
                print(f"  Baseline window from sweep_config: [{baseline_start_abs:.6f}, {baseline_end_abs:.6f}] s")
                print(f"  Parquet time range: [{df_sweep_all['t_s'].min():.6f}, {df_sweep_all['t_s'].max():.6f}] s")
                print(f"  Total samples in sweep: {len(df_sweep_all)}")
                debug_first_sweep = False
            
            df_sweep_baseline = df_sweep_all[(df_sweep_all["t_s"] >= baseline_start_abs) & 
                                             (df_sweep_all["t_s"] <= baseline_end_abs)]
        else:
            # For single protocol: use relative times directly
            df_sweep_baseline = df_sweep_all[(df_sweep_all["t_s"] >= baseline_start_rel) & 
                                             (df_sweep_all["t_s"] <= baseline_end_rel)]
        
        if debug_first_sweep is False and sweep_id == 4:
            print(f"  Baseline samples extracted: {len(df_sweep_baseline)}")
        
        df_baseline_all.append(df_sweep_baseline)
    
    df_baseline_combined = pd.concat(df_baseline_all, ignore_index=True)
    
    if len(df_baseline_combined) == 0:
        print("ERROR: No data found in any baseline windows")
        return
    
    # Get the actual baseline time range (should be consistent across sweeps)
    time_window_start = df_baseline_combined["t_s"].min()
    time_window_end = df_baseline_combined["t_s"].max()
    time_window_range_s = time_window_end - time_window_start
    
    print(f"Processing baseline periods over time range [{time_window_start:.6f}, {time_window_end:.6f}] s")
    print(f"Time window range: {time_window_range_s:.6f} s")
    
    # STEP 3: Calculate adaptive smoothing window duration (in time, not samples)
    # This will be converted to samples per-sweep based on each sweep's sampling rate
    desired_smooth_ms = smoothing_proportion * time_window_range_s * 1000
    print(f"Adaptive smoothing window: {desired_smooth_ms:.2f} ms")
    
    # === DEBUG: Key values to compare between machines ===
    print("\n" + "="*60)
    print("DEBUG: COMPARE THESE VALUES BETWEEN MACHINES")
    print("="*60)
    print(f"  smoothing_proportion: {smoothing_proportion}")
    print(f"  time_window_range_s: {time_window_range_s:.10f}")
    print(f"  desired_smooth_ms: {desired_smooth_ms:.6f}")
    print(f"  fs_default: {fs_default}")
    print(f"  poly_order: {poly_order}")
    print(f"  Valid sweeps in baseline_windows: {sorted(baseline_windows.keys())}")
    print(f"  Total valid sweeps: {len(baseline_windows)}")
    print(f"  sweep_rates: {sweep_rates}")
    print("="*60 + "\n")

    freq_before = []
    freq_after = []
    df_sav_gol = []
    
    # STEP 4: Process baseline period for each sweep
    for sweep_id, (baseline_start_rel, baseline_end_rel) in baseline_windows.items():
        print(f"\nSWEEP {sweep_id}:")
        
        # Get sweep-specific sampling rate
        if sweep_rates and sweep_id in sweep_rates:
            sweep_fs = sweep_rates[sweep_id]
            print(f"  Using sweep-specific rate: {sweep_fs} Hz")
        elif is_mixed:
            # For mixed protocol, sweep-specific rate is REQUIRED
            raise ValueError(f"Mixed protocol detected but no sampling rate found for sweep {sweep_id}. "
                           f"sweep_rates keys: {list(sweep_rates.keys()) if sweep_rates else 'None'}")
        else:
            # Single protocol: use the single sampling rate
            sweep_fs = fs_default
        
        # Calculate window length for this sweep based on its sampling rate
        window_length = int((desired_smooth_ms / 1000) * sweep_fs)
        
        # Ensure window_length is odd (required for Savitzky-Golay)
        if window_length % 2 == 0:
            window_length += 1
        
        print(f"  Window length: {window_length} samples ({(window_length / sweep_fs) * 1000:.2f} ms)")
        
        # Get this sweep's data
        df_sweep = df[df["sweep"] == sweep_id]
        
        # For MIXED protocol: sweep_config already has ABSOLUTE times
        # For SINGLE protocol: use relative times directly
        if is_mixed:
            if len(df_sweep) == 0:
                print(f"  No data for sweep {sweep_id}")
                continue
            # For mixed protocol: times in sweep_config are already absolute
            baseline_start_abs = baseline_start_rel  # Actually absolute, misnamed variable
            baseline_end_abs = baseline_end_rel      # Actually absolute, misnamed variable
            group = df_sweep[(df_sweep["t_s"] >= baseline_start_abs) & (df_sweep["t_s"] <= baseline_end_abs)]
        else:
            baseline_start_abs = baseline_start_rel
            baseline_end_abs = baseline_end_rel
            group = df_sweep[(df_sweep["t_s"] >= baseline_start_rel) & (df_sweep["t_s"] <= baseline_end_rel)]
        
        if len(group) == 0:
            print(f"  No baseline data for sweep {sweep_id}")
            continue
        
        print(f"  Baseline period: [{baseline_start_abs:.6f}, {baseline_end_abs:.6f}] s ({len(group)} samples)")
        time_absolute = group["t_s"].values  # Keep absolute times for output
        time = time_absolute.copy()  # For plotting (may be converted to relative)
        voltage = group["value"].values
        baseline_duration = baseline_end_abs - baseline_start_abs
        
        # For MIXED protocol: convert to relative times for plotting only
        # This makes the plot align with sweep_config markers which are in relative time
        if is_mixed and len(time) > 0:
            time_sweep_start = time[0]
            time = time - time_sweep_start  # Convert to relative time for plotting
        
        # Adjust window_length if it's larger than the data (Savitzky-Golay requirement)
        actual_window_length = window_length
        if window_length > len(voltage):
            # Use a smaller odd window (50% of data, at most)
            actual_window_length = int(len(voltage) * 0.5)
            if actual_window_length % 2 == 0:
                actual_window_length -= 1  # Make it odd
            actual_window_length = max(5, actual_window_length)  # Minimum 5 samples
            print(f"  Window too large ({window_length} > {len(voltage)}), using {actual_window_length}")
        
        peaks, props = find_peaks(voltage, height=PEAK_HEIGHT_THRESHOLD, prominence=PEAK_PROMINENCE)
        # Enforce minimum peak distance to remove duplicate noise detections
        filtered_peaks_list = []
        for peak_idx in peaks:
            t_peak = time[int(peak_idx)]
            if not filtered_peaks_list or (t_peak - time[int(filtered_peaks_list[-1])]) >= MIN_PEAK_DISTANCE_S:
                filtered_peaks_list.append(peak_idx)
        peaks = np.array(filtered_peaks_list)
        print(f"  Before filter: {len(peaks)} peaks")
        #get frequency here
        freq_before.append(len(peaks) / baseline_duration)
        #run filter here
        voltage_filtered = savgol_filter(voltage, actual_window_length, poly_order)
        peaks2, props = find_peaks(voltage_filtered, height=PEAK_HEIGHT_THRESHOLD, prominence=PEAK_PROMINENCE)
        # Enforce minimum peak distance on filtered signal as well
        filtered_peaks2 = []
        for peak_idx in peaks2:
            t_peak = time[int(peak_idx)]
            if not filtered_peaks2 or (t_peak - time[int(filtered_peaks2[-1])]) >= MIN_PEAK_DISTANCE_S:
                filtered_peaks2.append(peak_idx)
        peaks2 = np.array(filtered_peaks2)
        #get frequency again here
        freq_after.append(len(peaks2) / baseline_duration)
        print(f"  After filter: {len(peaks2)} peaks")

        # Plot the sweep before/after filter
        plot_dir = bundle_path / "Sav_Gol_Plots_Per_Sweep"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(12, 4))  # Wider figure for better visibility
        plt.plot(time, voltage, label="Raw", alpha=0.5, linewidth=1)
        plt.plot(time, voltage_filtered, label="SavGol filtered", linewidth=2)
        
        # No stimulus markers needed for baseline plots - they clutter the view
        # The entire plot IS the baseline period
        
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (mV)")
        plt.title(f"Sweep {sweep_id} - Baseline Period ({baseline_duration:.4f}s)")
        plt.legend()
        plt.grid(True, alpha=0.3)  # Add grid for easier reading
        plt.tight_layout()
        #plt.show()
        plt.savefig(plot_dir / f'SavGol_Sweep{sweep_id}_baseline.png', dpi=300)
        plt.close()

        sweep_df = pd.DataFrame({
            "sweep": sweep_id,
            "t_s": time_absolute,  # Use absolute times for output
            "value": voltage_filtered
        })
        df_sav_gol.append(sweep_df)


    # Re-calculate resting membrane potential
    df_sav_gol = pd.concat(df_sav_gol, ignore_index=True)
    
    # Ensure sweep column is integer (concatenation may convert to float)
    df_sav_gol["sweep"] = df_sav_gol["sweep"].astype(int)
    
    print("Post-Filtered Values:",df_sav_gol)
    
    # Validate that we have data after filtering
    if len(df_sav_gol) == 0:
        print("ERROR: No filtered data generated")
        raise ValueError("Failed to generate filtered data")

    # Metric 1: Histogram of filtered voltages
    # Downsample to bins for drift analysis
    # Using 25ms windows instead of 50ms to accommodate shorter baseline periods
    # (Many protocols have 50-75ms baselines, need ≥2 windows for derivative)
    window_ms = BASELINE_WINDOW_MS
    window_s  = BASELINE_WINDOW_S

    def downsample_sweep(group):
        v = group["value"].to_numpy()
        
        # Get sweep ID to determine its sampling rate
        sweep_id = group["sweep"].iloc[0]
        
        # Get sweep-specific sampling rate
        if sweep_rates and sweep_id in sweep_rates:
            sweep_fs = sweep_rates[sweep_id]
        elif is_mixed:
            # For mixed protocol, sweep-specific rate is REQUIRED
            raise ValueError(f"Mixed protocol detected but no sampling rate found for sweep {sweep_id} in downsample. "
                           f"sweep_rates keys: {list(sweep_rates.keys()) if sweep_rates else 'None'}")
        else:
            # Single protocol: use the single sampling rate
            sweep_fs = fs_default
        
        # Calculate samples per window for THIS sweep's sampling rate
        samples_per_window = int(sweep_fs * window_s)
        
        # Skip if insufficient data for even one window
        if len(v) < samples_per_window:
            print(f"  WARNING: Sweep {sweep_id} - Insufficient baseline data for derivative calculation")
            print(f"           Has {len(v)} samples, needs {samples_per_window} for one {window_ms}ms window")
            print(f"           → filtered_RMP_derivative will be NaN (reason: baseline < {window_ms}ms)")
            return pd.DataFrame({"sweep": [sweep_id], "window": [0], "value_50ms_mean": [np.nan]})
        
        # Warn if only one window (derivative needs ≥2 windows)
        n_windows = len(v) // samples_per_window
        if n_windows < 2:
            print(f"  WARNING: Sweep {sweep_id} - Only {n_windows} window(s) available")
            print(f"           Baseline duration: {len(v)/sweep_fs*1000:.1f}ms, need ≥{2*window_ms}ms for derivative")
            print(f"           → filtered_RMP_derivative will be NaN (reason: only 1 data point)")
        
        # Trim off the last x samples so it divides evenly into time bins
        n = (len(v) // samples_per_window) * samples_per_window
        v = v[:n]
        
        # Reshape into time chunks (window_ms each)
        temp = v.reshape(-1, samples_per_window)
        window_means = temp.mean(axis=1)
        
        # Return long-form data (one row per window)
        return pd.DataFrame({
            "sweep": sweep_id,
            "window": np.arange(len(window_means)),
            "value_50ms_mean": window_means
        })

    # Create long-form dataframe directly (no need for wide format)
    long_df = (
        df_sav_gol.groupby("sweep", group_keys=False)
        .apply(downsample_sweep)
        .reset_index(drop=True)
    )
    
    print(f"Downsampled to {len(long_df)} total 50ms windows across {long_df['sweep'].nunique()} sweeps")

    # Metric 1: Get spread of RMPs from all windows across all sweeps
    Vm_all = long_df["value_50ms_mean"].to_numpy()
    
    # Remove NaN values
    Vm_all = Vm_all[~np.isnan(Vm_all)]
    
    # Check for empty or all-NaN data
    if len(Vm_all) == 0:
        print("ERROR: No valid voltage data to analyze (all NaN or empty)")
        raise ValueError("No valid voltage data in downsampled dataframe")
    
    plt.figure(figsize=(8,6))
    vmin, vmax = Vm_all.min(), Vm_all.max()
    bin_step = 0.25 # fixed 0.25mV bins
    
    print(f"RMP range: [{vmin:.4f}, {vmax:.4f}] mV")
    
    # Ensure vmax > vmin to avoid numpy.arange error
    if vmax <= vmin:
        vmax = vmin + bin_step
    
    # Create histogram with safe binning using fixed number of bins
    n_bins = max(10, int(np.ceil((vmax - vmin) / bin_step)))
    plt.hist(Vm_all, bins=n_bins, color="skyblue", edgecolor='k')
    plt.xlabel("Resting Membrane Potential (mV)")
    plt.ylabel("Frequency")
    plt.title(f"Resting Membrane Potential Distribution ({window_ms} ms bins across sweeps)")
    plt.tight_layout()
    #plt.show()
    plt.savefig(bundle_path /'RMP_Dist_Post_Filter.png', dpi=300)


    # Metric 2: drift range
    drift_range = vmax - vmin

    # Metric 3: drift standard deviation
    d_std = np.std(Vm_all)

    # Metric 4: normalized measure of signal variability
    # for every sweep, get the difference between each mean RMP
    long_df["diff"] = (
        long_df
        .groupby("sweep")["value_50ms_mean"]
        .diff()
    )

    # take the absolute value of the difference values
    long_df["abs_diff"] = long_df["diff"].abs()

    # for s in [1, 2]:
    #     print(long_df[long_df["sweep"] == s].head(3))

    # Calculate time duration per sweep based on number of time windows
    # Each window is window_ms (e.g., 25ms = 0.025s), so total duration = num_windows * window_s
    sweep_durations = (
        long_df
        .groupby("sweep")
        .size()
        .mul(window_s)  # each window is window_s seconds
    )
    
    # Add up all abs_diff values per sweep, divide by that sweep's actual duration
    per_sweep_metric = (
        long_df
        .groupby("sweep")["abs_diff"]
        .sum()
        .div(sweep_durations) 
    )

    # after doing this for every sweep, average across all of them
    mean_metric = per_sweep_metric.mean()

    def safe_gradient(x, dt):
        arr = x.to_numpy()
        if len(arr) < 2:
            return np.full_like(arr, np.nan, dtype=float)
        return np.gradient(arr, dt)

    # Metric 5: Derivative of filtered RMPs
    # Calculated over time bins (window_ms). Change in time = window_s
    dt = window_s  # Time step between bins (e.g., 0.025s for 25ms windows)
    long_df["dVdt"] = (
        long_df
        .groupby("sweep")["value_50ms_mean"]
        .transform(lambda x: safe_gradient(x, dt))
    )

    # Take the average dv/dt for each sweep
    dvdt_means = (
        long_df.groupby("sweep")["dVdt"]
        .mean()
        .reset_index(name="filtered_RMP_derivative")
    )
    
    # Report on derivative calculation results
    print("\n--- Derivative Calculation Summary ---")
    total_sweeps = len(baseline_windows)
    calculated_sweeps = dvdt_means["filtered_RMP_derivative"].notna().sum()
    nan_sweeps = dvdt_means["filtered_RMP_derivative"].isna().sum()
    
    print(f"Total sweeps with baseline windows: {total_sweeps}")
    print(f"Sweeps with valid derivative: {calculated_sweeps}")
    print(f"Sweeps with NaN derivative: {nan_sweeps}")
    
    if nan_sweeps > 0:
        nan_sweep_ids = dvdt_means[dvdt_means["filtered_RMP_derivative"].isna()]["sweep"].tolist()
        print(f"Sweeps with NaN: {nan_sweep_ids}")
        print("Common reasons:")
        print(f"  - Baseline duration < {window_ms}ms (need ≥2 windows for derivative)")
        print("  - Insufficient samples at sweep's sampling rate")
    print("---------------------------------------\n")


    #_________________________________________________________________________
    # Convert sweep_config to use integer keys for compatibility with integer sweep column
    sweep_config_int = {"sweeps": {int(k): v for k, v in sweep_config.get("sweeps", {}).items()}}
    
    df_vm_per_sweep = resting_vm_per_sweep(df_sav_gol, sweep_config=sweep_config_int, bundle_dir=str(bundle_path))
    df_vm_per_sweep.rename(columns={"resting_vm_mean_mV": "filtered_resting_vm_mean_mV"}, inplace=True)
    
    # Ensure we have the right number of rows
    if len(df_vm_per_sweep) != len(freq_before):
        print(f"WARNING: Frequency lists have {len(freq_before)} sweeps, but df_vm_per_sweep has {len(df_vm_per_sweep)} rows")
        print(f"  df_vm_per_sweep shape: {df_vm_per_sweep.shape}")
        print(f"  df_sav_gol sweeps: {df_sav_gol['sweep'].unique()}")
        print(f"  freq_before: {len(freq_before)} entries")
        # Truncate to match
        min_len = min(len(freq_before), len(df_vm_per_sweep))
        freq_before = freq_before[:min_len]
        freq_after = freq_after[:min_len]
        df_vm_per_sweep = df_vm_per_sweep.iloc[:min_len]
    
    df_vm_per_sweep["frequency_pre_filter"] = freq_before
    df_vm_per_sweep["frequency_post_filter"] = freq_after

    #Add all the metrics from above to csv
    df_vm_per_sweep["drift_range"] = drift_range
    df_vm_per_sweep["d_std"] = d_std
    df_vm_per_sweep["normalized_signal_variability"] = mean_metric
    df_vm_per_sweep = df_vm_per_sweep.merge(dvdt_means, on="sweep", how="left")


    # Add this to the csv
    updated_analysis = df_analysis.merge(df_vm_per_sweep, on="sweep", how="left")
    
    # Sort by avg_injected_current_pA (ascending) before saving
    if "avg_injected_current_pA" in updated_analysis.columns:
        updated_analysis = updated_analysis.sort_values(by="avg_injected_current_pA", ascending=True).reset_index(drop=True)
    
    updated_analysis.to_parquet(bundle_path / "analysis.parquet", index=False)
    updated_analysis.to_csv(bundle_path / "analysis.csv", index=False)

    # Get averaged across all sweeps, add to manifest 
    avg_rmp = df_vm_per_sweep["filtered_resting_vm_mean_mV"].mean()
    print(f"Average RMP: {avg_rmp:.2f} mV")

    # Path to your manifest file
    manifest_path = bundle_path / 'manifest.json'

    # Load your manifest.json
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    # Add or update the key in the "analysis" section
    manifest.setdefault("analysis", {})["filtered_grand_average_resting_vm_mean"] = float(avg_rmp)

    # Save it back
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
