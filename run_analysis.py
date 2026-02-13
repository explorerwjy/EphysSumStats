import json
from pathlib import Path
from typing import Optional
import pandas as pd
from analysis import resting_vm_per_sweep, attach_manifest_to_analysis
from spike_detection_new import run_spike_detection
from sav_gol_filter import run_sav_gol
from input_resistance import get_input_resistance


def checkpoint(milestone_name: str) -> bool:
    """
    Auto-yes checkpoint - no longer prompts user.
    
    Args:
        milestone_name: Description of the completed milestone
        
    Returns:
        True (always proceeds)
    """
    print("\n" + "="*70)
    print(f"✓ {milestone_name} was successful!")
    print("="*70)
    return True


def detect_hardware_malfunction(bundle_dir: str):
    """
    Detect if hardware malfunction occurred: both channels recorded as mV (empty pA).
    
    Args:
        bundle_dir: Path to the bundle
    
    Returns:
        True if malfunction detected (empty pA), False otherwise
    """
    p = Path(bundle_dir)
    man = json.loads((p / "manifest.json").read_text())
    
    try:
        df_pa = pd.read_parquet(p / man["tables"]["pa"])
        # Malfunction if pA is empty or has very few data points
        return len(df_pa) == 0 or df_pa.shape[0] < 100
    except:
        return False

def fix_hardware_malfunction_mV(bundle_dir: str):
    """
    When hardware malfunction occurs, two mV channels are recorded (correct + nonsense).
    This function identifies and keeps only the correct mV channel by checking signal stability.
    The correct channel should have consistent morphology across sweeps.
    The nonsense channel will have random/inconsistent data.
    
    Args:
        bundle_dir: Path to the bundle
    
    Returns:
        True if fix successful, False otherwise
    """
    p = Path(bundle_dir)
    man = json.loads((p / "manifest.json").read_text())
    
    try:
        mv_path = p / man["tables"]["mv"]
        df_mv = pd.read_parquet(mv_path)
        
        # Check if there are multiple channels
        if "channel_index" not in df_mv.columns:
            return False
        
        channels = df_mv["channel_index"].unique()
        if len(channels) != 2:
            return False
        
        print(f"  Detected {len(channels)} mV channels. Identifying the correct one...")
        
        # For each channel, calculate variance across sweeps
        # The correct channel should have consistent patterns (lower variance in peak detection)
        # The nonsense channel will have random data (higher variance)
        
        channel_stats = {}
        for ch in channels:
            df_ch = df_mv[df_mv["channel_index"] == ch]
            
            # Group by sweep and calculate signal statistics
            sweep_stats = df_ch.groupby("sweep")["value"].agg(["mean", "std", "min", "max", "count"])
            
            # Calculate coefficient of variation (std / mean) - indicator of signal consistency
            # Nonsense data will have very high CV
            cv_per_sweep = sweep_stats["std"] / (sweep_stats["mean"].abs() + 1e-6)
            avg_cv = cv_per_sweep.mean()
            
            channel_stats[ch] = {
                "avg_cv": avg_cv,
                "mean_std": sweep_stats["std"].mean(),
                "data_points": len(df_ch)
            }
            
            print(f"    Channel {ch}: CV={avg_cv:.4f}, Mean Std={sweep_stats['std'].mean():.4f}, Points={len(df_ch)}")
        
        # Select channel with HIGHER CV (more variable = correct channel with real signal)
        # The nonsense channel will have near-zero CV (flat noise or constant value)
        # The correct channel will have natural signal variation (higher CV)
        correct_channel = max(channel_stats.keys(), key=lambda x: channel_stats[x]["avg_cv"])
        
        print(f"  ✓ Selected Channel {correct_channel} as correct signal (highest CV)")
        
        # Keep only correct channel
        df_mv_fixed = df_mv[df_mv["channel_index"] == correct_channel].copy()
        
        # Save back
        df_mv_fixed.to_parquet(mv_path, index=False)
        print(f"  ✓ Saved corrected mV data to {mv_path}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ ERROR fixing mV data: {e}")
        return False

def is_current_data_valid(bundle_dir: str, sweep_config: Optional[dict] = None):
    """
    Check if current data exists in the expected stimulus time window.
    
    IMPORTANT: For MIXED PROTOCOL files only:
    sweep_config.json uses RELATIVE times per sweep (0-27s)
    but the parquet files use ABSOLUTE times (all sweeps concatenated, e.g., 278-1856s)
    This function converts relative times to absolute times for mixed protocol files.
    
    For SINGLE PROTOCOL files, times in sweep_config and parquet match directly.
    
    Args:
        bundle_dir: Path to the bundle
        sweep_config: Dict from sweep_config.json with stimulus windows (optional, tries to load if None)

    Returns:
        True if valid current data exists, False otherwise
    """
    p = Path(bundle_dir)
    man = json.loads((p / "manifest.json").read_text())
    df_pa = pd.read_parquet(p / man["tables"]["pa"])
    
    # Detect if mixed protocol
    is_mixed = "stimulus" in man["tables"] and "response" in man["tables"]
    
    # Determine time window from sweep_config or use first 10% of data
    if sweep_config is not None:
        try:
            first_valid_sweep_id = None
            t_min_relative = None
            t_max_relative = None
            
            for sweep_id_str, sweep_data in sweep_config.get("sweeps", {}).items():
                if sweep_data.get("valid", False):
                    first_valid_sweep_id = int(sweep_id_str)
                    t_min_relative = sweep_data["windows"].get("stimulus_start_s", 0.1)
                    t_max_relative = sweep_data["windows"].get("stimulus_end_s", 0.75)
                    break
            
            if first_valid_sweep_id is not None:
                # For mixed protocol: sweep_config contains ABSOLUTE times (from NWB file)
                # For single protocol: sweep_config contains RELATIVE times (within each sweep)
                if is_mixed:
                    # Mixed protocol: use absolute times directly from sweep_config
                    t_min = t_min_relative  # Actually absolute times, misnamed variable
                    t_max = t_max_relative  # Actually absolute times, misnamed variable
                else:
                    # Single protocol: use relative times directly
                    t_min = t_min_relative
                    t_max = t_max_relative
            else:
                # No valid sweeps found: use first 10% of data
                t_min = df_pa["t_s"].min()
                t_max = t_min + (df_pa["t_s"].max() - t_min) * 0.1
        except (KeyError, TypeError):
            # If sweep_config lookup fails, use first 10% of data
            t_min = df_pa["t_s"].min()
            t_max = t_min + (df_pa["t_s"].max() - t_min) * 0.1
    else:
        # No sweep_config: use first 10% of data for validation
        t_min = df_pa["t_s"].min()
        t_max = t_min + (df_pa["t_s"].max() - t_min) * 0.1
    
    df_pa_filtered = df_pa[(df_pa["t_s"] >= t_min) & (df_pa["t_s"] <= t_max)]
    return len(df_pa_filtered) > 0


def replace_current_data_with_reference(bundle_dir: str, reference_bundle_dir: str, sweep_config: Optional[dict] = None):
    """
    Replace the VALUES inside the faulty pA parquet file with values from a reference bundle.
    
    Crucially: The reference data sweep numbers are remapped to match the target bundle's sweep numbers,
    since both recordings use the same protocol but may have different sweep numbering.
    The TARGET FILENAME is preserved (e.g., pa_660.parquet stays pa_660.parquet).
    
    Args:
        bundle_dir: Path to the bundle with faulty current data (e.g., pa_660.parquet)
        reference_bundle_dir: Path to the reference bundle with good current data (e.g., pa_668.parquet)
    """
    p = Path(bundle_dir)
    p_ref = Path(reference_bundle_dir)
    
    # Load manifests
    man = json.loads((p / "manifest.json").read_text())
    man_ref = json.loads((p_ref / "manifest.json").read_text())
    
    # Get the pA parquet file paths
    pa_table_name = man["tables"]["pa"]  # e.g., "pa_660.parquet" (target filename to keep)
    pa_ref_table_name = man_ref["tables"]["pa"]  # e.g., "pa_668.parquet" (source)
    
    pa_ref_path = p_ref / pa_ref_table_name
    pa_path = p / pa_table_name  # Target path (keep this filename)
    
    # Load BOTH current datasets
    df_pa_faulty = pd.read_parquet(pa_path)  # Target (faulty) dataset
    df_pa_ref = pd.read_parquet(pa_ref_path)  # Source (reference) dataset
    
    # Get unique sweep numbers from each
    target_sweeps = sorted(df_pa_faulty["sweep"].unique())
    ref_sweeps = sorted(df_pa_ref["sweep"].unique())

    # If the faulty pA file has no sweeps (empty), we'll write the reference sweeps
    # into the target filename and use the reference sweep numbering.
    if len(target_sweeps) == 0:
        print("  Note: Faulty pA contains no sweeps. Will write reference sweeps into target file.")
        target_sweeps = list(ref_sweeps)

    print(f"  Target sweeps: {len(target_sweeps)} sweeps (e.g., {target_sweeps[:5]}...)")
    print(f"  Reference sweeps: {len(ref_sweeps)} sweeps (e.g., {ref_sweeps[:5]}...)")

    # Create mapping by position: map the first N reference sweeps to the first N target sweeps
    # If counts differ, map up to the smaller length and drop any unmapped reference rows.
    n_map = min(len(ref_sweeps), len(target_sweeps))
    if n_map == 0:
        raise ValueError("Reference or target pA has no sweeps to map")

    if len(ref_sweeps) != len(target_sweeps):
        print(f"  WARNING: Sweep count mismatch (ref={len(ref_sweeps)} vs target={len(target_sweeps)}). Mapping first {n_map} sweeps.")

    sweep_mapping = {ref_sweeps[i]: target_sweeps[i] for i in range(n_map)}

    # Remap reference data to target sweep numbers
    df_pa_ref_remapped = df_pa_ref.copy()
    df_pa_ref_remapped["sweep"] = df_pa_ref_remapped["sweep"].map(sweep_mapping)

    # Drop rows that could not be mapped (NaN sweep) to avoid NaN sweep ids
    before_rows = len(df_pa_ref_remapped)
    df_pa_ref_remapped = df_pa_ref_remapped.dropna(subset=["sweep"]).copy()
    after_rows = len(df_pa_ref_remapped)
    if after_rows < before_rows:
        print(f"  Note: Dropped {before_rows - after_rows} reference rows that could not be mapped to target sweeps.")

    # Ensure sweep is integer type
    df_pa_ref_remapped["sweep"] = df_pa_ref_remapped["sweep"].astype(int)
    # Preview summary and ask for confirmation before overwriting target file
    print("\n--- Preview replacement ---")
    print(f"Target (will keep filename): {pa_table_name} -> {pa_path}")
    print(f"Source (reference): {pa_ref_table_name} from {p_ref}")
    print(f"Remapped rows: {len(df_pa_ref_remapped)} (from {len(df_pa_ref)} source rows)")
    print(f"Target sweeps (post-map) sample: {sorted(df_pa_ref_remapped['sweep'].unique())[:8]}")
    print("First 5 rows of remapped reference data:")
    try:
        print(df_pa_ref_remapped.head().to_string())
    except Exception:
        print(df_pa_ref_remapped.head())

    # Apply baseline offset correction + per-sweep averaging + rounding to 5 pA increments
    try:
        import numpy as _np
        # Step 1: Calculate baseline offset during quiet period (pre-stimulus period, no injection)
        # Use sweep_config if available, otherwise use first 10% of recording
        if sweep_config:
            try:
                # Find first sweep and get its stimulus start time
                for sweep_id, sweep_data in sweep_config.get("sweeps", {}).items():
                    if sweep_data.get("valid", False):
                        t_stim_start = sweep_data["windows"].get("stimulus_start_s", 0.1)
                        break
                baseline_window = df_pa_ref_remapped[df_pa_ref_remapped['t_s'] < t_stim_start]
                print(f"Using stimulus start time from sweep_config: {t_stim_start:.6f}s")
            except (KeyError, TypeError, StopIteration):
                # Fallback to first 10% if sweep_config extraction fails
                t_max = df_pa_ref_remapped['t_s'].max()
                baseline_window = df_pa_ref_remapped[df_pa_ref_remapped['t_s'] < (t_max * 0.1)]
                print(f"Using fallback: first 10% of recording (up to {t_max * 0.1:.6f}s)")
        else:
            # No sweep_config: use first 10% of recording as baseline
            t_max = df_pa_ref_remapped['t_s'].max()
            baseline_window = df_pa_ref_remapped[df_pa_ref_remapped['t_s'] < (t_max * 0.1)]
            print(f"No sweep_config provided: using first 10% of recording (up to {t_max * 0.1:.6f}s) as baseline")
        
        baseline_offset = baseline_window['value'].mean() if len(baseline_window) > 0 else 0.0
        print(f"\nBaseline offset (pre-stimulus quiet period): {baseline_offset:.2f} pA")

        # Step 2: Subtract baseline offset from all values
        df_pa_ref_remapped['value'] = df_pa_ref_remapped['value'] - baseline_offset

        # Step 3: Compute mean current in the stimulus window per sweep (after offset correction)
        # Again, use sweep_config if available
        if sweep_config:
            try:
                t_stim_start = None
                t_stim_end = None
                for sweep_id, sweep_data in sweep_config.get("sweeps", {}).items():
                    if sweep_data.get("valid", False):
                        windows = sweep_data["windows"]
                        t_stim_start = windows.get("stimulus_start_s", 0.1)
                        t_stim_end = windows.get("stimulus_end_s", 0.75)
                        break
                if t_stim_start is not None and t_stim_end is not None:
                    df_window = df_pa_ref_remapped[(df_pa_ref_remapped['t_s'] >= t_stim_start) & (df_pa_ref_remapped['t_s'] <= t_stim_end)]
                    print(f"Using stimulus window from sweep_config: [{t_stim_start:.6f}, {t_stim_end:.6f}]s")
                else:
                    raise KeyError("Could not extract stimulus window")
            except (KeyError, TypeError):
                # Fallback to 0.1-0.75 if extraction fails
                df_window = df_pa_ref_remapped[(df_pa_ref_remapped['t_s'] >= 0.1) & (df_pa_ref_remapped['t_s'] <= 0.75)]
                print("Using fallback stimulus window: [0.1, 0.75]s")
        else:
            # No sweep_config: use middle 50% of recording
            t_min = df_pa_ref_remapped['t_s'].min()
            t_max = df_pa_ref_remapped['t_s'].max()
            t_window_min = t_min + (t_max - t_min) * 0.2
            t_window_max = t_min + (t_max - t_min) * 0.7
            df_window = df_pa_ref_remapped[(df_pa_ref_remapped['t_s'] >= t_window_min) & (df_pa_ref_remapped['t_s'] <= t_window_max)]
            print(f"No sweep_config: using middle 50% of recording [{t_window_min:.6f}, {t_window_max:.6f}]s")
        
        avg_pa = df_window.groupby('sweep')['value'].mean().reset_index(name='avg_injected_current_pA')
        # if some sweeps missing in window, fallback to full-sweep mean
        if avg_pa['sweep'].nunique() < df_pa_ref_remapped['sweep'].nunique():
            fallback = df_pa_ref_remapped.groupby('sweep')['value'].mean().reset_index(name='avg_injected_current_pA')
            avg_pa = avg_pa.set_index('sweep').combine_first(fallback.set_index('sweep')).reset_index()

        # Step 4: Round to nearest 5 pA (or 0)
        avg_pa['avg_injected_current_pA_rounded'] = (_np.round(avg_pa['avg_injected_current_pA'] / 5) * 5).astype(float)

        # Step 5: Apply rounded mean to all rows in each sweep
        for _, row in avg_pa.iterrows():
            sw = int(row['sweep'])
            rounded_val = float(row['avg_injected_current_pA_rounded'])
            df_pa_ref_remapped.loc[df_pa_ref_remapped['sweep'] == sw, 'value'] = rounded_val

        print('\nApplied baseline correction + per-sweep mean and rounded to 5 pA increments (preview):')
        print(avg_pa.head().to_string())
    except Exception as _e:
        print(f"Warning: could not apply per-sweep rounding to remapped data: {_e}")

    # Auto-yes replacement
    print("Auto-proceeding with pA replacement...")

    # Save remapped reference data to the TARGET filename, replacing the faulty values
    df_pa_ref_remapped.to_parquet(pa_path, index=False)
    print(f"✓ Replaced VALUES in {pa_table_name} (kept original filename)")
    print(f"  Source: {pa_ref_table_name} from {p_ref}")
    print(f"  Destination: {pa_path}")
    print(f"  Sweep remapping applied for {n_map} sweeps")


def load_sweep_config(bundle_dir: str):
    """
    Load sweep_config.json if it exists, otherwise return None.
    
    This configuration file is created by nwb_integration.py and contains:
    - Classification of each sweep (valid or invalid)
    - Analysis windows for each sweep (baseline, stimulus, response)
    - Stimulus levels for each sweep
    
    Args:
        bundle_dir: Path to bundle directory
    
    Returns:
        dict: sweep_config if file exists, None otherwise
    """
    p = Path(bundle_dir)
    config_path = p / "sweep_config.json"
    
    if config_path.exists():
        print(f"✓ Loading sweep_config.json from {p.name}")
        with open(config_path) as f:
            return json.load(f)
    else:
        print(f"⚠ No sweep_config.json found in {p.name}")
        print("  (Spike detection will use default window calculation)")
        return None


def run_for_bundle(bundle_dir: str, reference_bundle_dir: str = None):
    p = Path(bundle_dir)
    pA_was_replaced = False  # Track if pA data was replaced
    
    # STEP 0: Load sweep_config early so we can use it for data processing
    sweep_config = load_sweep_config(bundle_dir)
    
    # STEP 1: Check for hardware malfunction (empty pA, 2 mV channels)
    if detect_hardware_malfunction(bundle_dir):
        print(f"\n⚠ HARDWARE MALFUNCTION DETECTED in {bundle_dir}")
        print("  Both channels recorded as voltage (empty current data).")
        
        # Step 1a: Fix mV data
        print("\n>>> Fixing voltage data: extracting correct mV channel...")
        if fix_hardware_malfunction_mV(bundle_dir):
            print("  ✓ Voltage data fixed")
        else:
            print("  ✗ Failed to fix voltage data")
        
        # Step 1b: Replace pA with reference
        print("\n>>> Replacing empty current data with reference recording...")
        
        if reference_bundle_dir is None:
            print("    No reference recording provided - skipping current data replacement")
            reference_bundle_dir = ""  # Auto-skip
        
        if reference_bundle_dir:
            try:
                replace_current_data_with_reference(bundle_dir, reference_bundle_dir, sweep_config)
                print("  ✓ Current data replaced")
                pA_was_replaced = True  # Mark that pA was replaced
            except Exception as e:
                print(f"  ✗ ERROR: Failed to load reference: {e}")
                print("    Proceeding with NaN current values...")
        else:
            print("    Proceeding without current data (input resistance analysis will have NaN)...")
        print()
    
    # STEP 2: Check for invalid current data (non-malfunction case)
    elif not is_current_data_valid(bundle_dir, sweep_config):
        print(f"\n⚠ WARNING: No valid current data found in {bundle_dir}")
        print("  Current data is required for accurate input resistance analysis.")
        
        # If no reference provided, auto-skip
        if reference_bundle_dir is None:
            print("\n>>> No reference recording provided - skipping current data replacement")
            reference_bundle_dir = ""  # Auto-skip
        
        if reference_bundle_dir:
            try:
                print(f"\n>>> Replacing faulty current data with reference recording...")
                replace_current_data_with_reference(bundle_dir, reference_bundle_dir, sweep_config)
                print()
            except Exception as e:
                print(f"✗ ERROR: Failed to load reference: {e}")
                print("  Proceeding with NaN current values...")
                print()
        else:
            print("  Proceeding without current data (results will have NaN for current-based metrics)...")
            print()
    else:
        print(f"\n✓ Current data looks valid in {bundle_dir} and no malfunction detected.\n")
    
    man = json.loads((p / "manifest.json").read_text())

    # load tables
    df_mv = pd.read_parquet(p / man["tables"]["mv"])
    df_pa = pd.read_parquet(p / man["tables"]["pa"])
    
    # Filter to only kept sweeps for all analysis
    kept_sweeps = sweep_config.get("kept_sweeps", [])
    print(f"\n>>> Filtering to kept sweeps: {len(kept_sweeps)} sweeps")
    
    # Filter all dataframes to only include kept sweeps
    df_mv_kept = df_mv[df_mv["sweep"].isin(kept_sweeps)].copy()
    df_pa_kept = df_pa[df_pa["sweep"].isin(kept_sweeps)].copy()
    
    print(f"    mV data: {len(df_mv_kept)} rows (from {len(df_mv)})")
    print(f"    pA data: {len(df_pa_kept)} rows (from {len(df_pa)})")
    
    # sweep_config was already loaded at the beginning of this function
    
    df_vm_per_sweep = resting_vm_per_sweep(df_mv_kept, sweep_config, bundle_dir)  # one row per sweep, columns like resting_vm_mean_mV
    combined_mean = float(df_vm_per_sweep["resting_vm_mean_mV"].mean())

    # save analysis outputs
    out_parq = p / "analysis.parquet"
    out_csv  = p / "analysis.csv"
    df_vm_per_sweep.to_parquet(out_parq, index=False)
    df_vm_per_sweep.to_csv(out_csv, index=False)

    # update manifest with analysis pointers (non-destructive)
    man.setdefault("analysis", {})
    man["analysis"]["resting_vm_table"] = out_parq.name
    man["analysis"]["resting_vm_mean"]  = combined_mean
    (p / "manifest.json").write_text(json.dumps(man, indent=2))

    # MILESTONE CHECKPOINT 1: Resting Membrane Potential
    if not checkpoint("Resting Membrane Potential (resting_vm_per_sweep) calculated"):
        return

    # spike detection
    # CRITICAL: Reload pA from disk to pick up replaced data (if malfunction was fixed above)
    df_pa_kept = pd.read_parquet(p / man["tables"]["pa"])
    df_pa_kept = df_pa_kept[df_pa_kept["sweep"].isin(kept_sweeps)].copy()
    df_analysis = pd.read_parquet(p /"analysis.parquet")
    fs = man["meta"]["sampleRate_Hz"]
    
    run_spike_detection(df_mv_kept, df_pa_kept, df_analysis, fs, bundle_dir, 
                       pA_was_replaced=pA_was_replaced, sweep_config=sweep_config)
    print("Spike detection was successful")
    #After running above line, analysis.parquet and analysis.csv and manifest.json will be updated

    # MILESTONE CHECKPOINT 2: Spike Detection
    if not checkpoint("Spike Detection completed"):
        return

    #low pass filter
    df_analysis = pd.read_parquet(p /"analysis.parquet")
    run_sav_gol(df_mv_kept, df_analysis, fs, bundle_dir, sweep_config=sweep_config)
    print("Running Sav Gol was successful")
    
    # MILESTONE CHECKPOINT 3: Sav-Gol Filtering
    if not checkpoint("Sav-Gol Filtering (trace filtering) completed"):
        return
    #After running above line, analysis.parquet and analysis.csv and manifest.json will be updated

    #input resistance
    # CRITICAL: Reload pA from disk to pick up replaced data (if malfunction was fixed above)
    df_pa_kept = pd.read_parquet(p / man["tables"]["pa"])
    df_pa_kept = df_pa_kept[df_pa_kept["sweep"].isin(kept_sweeps)].copy()
    get_input_resistance(df_mv_kept, df_pa_kept, bundle_dir, sweep_config=sweep_config)
    print("Getting input resistance was successful")
    #After running above line, manifest.json will be updated

    # MILESTONE CHECKPOINT 4: Input Resistance
    if not checkpoint("Input Resistance calculation completed"):
        return

    #attach manifest details to analysis results
    df_analysis = pd.read_parquet(p /"analysis.parquet")
    attach_manifest_to_analysis(bundle_dir, df_analysis)
    print("Adding to analysis was successful")
    print(f"All updates completed and successful for {bundle_dir}.")


