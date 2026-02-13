
"""
Complete integration: NWB sweep analysis + spike detection
This script combines sweep classification with automatic analysis window calculation.
"""

import json

from pynwb import NWBHDF5IO
import numpy as np
import pandas as pd
import warnings
import sys
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from spike_detection_new import run_spike_detection
from analysis_config import (
    BASELINE_THRESHOLD_PA,
    STIMULUS_THRESHOLD_PA,
    MIN_STIMULUS_DURATION_S,
    MIN_FLAT_RATIO,
    RESPONSE_PADDING_S,
    BASELINE_FALLBACK_S,
    SECOND_DERIV_THRESHOLD,
    VOLTAGE_JUMP_THRESHOLD,
)

warnings.filterwarnings('ignore', message='.*cached namespace.*')

# Legacy commented-out function uses old thresholds
# Active code uses thresholds defined in CONFIGURATION section below (lines ~419-424)


def classify_sweeps_from_nwb(nwbfile, min_step_s=0.300, min_flat_ratio=0.7):
    """
    Classify sweeps as "kept" (suitable for analysis) or "dropped".
    
    Returns:
        dict with 'kept' (list of sweep indices) and 'dropped' (list of sweep indices)
    """
    stim_keys = list(nwbfile.stimulus.keys())
    
    kept = []
    dropped = []
    
    for idx, stim_key in enumerate(stim_keys):
        stim_series = nwbfile.stimulus[stim_key]
        sr = float(stim_series.rate)
        stimulus = np.array(stim_series.data)
        
        # Find stimulus segments
        # Smart baseline detection:
        # 1. If 0 is present in the data, use it (standard baseline for electrophysiology)
        # 2. Otherwise, use the most common value (mode)
        if 0.0 in stimulus:
            baseline = 0.0
        else:
            stimulus_rounded = np.round(stimulus, 1)
            unique_vals, counts = np.unique(stimulus_rounded, return_counts=True)
            baseline = unique_vals[np.argmax(counts)]
        is_active = np.abs(stimulus - baseline) > (0.05 * (np.max(stimulus) - np.min(stimulus)) + 1e-12)
        
        edges = np.diff(is_active.astype(int), prepend=0, append=0)
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        
        widths_s = (ends - starts) / sr
        long_segments = [i for i, w in enumerate(widths_s) if w >= min_step_s]
        
        if len(widths_s) == 0:
            # Flat (no stimulus) - keep it
            kept.append(idx)
        elif len(long_segments) >= 1:
            # Has at least one long step - check if it's square-shaped
            found_square = False
            for seg_idx in long_segments:
                segment_data = stimulus[starts[seg_idx]:ends[seg_idx]]
                peak = np.max(np.abs(segment_data))  # Use absolute value for negative stimuli
                flat_ratio = np.sum(np.abs(segment_data) > 0.9 * peak) / len(segment_data)
                
                if flat_ratio >= min_flat_ratio:
                    kept.append(idx)
                    found_square = True
                    break
            
            if not found_square:
                dropped.append(idx)
        else:
            dropped.append(idx)
    
    return {'kept': kept, 'dropped': dropped}


# def get_analysis_windows_for_sweep(nwbfile, sweep_idx):
#     """
#     Get calculated analysis windows for a specific sweep.
#     Uses stimulus bounds from that sweep to calculate windows organically.
#     """
#     stim_key = list(nwbfile.stimulus.keys())[sweep_idx]
#     stim_series = nwbfile.stimulus[stim_key]
#     sr = float(stim_series.rate)
#     stimulus = np.array(stim_series.data)
    
#     # Calculate analysis windows based on stimulus bounds
#     windows = calculate_analysis_windows(
#         stimulus, 
#         sr,
#         pre_ms=4.5,
#         post_ms=20.0,
#         padding_factor=0.05
#     )
    
#     return windows


# def extract_sweep_data(nwbfile, sweep_idx):
#     """
#     Extract stimulus and response data for a specific sweep.
#     """
#     stim_keys = list(nwbfile.stimulus.keys())
#     acq_keys = list(nwbfile.acquisition.keys())
    
#     stim_key = stim_keys[sweep_idx]
#     acq_key = acq_keys[sweep_idx]
    
#     stim_series = nwbfile.stimulus[stim_key]
#     resp_series = nwbfile.acquisition[acq_key]
    
#     sr = float(stim_series.rate)
#     stimulus = np.array(stim_series.data)
#     response = np.array(resp_series.data)
#     time = np.arange(len(stimulus)) / sr
    
#     return {
#         'time': time,
#         'stimulus': stimulus,
#         'response': response,
#         'sampling_rate': sr,
#         'sweep_idx': sweep_idx,
#         'stim_key': stim_key,
#         'acq_key': acq_key
#     }


# def print_integration_summary(nwbfile, sweep_classification):
#     """Print a summary of the sweep classification and integration."""
#     print("\n" + "="*70)
#     print("SWEEP CLASSIFICATION & ANALYSIS WINDOW EXTRACTION")
#     print("="*70)
    
#     kept = sweep_classification['kept']
#     dropped = sweep_classification['dropped']
    
#     print(f"\nTotal sweeps: {len(kept) + len(dropped)}")
#     print(f"Kept sweeps: {len(kept)}")
#     print(f"  Indices: {kept}")
#     print(f"\nDropped sweeps: {len(dropped)}")
    
#     # Show details for first few kept sweeps
#     print(f"\n" + "="*70)
#     print("ANALYSIS WINDOWS FOR FIRST 5 KEPT SWEEPS")
#     print("="*70)
    
#     for sweep_idx in kept[:5]:
#         windows = get_analysis_windows_for_sweep(nwbfile, sweep_idx)
        
#         if windows:
#             print(f"\nSweep {sweep_idx}:")
#             print(f"  Stimulus period: [{windows['t_stim_start']:.6f}, {windows['t_stim_end']:.6f}] s")
#             print(f"  Analysis window: [{windows['t_min']:.6f}, {windows['t_max']:.6f}] s")
#             print(f"  Extended window: [{windows['t_start']:.6f}, {windows['t_end']:.6f}] s")
#         else:
#             print(f"\nSweep {sweep_idx}: Could not calculate analysis windows")


# def nwb_to_dataframes_for_spike_detection(nwbfile, kept_sweep_indices):
#     """
#     Convert NWB data to DataFrame format required by spike_detection_new.py
    
#     Args:
#         nwbfile: NWB file object
#         kept_sweep_indices: List of sweep indices to include
    
#     Returns:
#         tuple: (df, df_pA, df_analysis, fs)
#             - df: DataFrame with voltage data (columns: sweep, t_s, value)
#             - df_pA: DataFrame with stimulus data (columns: sweep, t_s, value)
#             - df_analysis: DataFrame with sweep metadata
#             - fs: Sampling rate in Hz
#     """
#     stim_keys = list(nwbfile.stimulus.keys())
#     acq_keys = list(nwbfile.acquisition.keys())
    
#     # Get sampling rate from first sweep
#     first_stim = nwbfile.stimulus[stim_keys[0]]
#     fs = float(first_stim.rate)
    
#     # Collect voltage and stimulus data
#     voltage_records = []
#     stimulus_records = []
#     sweep_metadata = []
    
#     for idx in kept_sweep_indices:
#         stim_series = nwbfile.stimulus[stim_keys[idx]]
#         resp_series = nwbfile.acquisition[acq_keys[idx]]
        
#         sr = float(stim_series.rate)
#         stimulus = np.array(stim_series.data)
#         response = np.array(resp_series.data)
#         time = np.arange(len(stimulus)) / sr
        
#         # Create voltage records
#         for t, v in zip(time, response):
#             voltage_records.append({
#                 'sweep': idx,
#                 't_s': t,
#                 'value': v
#             })
        
#         # Create stimulus records
#         for t, s in zip(time, stimulus):
#             stimulus_records.append({
#                 'sweep': idx,
#                 't_s': t,
#                 'value': s
#             })
        
#         # Metadata for this sweep
#         sweep_metadata.append({
#             'sweep': idx,
#             'stim_key': stim_keys[idx],
#             'acq_key': acq_keys[idx],
#             'duration_s': float(len(stimulus) / sr)
#         })
    
#     # Create DataFrames
#     df = pd.DataFrame(voltage_records)
#     df_pA = pd.DataFrame(stimulus_records)
#     df_analysis = pd.DataFrame(sweep_metadata)
    
#     return df, df_pA, df_analysis, fs


# def run_spike_detection_on_kept_sweeps(nwb_file_path, output_bundle_path=None, kept_only=True):
#     """
#     Complete pipeline: Load NWB → classify sweeps → run spike detection with automatic windows
    
#     Args:
#         nwb_file_path: Path to NWB file
#         output_bundle_path: Path to save spike detection results (optional)
#         kept_only: If True, only process kept sweeps; if False, process all sweeps
    
#     Returns:
#         dict with spike detection results
#     """
    
#     print("\n" + "="*70)
#     print("SPIKE DETECTION WITH AUTOMATIC ANALYSIS WINDOWS")
#     print("="*70)
    
#     # Read and classify
#     with NWBHDF5IO(nwb_file_path, mode='r') as io:
#         nwbfile = io.read()
        
#         # Classify sweeps
#         print("\nStep 1: Classifying sweeps...")
#         classification = classify_sweeps_from_nwb(nwbfile, MIN_STEP_S, MIN_FLAT_RATIO)
#         kept_sweeps = classification['kept']
#         dropped_sweeps = classification['dropped']
        
#         print(f"  Found {len(kept_sweeps)} kept sweeps: {kept_sweeps}")
#         print(f"  Found {len(dropped_sweeps)} dropped sweeps")
        
#         # Select which sweeps to analyze
#         sweeps_to_analyze = kept_sweeps if kept_only else (kept_sweeps + dropped_sweeps)
        
#         # Convert to DataFrames
#         print(f"\nStep 2: Converting NWB data to DataFrames for {len(sweeps_to_analyze)} sweeps...")
#         df, df_pA, df_analysis, fs = nwb_to_dataframes_for_spike_detection(
#             nwbfile, 
#             sweeps_to_analyze
#         )
#         print(f"  Voltage data: {len(df)} samples")
#         print(f"  Stimulus data: {len(df_pA)} samples")
#         print(f"  Sampling rate: {fs} Hz")
        
#         # Run spike detection on first kept sweep as demonstration
#         if len(kept_sweeps) > 0:
#             example_sweep = kept_sweeps[0]
#             print(f"\nStep 3: Example - Running spike detection on sweep {example_sweep}...")
            
#             # Get automatic analysis windows for this sweep
#             windows = get_analysis_windows_for_sweep(nwbfile, example_sweep)
            
#             print(f"  Stimulus period: [{windows['t_stim_start']:.6f}, {windows['t_stim_end']:.6f}] s")
#             print(f"  Analysis window: [{windows['t_min']:.6f}, {windows['t_max']:.6f}] s")
#             print(f"  Extended window: [{windows['t_start']:.6f}, {windows['t_end']:.6f}] s")
            
#             # Note: Full spike detection would require bundle path
#             if output_bundle_path:
#                 print(f"\n  Running spike detection with output to: {output_bundle_path}")
#                 results = run_spike_detection(
#                     df, 
#                     df_pA, 
#                     df_analysis, 
#                     fs, 
#                     output_bundle_path,
#                     pA_was_replaced=False,
#                     analysis_windows=windows
#                 )
#                 return {
#                     'results': results,
#                     'classification': classification,
#                     'df': df,
#                     'df_pA': df_pA,
#                     'df_analysis': df_analysis,
#                     'fs': fs,
#                     'windows_used': windows
#                 }
#             else:
#                 print("\n  (Skipping full spike detection - provide output_bundle_path to run)")
        
#         return {
#             'classification': classification,
#             'df': df,
#             'df_pA': df_pA,
#             'df_analysis': df_analysis,
#             'fs': fs,
#             'windows_available': {idx: get_analysis_windows_for_sweep(nwbfile, idx) 
#                                   for idx in kept_sweeps[:5]}  # Show first 5 as example
#         }


# def main():
#     """Main integration pipeline."""
    
#     print("\n" + "="*70)
#     print("NWB SWEEP ANALYSIS + SPIKE DETECTION INTEGRATION")
#     print("="*70)
    
#     # Read NWB file
#     print("\n1. Loading NWB file...")
#     with NWBHDF5IO(NWB_PATH, mode='r') as io:
#         nwbfile = io.read()
        
#         # Classify sweeps
#         print("2. Classifying sweeps...")
#         sweep_classification = classify_sweeps_from_nwb(nwbfile, MIN_STEP_S, MIN_FLAT_RATIO)
        
#         # Print summary
#         print_integration_summary(nwbfile, sweep_classification)
        
#         # Show how to use with spike detection
#         print(f"\n" + "="*70)
#         print("USAGE WITH SPIKE DETECTION")
#         print("="*70)
#         print("""
# Option 1: Run full pipeline (recommended)
    
#     from nwb_integration import run_spike_detection_on_kept_sweeps
    
#     results = run_spike_detection_on_kept_sweeps(
#         nwb_file_path="/path/to/your/file.nwb",
#         output_bundle_path="/path/to/output/bundle"
#     )
    
#     # This will:
#     #   1. Classify sweeps automatically
#     #   2. Convert NWB data to DataFrames
#     #   3. Run spike detection with automatic analysis windows

# Option 2: Manual control (for advanced use)
    
#     from nwb_integration import nwb_to_dataframes_for_spike_detection
#     from nwb_integration import get_analysis_windows_for_sweep
#     from spike_detection_new import run_spike_detection
    
#     # Convert NWB to DataFrames
#     df, df_pA, df_analysis, fs = nwb_to_dataframes_for_spike_detection(
#         nwbfile, 
#         kept_sweep_indices=[4, 5, 6, ...]  # your kept sweeps
#     )
    
#     # Get automatic analysis windows for a sweep
#     windows = get_analysis_windows_for_sweep(nwbfile, sweep_idx=4)
    
#     # Run spike detection
#     results = run_spike_detection(
#         df, df_pA, df_analysis, fs, bundle_path,
#         analysis_windows=windows
#     )
#         """)
    
#     print(f"\n" + "="*70)
#     print("INTEGRATION COMPLETE")
#     print("="*70)
#     print(f"\nKey Functions:")
#     print(f"  - classify_sweeps_from_nwb()            : Identify valid sweeps")
#     print(f"  - nwb_to_dataframes_for_spike_detection(): Convert NWB → DataFrames")
#     print(f"  - get_analysis_windows_for_sweep()      : Calculate windows from stimulus")
#     print(f"  - run_spike_detection_on_kept_sweeps()  : Complete pipeline")

# ============================================================================
# CONFIGURATION
# ============================================================================
# All thresholds and parameters are now centralized in analysis_config.py
# See analysis_config.py for:
#   - BASELINE_THRESHOLD_PA, STIMULUS_THRESHOLD_PA
#   - MIN_STIMULUS_DURATION_S, MIN_FLAT_RATIO
#   - RESPONSE_PADDING_S, BASELINE_FALLBACK_S
#   - SECOND_DERIV_THRESHOLD, VOLTAGE_JUMP_THRESHOLD
# ============================================================================

def find_contiguous_segments(mask):
    """
    Find start and end indices of contiguous True segments in a boolean mask.
    
    Returns: list of (start_idx, end_idx) tuples
    """
    edges = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return [(s, e) for s, e in zip(starts, ends) if e > s]


def find_baseline_window(current, time):
    """
    Find when NO current is injected (baseline period).
    
    Strategy:
    1. Look for first contiguous segment at start where current ≈ 0
    2. If none at start, use last segment at end
    3. Fallback: use first 10ms
    """
    is_baseline = np.abs(current) < BASELINE_THRESHOLD_PA
    baseline_segments = find_contiguous_segments(is_baseline)
    
    # Prefer first baseline segment
    if baseline_segments and baseline_segments[0][1] > baseline_segments[0][0]:
        start_idx, end_idx = baseline_segments[0]
        return time[start_idx], time[end_idx - 1], None
    
    # Otherwise use last baseline segment
    elif baseline_segments and baseline_segments[-1][1] > baseline_segments[-1][0]:
        start_idx, end_idx = baseline_segments[-1]
        return time[start_idx], time[end_idx - 1], None
    
    # Fallback: first 10ms (only if both detection methods fail)
    else:
        baseline_end = min(BASELINE_FALLBACK_S, time[-1])
        return 0.0, baseline_end, "WARNING: No baseline segment detected; using first 10ms as emergency fallback"


def find_stimulus_window(current, time):
    """
    Find when current IS injected (stimulus period).
    
    Returns the longest contiguous segment where |current| > threshold.
    Also returns the most common current value (the stimulus level).
    """
    is_stimulus = np.abs(current) > STIMULUS_THRESHOLD_PA
    stim_segments = find_contiguous_segments(is_stimulus)
    
    if not stim_segments:
        # No stimulus found - return full sweep time range
        return time[0], time[-1], 0.0
    
    # Use the longest stimulus segment
    longest_segment = max(stim_segments, key=lambda seg: seg[1] - seg[0])
    start_idx, end_idx = longest_segment
    
    # Calculate stimulus level (most common current value in this segment)
    stim_values = current[start_idx:end_idx]
    stim_rounded = np.round(stim_values, 1)
    unique_vals, counts = np.unique(stim_rounded, return_counts=True)
    stimulus_level = float(unique_vals[np.argmax(counts)])
    
    return time[start_idx], time[end_idx - 1], stimulus_level


def is_square_wave(current_segment):
    """
    Check if a current segment is a nice square wave (not a ramp or noisy).
    
    Criteria: At least 90% of the segment must be at or near the peak value.
    """
    peak = np.max(np.abs(current_segment))
    near_peak = np.abs(current_segment) > 0.9 * peak
    flat_ratio = np.sum(near_peak) / len(current_segment)
    
    return flat_ratio >= MIN_FLAT_RATIO


def validate_sweep(current, time):
    """
    Check if a sweep has valid data.
    
    Constraints:
    1. If stimulus is injected:
       - Must have at least one contiguous stimulus segment ≥300ms duration
       - That segment must be square-shaped (≥70% at peak value, not ramped)
    2. If no stimulus injected:
       - Can be a flat baseline sweep (≥90% at constant current value)
    3. No spikes are checked here (assumed voltage is in mV, beyond scope of current validation)
    
    Args:
        current: numpy array of current values (in pA)
        time: numpy array of time values (in seconds)
    
    Returns: (is_valid, reason_if_invalid)
    """
    is_stimulus = np.abs(current) > STIMULUS_THRESHOLD_PA
    stim_segments = find_contiguous_segments(is_stimulus)
    
    # Filter to only long segments (≥300ms) using actual time duration
    long_segments = [
        (s, e) for s, e in stim_segments 
        if (time[e-1] - time[s]) >= MIN_STIMULUS_DURATION_S
    ]
    
    # Check if sweep has a long stimulus segment
    if long_segments:
        # Check if any long segment is square-shaped (70% at peak)
        for start_idx, end_idx in long_segments:
            segment = current[start_idx:end_idx]
            if is_square_wave(segment):
                return True, None
        
        # Has stimulus but not square-shaped
        return False, "Stimulus not square-shaped"
    
    # Check if we have stimulus segments but they're all too short
    if stim_segments:
        return False, f"Stimulus segments too short (minimum {MIN_STIMULUS_DURATION_S}s required)"
    
    # No stimulus segments - check if it's a valid flat baseline sweep
    # A flat baseline must have very little variation
    current_rounded = np.round(current, 1)
    unique_vals, counts = np.unique(current_rounded, return_counts=True)
    mode_val = unique_vals[np.argmax(counts)]
    mode_ratio = np.max(counts) / len(current)
    
    # If >90% of the sweep is at one constant value, it's a valid flat baseline
    if mode_ratio >= 0.90:
        return True, None
    
    return False, "No stimulus and not a valid flat baseline"


def detect_right_angle_in_voltage(voltage, time, stim_start, stim_end, sampling_rate=None, sweep_id=None):
    """
    Detect sharp right angles (artifacts) in voltage trace during stimulus period.
    
    A "right angle" is detected using second derivative analysis - looking for
    sudden changes in the slope that indicate recording artifacts.
    
    Args:
        voltage: numpy array of voltage values (in mV)
        time: numpy array of time values (in seconds)
        stim_start: stimulus start time (seconds)
        stim_end: stimulus end time (seconds)
        sampling_rate: sampling rate in Hz (if None, will be estimated from time array)
        sweep_id: sweep ID for debugging output
    
    Returns:
        tuple: (has_artifact, description)
            has_artifact: True if right angle detected
            description: string describing the artifact if found
    """
    # Debug: print time range for first few sweeps
    if sweep_id is not None and sweep_id <= 10:
        print(f"  [DEBUG] Sweep {sweep_id}: Checking voltage in time range [{stim_start:.6f}, {stim_end:.6f}] s")
    
    # Extract voltage during stimulus period
    stim_mask = (time >= stim_start) & (time <= stim_end)
    v_stim = voltage[stim_mask]
    t_stim = time[stim_mask]
    
    if len(v_stim) < 10:
        return False, None  # Not enough data to analyze
    
    # Calculate sampling interval (dt)
    if sampling_rate is not None:
        dt = 1.0 / sampling_rate
    else:
        # Estimate from time array (more robust)
        dt = np.median(np.diff(t_stim))
        if dt <= 0:
            return False, None  # Invalid time data
    
    # Calculate first derivative (dV/dt) - rate of voltage change
    dv_dt = np.gradient(v_stim, dt)
    
    # Calculate second derivative (d²V/dt²) - rate of change of the slope
    # This peaks at sharp corners
    d2v_dt2 = np.gradient(dv_dt, dt)
    
    # Define thresholds for artifact detection:
    # 1. Extremely large second derivative indicates sharp corner
    #    Normal AP and typical variations: d²V/dt² up to ~2 billion mV/s²
    #    True right-angle artifacts: 20+ billion mV/s²
    second_deriv_threshold = SECOND_DERIV_THRESHOLD
    
    # 2. Sudden voltage jump > 10 mV in single sample
    voltage_jump_threshold = VOLTAGE_JUMP_THRESHOLD
    voltage_jumps = np.abs(np.diff(v_stim))
    max_jump = np.max(voltage_jumps) if len(voltage_jumps) > 0 else 0
    
    # Check for artifacts
    max_d2v = np.max(np.abs(d2v_dt2))
    
    # Detect artifact
    if max_d2v > second_deriv_threshold:
        return True, f"Sharp corner detected (d²V/dt²={max_d2v:.0f} mV/s²)"
    elif max_jump > voltage_jump_threshold:
        return True, f"Voltage discontinuity detected ({max_jump:.2f} mV jump)"
    
    return False, None


def analyze_single_sweep(current, time, voltage=None, sweep_id=None):
    """
    Analyze one sweep to identify baseline, stimulus, and response windows.
    
    Args:
        current: numpy array of current values (in pA)
        time: numpy array of time values (in seconds)
        voltage: numpy array of voltage values (in mV), optional
        sweep_id: sweep ID for debugging output
    
    Returns:
        dict with validation status and time windows
    """
    # Check if sweep is valid (using actual time array)
    is_valid, invalid_reason = validate_sweep(current, time)
    
    # Find baseline window (no current)
    baseline_start, baseline_end, baseline_note = find_baseline_window(current, time)
    
    # Find stimulus window (current injected)
    stim_start, stim_end, stim_level = find_stimulus_window(current, time)
    
    # Check for voltage artifacts (right angles) if voltage data is provided
    if voltage is not None and is_valid:
        has_artifact, artifact_desc = detect_right_angle_in_voltage(
            voltage, time, stim_start, stim_end, sweep_id=sweep_id
        )
        if has_artifact:
            is_valid = False
            invalid_reason = f"Voltage artifact in stimulus period: {artifact_desc}"
    
    # Build result
    result = {
        "valid": is_valid,
        "reason": invalid_reason,
        "stimulus_level_pA": float(stim_level),
        "windows": {
            "baseline_start_s": float(baseline_start),
            "baseline_end_s": float(baseline_end),
            "stimulus_start_s": float(stim_start),
            "stimulus_end_s": float(stim_end)
        }
    }
    
    if baseline_note:
        result["baseline_note"] = baseline_note
    
    return result


def sweep_config_to_json(bundle_dir, df_stim, manifest, df_voltage=None):
    """
    Export sweep classification and analysis windows as sweep_config.json
    
    Args:
        bundle_dir: Path to bundle directory
        df_stim: DataFrame with stimulus data (has 'sweep', 't_s', and 'value' columns)
        manifest: Manifest with metadata (currently unused in this function)
        df_voltage: DataFrame with voltage data (has 'sweep', 't_s', and 'value' columns), optional
    """
    p = Path(bundle_dir)
    
    # Get all unique sweep IDs (column is 'sweep', not 'sweep_id')
    all_sweeps = sorted(df_stim['sweep'].unique())
    
    # Initialize results
    sweep_config = {
        "sweeps": {},
        "kept_sweeps": [],
        "dropped_sweeps": [],
        "total_sweeps": len(all_sweeps)
    }
    
    valid_count = 0
    
    # Process each sweep
    print(f"\nAnalyzing {len(all_sweeps)} sweeps...")
    for sweep_id in all_sweeps:
        # Get data for this sweep
        df_sweep_stim = df_stim[df_stim['sweep'] == sweep_id].copy()
        
        if len(df_sweep_stim) == 0:
            print(f"  Warning: Sweep {sweep_id} has no data, skipping")
            continue
        
        # Extract stimulus array and time vector
        stimulus = df_sweep_stim["value"].values
        time = df_sweep_stim["t_s"].values
        
        # Extract voltage data if available
        voltage = None
        if df_voltage is not None:
            df_sweep_voltage = df_voltage[df_voltage['sweep'] == sweep_id].copy()
            if len(df_sweep_voltage) > 0:
                voltage = df_sweep_voltage["value"].values
        
        # Analyze this sweep using the stored time vector
        sweep_result = analyze_single_sweep(stimulus, time, voltage=voltage, sweep_id=sweep_id)
        
        # Store result
        sweep_config["sweeps"][str(sweep_id)] = sweep_result
        
        # Track valid/invalid
        if sweep_result["valid"]:
            valid_count += 1
            sweep_config["kept_sweeps"].append(int(sweep_id))
        else:
            sweep_config["dropped_sweeps"].append(int(sweep_id))
            print(f"  Sweep {sweep_id}: REJECTED - {sweep_result['reason']}")
    
    # Add summary statistics
    sweep_config["valid_sweeps"] = valid_count
    sweep_config["rejected_sweeps"] = len(all_sweeps) - valid_count
    
    # Check for lost current levels due to voltage artifacts
    print("\n[Checking for lost current levels due to voltage artifacts...]")
    voltage_artifact_sweeps = []
    for sweep_id, sweep_data in sweep_config["sweeps"].items():
        if not sweep_data["valid"] and "voltage artifact" in sweep_data.get("reason", "").lower():
            voltage_artifact_sweeps.append({
                "sweep_id": int(sweep_id),
                "current_pA": sweep_data["stimulus_level_pA"]
            })
    
    if voltage_artifact_sweeps:
        # Group valid sweeps by current level
        valid_currents = {}
        for sweep_id in sweep_config["kept_sweeps"]:
            sweep_data = sweep_config["sweeps"][str(sweep_id)]
            current = sweep_data["stimulus_level_pA"]
            if current not in valid_currents:
                valid_currents[current] = []
            valid_currents[current].append(sweep_id)
        
        # Check each artifact sweep
        lost_currents = []
        for artifact_sweep in voltage_artifact_sweeps:
            sweep_id = artifact_sweep["sweep_id"]
            current = artifact_sweep["current_pA"]
            
            if current not in valid_currents:
                lost_currents.append({
                    "sweep_id": sweep_id,
                    "current_pA": current
                })
                print(f"  ⚠ WARNING: Sweep {sweep_id} ({current:.1f} pA) rejected for voltage artifact, NO other valid sweep at this current level")
        
        if lost_currents:
            print(f"\n  ⚠ {len(lost_currents)} current level(s) lost due to voltage artifacts:")
            for item in lost_currents:
                print(f"    - {item['current_pA']:.1f} pA (sweep {item['sweep_id']})")
        else:
            print(f"  ✓ All rejected artifact sweeps have alternative valid sweeps at same current levels")
    
    # Save to JSON
    config_path = p / "sweep_config.json"
    with open(config_path, "w") as f:
        json.dump(sweep_config, f, indent=2)
    
    # Print summary
    print(f"\n✓ Saved sweep_config.json: {config_path}")
    print(f"  Total sweeps: {len(all_sweeps)}")
    print(f"  Valid sweeps: {valid_count}")
    print(f"  Rejected sweeps: {len(all_sweeps) - valid_count}")
    print(f"  Kept sweeps: {sweep_config['kept_sweeps']}")
    print(f"  Dropped sweeps: {sweep_config['dropped_sweeps']}")
    
    return sweep_config



def process_bundle(bundle_dir):
    """
    Analyze a bundle directory and create sweep_config.json
    
    This is the ENTRY POINT for Option A (pre-processing step).
    
    Args:
        bundle_dir: Path to bundle directory containing:
                   - mv_*.parquet (voltage)
                   - pa_*.parquet (current)
                   - manifest.json (metadata)
    
    Returns:
        dict: sweep_config
    """
    import json
    from pathlib import Path
    
    p = Path(bundle_dir)
    
    # Check if sweep_config already exists
    config_path = p / "sweep_config.json"
    if config_path.exists():
        print(f"\n⚠ sweep_config.json already exists in {p.name}")
        # Auto-yes overwrite
        print("Auto-overwriting existing sweep_config.json...")
    
    print(f"\n{'='*70}")
    print(f"ANALYZING BUNDLE: {p.name}")
    print(f"{'='*70}")
    
    # Load manifest
    manifest_path = p / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {bundle_dir}")
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Load parquet files
    print("\nLoading data files...")
    
    # Check if this is a mixed protocol file
    # Mixed protocol files have both "stimulus" and "response" tables
    is_mixed_protocol = "stimulus" in manifest["tables"] and "response" in manifest["tables"]
    
    if is_mixed_protocol:
        # For mixed protocol, use stimulus table to determine valid sweeps
        print("  Detected mixed protocol - using stimulus table for classification")
        stim_path = p / manifest["tables"]["stimulus"]
        df_analysis = pd.read_parquet(stim_path)
    else:
        # For single protocol, use pa table (original behavior)
        print("  Detected single protocol - using pA table for classification")
        pa_path = p / manifest["tables"]["pa"]
        df_analysis = pd.read_parquet(pa_path)
    
    print(f"✓ Loaded {len(df_analysis)} stimulus samples across {df_analysis['sweep'].nunique()} sweeps")
    
    # Load voltage data for artifact detection
    print("  Loading voltage data for artifact detection...")
    if is_mixed_protocol:
        mv_path = p / manifest["tables"]["response"]
    else:
        mv_path = p / manifest["tables"]["mv"]
    df_voltage = pd.read_parquet(mv_path)
    print(f"✓ Loaded {len(df_voltage)} voltage samples")
    
    # Generate sweep config
    print("\nClassifying sweeps and calculating analysis windows...")
    sweep_config = sweep_config_to_json(bundle_dir, df_analysis, manifest, df_voltage=df_voltage)
    
    # Print summary
    print("\n" + "="*70)
    print("SWEEP ANALYSIS SUMMARY")
    print("="*70)
    print(f"\nBundle: {p.name}")
    print(f"Cell Number: {manifest.get('meta', {}).get('cellNum', 'unknown')}")
    print(f"Sample Rate: {manifest.get('meta', {}).get('sampleRate_Hz', 'unknown')} Hz")
    print(f"Total Sweeps: {sweep_config['total_sweeps']}")
    print(f"Valid Sweeps: {sweep_config['valid_sweeps']}")
    print(f"Rejected Sweeps: {sweep_config['rejected_sweeps']}")
    
    # Show details for first few valid sweeps
    print(f"\nFirst 3 valid sweeps:")
    count = 0
    for sweep_id_str, sweep_info in sweep_config["sweeps"].items():
        if sweep_info["valid"] and count < 3:
            sweep_id = int(sweep_id_str)
            windows = sweep_info["windows"]
            stim_level = sweep_info["stimulus_level_pA"]
            print(f"\n  Sweep {sweep_id}:")
            print(f"    Stimulus level: {stim_level:.1f} pA")
            print(f"    Stimulus window: [{windows['stimulus_start_s']:.3f}, {windows['stimulus_end_s']:.3f}] s")
            print(f"    Baseline window: [{windows['baseline_start_s']:.3f}, {windows['baseline_end_s']:.3f}] s")
            count += 1
    
    print("\n" + "="*70)
    print("✓ Bundle processing complete. Ready for analysis.")
    print("="*70)
    
    # Optional diagnostic output - only warn if critically low valid sweeps
    # Warn if: fewer than 5 valid sweeps OR less than 10% of total sweeps are valid
    valid_ratio = sweep_config['valid_sweeps'] / sweep_config['total_sweeps'] if sweep_config['total_sweeps'] > 0 else 0
    if sweep_config['valid_sweeps'] < 5 or valid_ratio < 0.10:
        print(f"\n⚠ WARNING: Very few valid sweeps detected ({sweep_config['valid_sweeps']} out of {sweep_config['total_sweeps']}, {valid_ratio*100:.1f}%)")
        print(f"  This may indicate an issue with:")
        print(f"    - Classification thresholds (MIN_STIMULUS_DURATION_S={MIN_STIMULUS_DURATION_S}, MIN_FLAT_RATIO={MIN_FLAT_RATIO})")
        print(f"    - Data quality")
        print(f"    - Stimulus protocol")
        
        # Auto-skip diagnostics (info already printed during classification)
        print(f"\n  (Detailed diagnostics available in sweep_config.json)")
    
    return sweep_config


def combine_images_to_pdf(image_paths, pdf_path):
    """
    Combine multiple image files into a single PDF.
    
    Args:
        image_paths: List of Path objects or strings pointing to image files
        pdf_path: Path object or string for output PDF file
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    
    # Filter to only existing files
    existing_paths = [p for p in image_paths if Path(p).exists()]
    
    if not existing_paths:
        print("  ⚠ No images found to combine into PDF")
        return
    
    with PdfPages(pdf_path) as pdf:
        for img_path in existing_paths:
            # Read image
            img = mpimg.imread(img_path)
            
            # Get image aspect ratio to maintain proportions
            height, width = img.shape[:2]
            aspect = width / height
            
            # Create figure with size based on image aspect ratio (larger for better quality)
            fig_width = 16
            fig_height = fig_width / aspect
            fig = plt.figure(figsize=(fig_width, fig_height))
            ax = fig.add_subplot(111)
            ax.imshow(img, interpolation='lanczos')  # High-quality interpolation
            ax.axis('off')
            plt.tight_layout(pad=0)
            
            # Add to PDF with high DPI for better quality
            pdf.savefig(fig, dpi=300, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
    
    print(f"  ✓ Combined {len(existing_paths)} plots into: {pdf_path}")


def visualize_sweeps_from_parquet(bundle_dir, kept_sweeps, dropped_sweeps):
    """
    Create visualization of kept vs dropped sweeps from parquet data.
    Creates separate plots for voltage (mV) and current (pA) for both kept and dropped sweeps.
    
    Args:
        bundle_dir: Path to bundle directory
        kept_sweeps: List of sweep indices to keep
        dropped_sweeps: List of sweep indices to drop
    
    Returns:
        None (saves JPEG files and combined PDF)
    """
    import matplotlib.pyplot as plt
    
    p = Path(bundle_dir)
    manifest_path = p / "manifest.json"
    
    # Track all saved plot paths for PDF combination
    plot_paths = []
    
    # Add grid plots created during data preparation (if they exist)
    voltage_grid = p / "voltage_grid.png"
    current_grid = p / "current_grid.png"
    if voltage_grid.exists():
        plot_paths.append(voltage_grid)
    if current_grid.exists():
        plot_paths.append(current_grid)
    
    # Load manifest and parquets
    import json
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Check if mixed protocol
    is_mixed = "stimulus" in manifest["tables"] and "response" in manifest["tables"]
    
    if is_mixed:
        # For mixed protocol, use stimulus and response tables
        df_stim = pd.read_parquet(p / manifest["tables"]["stimulus"])
        df_resp = pd.read_parquet(p / manifest["tables"]["response"])
    else:
        # For single protocol, use the unified tables
        df_mv = pd.read_parquet(p / manifest["tables"]["mv"])
        df_pa = pd.read_parquet(p / manifest["tables"]["pa"])
        df_stim = df_pa  # For single protocol, stimulus is current
        df_resp = df_mv  # For single protocol, response is voltage
    
    # Load sweep_config if available for stimulus window markers
    sweep_config = None
    sweep_config_path = p / "sweep_config.json"
    if sweep_config_path.exists():
        with open(sweep_config_path) as f:
            sweep_config = json.load(f)
    
    # Get unique sweeps
    all_sweeps = sorted(df_resp['sweep'].unique())
    
    # Helper function to add vertical dashed lines for stimulus window (for single protocol kept sweeps)
    def add_stimulus_vertical_lines(ax, sweep_id=None, sweep_start_time=None):
        """Add vertical dashed lines for stimulus window if sweep_config available
        
        Args:
            ax: matplotlib axis
            sweep_id: ID of the sweep to get stimulus times from
            sweep_start_time: Absolute start time of the sweep to convert to relative time
        """
        if sweep_config is None:
            return
        try:
            if sweep_id is not None:
                # Use specific sweep's stimulus window
                sweep_windows = sweep_config["sweeps"].get(str(sweep_id), {}).get("windows", {})
                if not sweep_windows and isinstance(sweep_id, int):
                    # Try integer key
                    sweep_windows = sweep_config["sweeps"].get(str(sweep_id), {}).get("windows", {})
            else:
                # Use first valid sweep's stimulus window for uniform display
                for sweep_key, sweep_data in sweep_config.get("sweeps", {}).items():
                    if sweep_data.get("valid", False):
                        sweep_windows = sweep_data.get("windows", {})
                        break
            
            t_stim_start_abs = sweep_windows.get("stimulus_start_s")
            t_stim_end_abs = sweep_windows.get("stimulus_end_s")
            if t_stim_start_abs is not None and t_stim_end_abs is not None:
                # Convert to relative time if sweep_start_time provided
                if sweep_start_time is not None:
                    t_stim_start = t_stim_start_abs - sweep_start_time
                    t_stim_end = t_stim_end_abs - sweep_start_time
                else:
                    t_stim_start = t_stim_start_abs
                    t_stim_end = t_stim_end_abs
                
                # Add vertical dashed lines (NO stars, NO shading)
                ax.axvline(x=t_stim_start, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Stim start')
                ax.axvline(x=t_stim_end, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Stim end')
        except (KeyError, TypeError, AttributeError):
            pass  # Stimulus window not available
    
    # Helper function to add stimulus window star markers only
    def add_stimulus_stars_only(ax, sweep_id=None, sweep_start_time=None):
        """Add star markers for stimulus window if sweep_config available
        
        Args:
            ax: matplotlib axis
            sweep_id: ID of the sweep to get stimulus times from
            sweep_start_time: Absolute start time of the sweep to convert to relative time
        """
        if sweep_config is None:
            return
        try:
            if sweep_id is not None:
                # Use specific sweep's stimulus window
                sweep_windows = sweep_config["sweeps"].get(str(sweep_id), {}).get("windows", {})
                if not sweep_windows and isinstance(sweep_id, int):
                    # Try integer key
                    sweep_windows = sweep_config["sweeps"].get(str(sweep_id), {}).get("windows", {})
            else:
                # Use first valid sweep's stimulus window for uniform display
                for sweep_key, sweep_data in sweep_config.get("sweeps", {}).items():
                    if sweep_data.get("valid", False):
                        sweep_windows = sweep_data.get("windows", {})
                        break
            
            t_stim_start_abs = sweep_windows.get("stimulus_start_s")
            t_stim_end_abs = sweep_windows.get("stimulus_end_s")
            if t_stim_start_abs is not None and t_stim_end_abs is not None:
                # Convert to relative time if sweep_start_time provided
                if sweep_start_time is not None:
                    t_stim_start = t_stim_start_abs - sweep_start_time
                    t_stim_end = t_stim_end_abs - sweep_start_time
                else:
                    t_stim_start = t_stim_start_abs
                    t_stim_end = t_stim_end_abs
                
                # Get all lines from the plot to find y-values at stimulus times
                lines = ax.get_lines()
                if lines:
                    # Use the first line (first sweep) to get y-values
                    line = lines[0]
                    x_data = line.get_xdata()
                    y_data = line.get_ydata()
                    
                    # Find closest data point to stimulus start
                    idx_start = np.argmin(np.abs(x_data - t_stim_start))
                    y_start = y_data[idx_start]
                    
                    # Find closest data point to stimulus end
                    idx_end = np.argmin(np.abs(x_data - t_stim_end))
                    y_end = y_data[idx_end]
                    
                    # Add star markers ON the data line
                    ax.plot(t_stim_start, y_start, marker='*', markersize=15, color='green', 
                           markeredgecolor='darkgreen', markeredgewidth=1.5, zorder=10, label='Stim start')
                    ax.plot(t_stim_end, y_end, marker='*', markersize=15, color='orange',
                           markeredgecolor='darkorange', markeredgewidth=1.5, zorder=10, label='Stim end')
        except (KeyError, TypeError, AttributeError):
            pass  # Stimulus window not available
    
    # Helper function to add stimulus window markers (full version for mixed protocol)
    def add_stimulus_markers(ax, sweep_id=None, sweep_start_time=None):
        """Add vertical lines for stimulus window if sweep_config available
        
        Args:
            ax: matplotlib axis
            sweep_id: ID of the sweep to get stimulus times from
            sweep_start_time: Absolute start time of the sweep to convert to relative time
        """
        if sweep_config is None:
            return
        try:
            if sweep_id is not None:
                # Use specific sweep's stimulus window
                sweep_windows = sweep_config["sweeps"].get(str(sweep_id), {}).get("windows", {})
                if not sweep_windows and isinstance(sweep_id, int):
                    # Try integer key
                    sweep_windows = sweep_config["sweeps"].get(str(sweep_id), {}).get("windows", {})
            else:
                # Use first valid sweep's stimulus window for uniform display
                for sweep_key, sweep_data in sweep_config.get("sweeps", {}).items():
                    if sweep_data.get("valid", False):
                        sweep_windows = sweep_data.get("windows", {})
                        break
            
            t_stim_start_abs = sweep_windows.get("stimulus_start_s")
            t_stim_end_abs = sweep_windows.get("stimulus_end_s")
            if t_stim_start_abs is not None and t_stim_end_abs is not None:
                # Convert to relative time if sweep_start_time provided
                if sweep_start_time is not None:
                    t_stim_start = t_stim_start_abs - sweep_start_time
                    t_stim_end = t_stim_end_abs - sweep_start_time
                else:
                    t_stim_start = t_stim_start_abs
                    t_stim_end = t_stim_end_abs
                
                # Add vertical lines
                ax.axvline(x=t_stim_start, color='green', linestyle='--', linewidth=2, alpha=0.6, label='Stim start')
                ax.axvline(x=t_stim_end, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='Stim end')
                ax.axvspan(t_stim_start, t_stim_end, alpha=0.05, color='yellow')
                
                # Add star markers at the top of the plot
                ylim = ax.get_ylim()
                y_star = ylim[1] - 0.05 * (ylim[1] - ylim[0])  # 5% from top
                ax.plot(t_stim_start, y_star, marker='*', markersize=15, color='green', 
                       markeredgecolor='darkgreen', markeredgewidth=1.5, zorder=10)
                ax.plot(t_stim_end, y_star, marker='*', markersize=15, color='orange',
                       markeredgecolor='darkorange', markeredgewidth=1.5, zorder=10)
        except (KeyError, TypeError, AttributeError):
            pass  # Stimulus window not available
    
    # ========== KEPT SWEEPS - VOLTAGE (Response) ==========
    fig_kept_mv, ax_kept_mv = plt.subplots(figsize=(14, 6))
    kept_count = 0
    first_sweep_start_time = None
    for sweep_id in kept_sweeps:
        if sweep_id in all_sweeps:
            sweep_data = df_resp[df_resp['sweep'] == sweep_id]
            # For mixed protocol, filter to only voltage/volts units
            if is_mixed:
                sweep_data = sweep_data[sweep_data['unit'].str.lower().str.contains('volt', na=False)]
            time = sweep_data['t_s'].values
            if first_sweep_start_time is None:
                first_sweep_start_time = time[0]
            time_rel = time - time[0]  # Convert to relative time
            voltage = sweep_data['value'].values
            # Note: Units in NWB file say "volts" but values are already in mV
            ax_kept_mv.plot(time_rel, voltage, alpha=0.7, linewidth=1.5, label=f"Sweep {sweep_id}")
            kept_count += 1
    
    ax_kept_mv.set_title(f"KEPT Sweeps - Voltage Response (mV) ({kept_count} total)", fontsize=14, fontweight='bold')
    ax_kept_mv.set_xlabel("Time (s)", fontsize=12)
    ax_kept_mv.set_ylabel("Voltage (mV)", fontsize=12)
    ax_kept_mv.grid(True, alpha=0.3)
    
    if kept_count > 0 and kept_count <= 6:
        ax_kept_mv.legend(fontsize=9, loc='best')
    elif kept_count > 6:
        ax_kept_mv.legend(fontsize=7, loc='best', ncol=2)
    plt.tight_layout()
    
    # Add vertical dashed lines for stimulus window (single protocol only)
    # Must be after tight_layout to get correct y-axis limits
    if kept_count > 0 and not is_mixed:
        # For single protocol, add stimulus window vertical lines using first kept sweep
        add_stimulus_vertical_lines(ax_kept_mv, sweep_id=kept_sweeps[0], sweep_start_time=first_sweep_start_time)
    
    kept_mv_path = p / "kept_sweeps_voltage.png"
    fig_kept_mv.savefig(kept_mv_path, dpi=300, format='png')
    plt.close(fig_kept_mv)
    print(f"  ✓ Saved: {kept_mv_path}")
    plot_paths.append(kept_mv_path)
    
    # ========== KEPT SWEEPS - CURRENT (Stimulus) ==========
    fig_kept_pa, ax_kept_pa = plt.subplots(figsize=(14, 6))
    first_sweep_start_time_pa = None
    for sweep_id in kept_sweeps:
        if sweep_id in all_sweeps:
            sweep_data = df_stim[df_stim['sweep'] == sweep_id]
            # For mixed protocol, filter to only current/amperes units
            if is_mixed:
                sweep_data = sweep_data[sweep_data['unit'].str.lower().str.contains('amp', na=False)]
            time = sweep_data['t_s'].values
            if first_sweep_start_time_pa is None:
                first_sweep_start_time_pa = time[0]
            time_rel = time - time[0]  # Convert to relative time
            current = sweep_data['value'].values
            # Note: Units in NWB file say "amperes" but values are already in pA
            ax_kept_pa.plot(time_rel, current, alpha=0.7, linewidth=1.5, label=f"Sweep {sweep_id}")
    
    ax_kept_pa.set_title(f"KEPT Sweeps - Current Stimulus (pA) ({kept_count} total)", fontsize=14, fontweight='bold')
    ax_kept_pa.set_xlabel("Time (s)", fontsize=12)
    ax_kept_pa.set_ylabel("Current (pA)", fontsize=12)
    ax_kept_pa.grid(True, alpha=0.3)
    
    if kept_count > 0 and kept_count <= 6:
        ax_kept_pa.legend(fontsize=9, loc='best')
    elif kept_count > 6:
        ax_kept_pa.legend(fontsize=7, loc='best', ncol=2)
    plt.tight_layout()
    
    # Add vertical dashed lines for stimulus window (single protocol only)
    # Must be after tight_layout to get correct y-axis limits
    if kept_count > 0 and not is_mixed:
        # For single protocol, add stimulus window vertical lines using first kept sweep
        add_stimulus_vertical_lines(ax_kept_pa, sweep_id=kept_sweeps[0], sweep_start_time=first_sweep_start_time_pa)
    
    kept_pa_path = p / "kept_sweeps_current.png"
    fig_kept_pa.savefig(kept_pa_path, dpi=300, format='png')
    plt.close(fig_kept_pa)
    print(f"  ✓ Saved: {kept_pa_path}")
    plot_paths.append(kept_pa_path)
    
    # ========== DROPPED SWEEPS - VOLTAGE ==========
    if len(dropped_sweeps) > 0:
        fig_dropped_mv, ax_dropped_mv = plt.subplots(figsize=(14, 6))
        dropped_count = 0
        first_dropped_start_time = None
        for sweep_id in dropped_sweeps:
            if sweep_id in all_sweeps:
                sweep_data = df_resp[df_resp['sweep'] == sweep_id]
                # For mixed protocol, filter to only voltage/volts units
                if is_mixed:
                    sweep_data = sweep_data[sweep_data['unit'].str.lower().str.contains('volt', na=False)]
                time = sweep_data['t_s'].values
                if first_dropped_start_time is None:
                    first_dropped_start_time = time[0]
                time_rel = time - time[0]  # Convert to relative time
                voltage = sweep_data['value'].values
                # Note: Units in NWB file say "volts" but values are already in mV
                ax_dropped_mv.plot(time_rel, voltage, alpha=0.7, linewidth=1.5, label=f"Sweep {sweep_id}", color='red')
                dropped_count += 1
        
        ax_dropped_mv.set_title(f"DROPPED Sweeps - Voltage Response (mV) ({dropped_count} total)", fontsize=14, fontweight='bold', color='red')
        ax_dropped_mv.set_xlabel("Time (s)", fontsize=12)
        ax_dropped_mv.set_ylabel("Voltage (mV)", fontsize=12)
        ax_dropped_mv.grid(True, alpha=0.3)
        if dropped_count > 0 and dropped_count <= 6:
            ax_dropped_mv.legend(fontsize=9, loc='best')
        elif dropped_count > 6:
            ax_dropped_mv.legend(fontsize=7, loc='best', ncol=2)
        plt.tight_layout()
        
        dropped_mv_path = p / "dropped_sweeps_voltage.png"
        fig_dropped_mv.savefig(dropped_mv_path, dpi=300, format='png')
        plt.close(fig_dropped_mv)
        print(f"  ✓ Saved: {dropped_mv_path}")
        plot_paths.append(dropped_mv_path)
        
        # ========== DROPPED SWEEPS - CURRENT ==========
        fig_dropped_pa, ax_dropped_pa = plt.subplots(figsize=(14, 6))
        first_dropped_start_time_pa = None
        for sweep_id in dropped_sweeps:
            if sweep_id in all_sweeps:
                sweep_data = df_stim[df_stim['sweep'] == sweep_id]
                # For mixed protocol, filter to only current/amperes units
                if is_mixed:
                    sweep_data = sweep_data[sweep_data['unit'].str.lower().str.contains('amp', na=False)]
                time = sweep_data['t_s'].values
                if first_dropped_start_time_pa is None:
                    first_dropped_start_time_pa = time[0]
                time_rel = time - time[0]  # Convert to relative time
                current = sweep_data['value'].values
                # Note: Units in NWB file say "amperes" but values are already in pA
                ax_dropped_pa.plot(time_rel, current, alpha=0.7, linewidth=1.5, label=f"Sweep {sweep_id}", color='red')
        
        ax_dropped_pa.set_title(f"DROPPED Sweeps - Current Stimulus (pA) ({dropped_count} total)", fontsize=14, fontweight='bold', color='red')
        ax_dropped_pa.set_xlabel("Time (s)", fontsize=12)
        ax_dropped_pa.set_ylabel("Current (pA)", fontsize=12)
        ax_dropped_pa.grid(True, alpha=0.3)
        if dropped_count > 0 and dropped_count <= 6:
            ax_dropped_pa.legend(fontsize=9, loc='best')
        elif dropped_count > 6:
            ax_dropped_pa.legend(fontsize=7, loc='best', ncol=2)
        plt.tight_layout()
        
        dropped_pa_path = p / "dropped_sweeps_current.png"
        fig_dropped_pa.savefig(dropped_pa_path, dpi=300, format='png')
        plt.close(fig_dropped_pa)
        print(f"  ✓ Saved: {dropped_pa_path}")
        plot_paths.append(dropped_pa_path)
    else:
        print(f"  ℹ No dropped sweeps to visualize")
    
    # Combine all plots into a single PDF
    pdf_path = p / "all_plots_summary.pdf"
    combine_images_to_pdf(plot_paths, pdf_path)


def visualize_mixed_protocol_sweeps(bundle_dir, kept_sweeps, dropped_sweeps):
    """
    Visualization for MIXED PROTOCOL files - GRID LAYOUT.
    
    Shows each sweep SEPARATELY in a grid on the same screen.
    - Kept sweeps: Both stimulus and response plots
    - Dropped sweeps: Stimulus only
    
    Markers show stimulus start (*) and end (*) points for kept sweeps.
    """
    import matplotlib.pyplot as plt
    import math
    
    p = Path(bundle_dir)
    manifest_path = p / "manifest.json"
    
    # Track all saved plot paths for PDF combination
    plot_paths = []
    
    # Add grid plots created during data preparation (if they exist)
    stimulus_grid = p / "stimulus_grid.png"
    response_grid = p / "response_grid.png"
    if stimulus_grid.exists():
        plot_paths.append(stimulus_grid)
    if response_grid.exists():
        plot_paths.append(response_grid)
    
    import json
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    df_stim = pd.read_parquet(p / manifest["tables"]["stimulus"])
    df_resp = pd.read_parquet(p / manifest["tables"]["response"])
    
    # Load sweep_config to get stimulus windows
    sweep_config = None
    sweep_config_path = p / "sweep_config.json"
    if sweep_config_path.exists():
        with open(sweep_config_path) as f:
            sweep_config = json.load(f)
    
    all_sweeps = sorted(df_stim['sweep'].unique())
    
    # ========== KEPT SWEEPS - STIMULUS (GRID) ==========
    if len(kept_sweeps) > 0:
        # Calculate grid dimensions (aim for roughly square grid)
        ncols = 4
        nrows = math.ceil(len(kept_sweeps) / ncols)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3 * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1 or ncols == 1:
            axes = axes.reshape(nrows, ncols)
        
        for idx, sweep_id in enumerate(kept_sweeps):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row, col]
            
            if sweep_id in all_sweeps:
                sweep_data = df_stim[df_stim['sweep'] == sweep_id]
                sweep_data = sweep_data[sweep_data['unit'].str.lower().str.contains('amp', na=False)]
                
                if len(sweep_data) > 0:
                    time = sweep_data['t_s'].values
                    time_rel = time - time[0]
                    current = sweep_data['value'].values
                    
                    ax.plot(time_rel, current, linewidth=1.5, color='steelblue')
                    
                    # Add stimulus start/end markers if available
                    if sweep_config:
                        try:
                            sweep_windows = sweep_config["sweeps"].get(str(sweep_id), {}).get("windows", {})
                            t_stim_start_abs = sweep_windows.get("stimulus_start_s")
                            t_stim_end_abs = sweep_windows.get("stimulus_end_s")
                            if t_stim_start_abs is not None and t_stim_end_abs is not None:
                                # Convert absolute times to relative times for this sweep
                                t_stim_start_rel = t_stim_start_abs - time[0]
                                t_stim_end_rel = t_stim_end_abs - time[0]
                                
                                # Find the closest indices in the relative time array
                                idx_start = np.argmin(np.abs(time_rel - t_stim_start_rel))
                                idx_end = np.argmin(np.abs(time_rel - t_stim_end_rel))
                                ax.plot(time_rel[idx_start], current[idx_start], '*', 
                                       markersize=15, color='green', zorder=5)
                                ax.plot(time_rel[idx_end], current[idx_end], '*', 
                                       markersize=15, color='red', zorder=5)
                        except (KeyError, TypeError, IndexError):
                            pass
                    
                    ax.set_title(f"Sweep {sweep_id}", fontsize=10, fontweight='bold')
                    ax.set_xlabel("Time (s)", fontsize=9)
                    ax.set_ylabel("Stimulus", fontsize=9)
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(labelsize=8)
        
        # Hide unused subplots
        for idx in range(len(kept_sweeps), nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row, col].set_visible(False)
        
        fig.suptitle(f"KEPT Sweeps - Current Stimulus ({len(kept_sweeps)} sweeps) [* = stim start/end]", 
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        kept_stim_path = p / "kept_sweeps_stimulus.png"
        fig.savefig(kept_stim_path, dpi=300, format='png', bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {kept_stim_path}")
        plot_paths.append(kept_stim_path)
    
    # ========== KEPT SWEEPS - RESPONSE (GRID) ==========
    if len(kept_sweeps) > 0:
        ncols = 4
        nrows = math.ceil(len(kept_sweeps) / ncols)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3 * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1 or ncols == 1:
            axes = axes.reshape(nrows, ncols)
        
        for idx, sweep_id in enumerate(kept_sweeps):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row, col]
            
            if sweep_id in all_sweeps:
                sweep_data = df_resp[df_resp['sweep'] == sweep_id]
                sweep_data = sweep_data[sweep_data['unit'].str.lower().str.contains('volt', na=False)]
                
                if len(sweep_data) > 0:
                    time = sweep_data['t_s'].values
                    time_rel = time - time[0]
                    response = sweep_data['value'].values
                    
                    ax.plot(time_rel, response, linewidth=1.5, color='darkgreen')
                    
                    # Add stimulus start/end markers if available
                    if sweep_config:
                        try:
                            sweep_windows = sweep_config["sweeps"].get(str(sweep_id), {}).get("windows", {})
                            t_stim_start_abs = sweep_windows.get("stimulus_start_s")
                            t_stim_end_abs = sweep_windows.get("stimulus_end_s")
                            if t_stim_start_abs is not None and t_stim_end_abs is not None:
                                # Convert absolute times to relative times for this sweep
                                t_stim_start_rel = t_stim_start_abs - time[0]
                                t_stim_end_rel = t_stim_end_abs - time[0]
                                
                                # Find the closest indices in the relative time array
                                idx_start = np.argmin(np.abs(time_rel - t_stim_start_rel))
                                idx_end = np.argmin(np.abs(time_rel - t_stim_end_rel))
                                ax.plot(time_rel[idx_start], response[idx_start], '*', 
                                       markersize=15, color='green', zorder=5)
                                ax.plot(time_rel[idx_end], response[idx_end], '*', 
                                       markersize=15, color='red', zorder=5)
                        except (KeyError, TypeError, IndexError):
                            pass
                    
                    ax.set_title(f"Sweep {sweep_id}", fontsize=10, fontweight='bold')
                    ax.set_xlabel("Time (s)", fontsize=9)
                    ax.set_ylabel("Response", fontsize=9)
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(labelsize=8)
        
        # Hide unused subplots
        for idx in range(len(kept_sweeps), nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row, col].set_visible(False)
        
        fig.suptitle(f"KEPT Sweeps - Voltage Response ({len(kept_sweeps)} sweeps) [* = stim start/end]", 
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        kept_resp_path = p / "kept_sweeps_response.png"
        fig.savefig(kept_resp_path, dpi=300, format='png', bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {kept_resp_path}")
        plot_paths.append(kept_resp_path)
    
    # ========== DROPPED SWEEPS - STIMULUS (GRID) ==========
    # Dropped sweeps contain:
    # - Sweeps 0-3, 94-96: VoltageClamp with voltage stimulus
    # - Sweeps 50-93: CurrentClamp with current stimulus
    if len(dropped_sweeps) > 0:
        ncols = 4
        nrows = math.ceil(len(dropped_sweeps) / ncols)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3 * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1 or ncols == 1:
            axes = axes.reshape(nrows, ncols)
        
        for idx, sweep_id in enumerate(dropped_sweeps):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row, col]
            
            if sweep_id in all_sweeps:
                sweep_data = df_stim[df_stim['sweep'] == sweep_id]
                
                if len(sweep_data) > 0:
                    time = sweep_data['t_s'].values
                    time_rel = time - time[0]
                    stim_value = sweep_data['value'].values
                    
                    ax.plot(time_rel, stim_value, linewidth=1.5, color='coral')
                    
                    ax.set_title(f"Sweep {sweep_id}", fontsize=10, fontweight='bold')
                    ax.set_xlabel("Time (s)", fontsize=9)
                    ax.set_ylabel("Stimulus", fontsize=9)
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(labelsize=8)
        
        # Hide unused subplots
        for idx in range(len(dropped_sweeps), nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row, col].set_visible(False)
        
        fig.suptitle(f"DROPPED Sweeps - Stimulus ({len(dropped_sweeps)} sweeps)", 
                     fontsize=14, fontweight='bold', y=0.995, color='red')
        plt.tight_layout()
        dropped_stim_path = p / "dropped_sweeps_stimulus.png"
        fig.savefig(dropped_stim_path, dpi=300, format='png', bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Saved: {dropped_stim_path}")
        plot_paths.append(dropped_stim_path)
    else:
        print(f"  ℹ No dropped sweeps to visualize")
    
    # Combine all plots into a single PDF
    pdf_path = p / "all_plots_summary.pdf"
    combine_images_to_pdf(plot_paths, pdf_path)



if __name__ == "__main__":
    print("sweep_classifier.py - Sweep classification utilities")
    print("\nThis module is used by bundle_analyzer.py")
    print("Run: python bundle_analyzer.py /path/to/bundle_dir")
