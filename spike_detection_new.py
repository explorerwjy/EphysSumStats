from math import floor
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path
from analysis_config import (
    PRE_THRESHOLD_WINDOW_MS,
    POST_THRESHOLD_WINDOW_MS,
    PRE_THRESHOLD_WINDOW_S,
    POST_THRESHOLD_WINDOW_S,
    THRESHOLD_PERCENT,
    FAST_TROUGH_PERCENT,
    PEAK_HEIGHT_THRESHOLD,
    PEAK_PROMINENCE,
    MIN_PEAK_DISTANCE_MS,
    MIN_PEAK_DISTANCE_S,
    MIN_PEAK_THRESHOLD_AMPLITUDE_MV,
)

def dbg(msg):
    print(f"[DEBUG] {msg}")

# def calculate_stimulus_bounds_from_nwb(stimulus_data, sampling_rate, baseline_fraction=0.05, tolerance=0.05):
#     """
#     Calculate stimulus onset and offset times from injected current data.
    
#     Args:
#         stimulus_data: numpy array of stimulus/current values
#         sampling_rate: sampling rate in Hz
#         baseline_fraction: fraction of data at start to use for baseline estimation
#         tolerance: multiplier for baseline to determine threshold
    
#     Returns:
#         tuple: (t_stim_start, t_stim_end) in seconds, or (None, None) if no stimulus found
#     """
#     # Calculate baseline from first N% of samples
#     baseline = np.median(stimulus_data[:max(1, int(baseline_fraction * len(stimulus_data)))])
    
#     # Mark where stimulus differs significantly from baseline
#     threshold = tolerance * (np.max(stimulus_data) - np.min(stimulus_data)) + 1e-12
#     is_active = np.abs(stimulus_data - baseline) > threshold
    
#     # Find edges of stimulus period
#     edges = np.diff(is_active.astype(int), prepend=0, append=0)
#     starts = np.where(edges == 1)[0]
#     ends = np.where(edges == -1)[0]
    
#     # Return first contiguous stimulus period in seconds
#     if len(starts) > 0 and len(ends) > 0:
#         t_start = starts[0] / sampling_rate
#         t_end = ends[0] / sampling_rate
#         return t_start, t_end
    
#     return None, None


# def calculate_analysis_windows(stimulus_data, sampling_rate, pre_ms=4.5, post_ms=20.0, 
#                                 padding_factor=0.05, baseline_fraction=0.05, tolerance=0.05):
#     """
#     Calculate analysis windows based on stimulus bounds with padding.
    
#     Args:
#         stimulus_data: numpy array of stimulus/current values
#         sampling_rate: sampling rate in Hz
#         pre_ms: pre-stimulus window in milliseconds
#         post_ms: post-stimulus window in milliseconds
#         padding_factor: fraction of stimulus duration to use as padding on each side
#         baseline_fraction: fraction of data at start to use for baseline estimation
#         tolerance: multiplier for baseline to determine threshold
    
#     Returns:
#         dict with keys: 't_min', 't_max', 't_plot_min', 't_plot_max', 't_start', 't_end', 't_min_current', 't_max_current'
#         or None if stimulus not found
#     """
#     # Get stimulus bounds
#     t_stim_start, t_stim_end = calculate_stimulus_bounds_from_nwb(
#         stimulus_data, sampling_rate, baseline_fraction, tolerance
#     )
    
#     if t_stim_start is None or t_stim_end is None:
#         return None
    
#     # Convert milliseconds to seconds
#     pre_s = pre_ms / 1000.0
#     post_s = post_ms / 1000.0
    
#     # Calculate stimulus duration for padding
#     stim_duration = t_stim_end - t_stim_start
#     padding = stim_duration * padding_factor
    
#     # Define windows with padding
#     t_min = t_stim_start + padding
#     t_max = t_stim_end - padding
#     t_plot_min = max(0, t_min - 0.02)  # Add small margin for plotting
#     t_plot_max = t_max + 0.02
#     t_start = t_min - pre_s
#     t_end = t_max + post_s
#     t_min_current = t_start
#     t_max_current = t_end
    
#     return {
#         't_min': t_min,
#         't_max': t_max,
#         't_plot_min': t_plot_min,
#         't_plot_max': t_plot_max,
#         't_start': t_start,
#         't_end': t_end,
#         't_min_current': t_min_current,
#         't_max_current': t_max_current,
#         't_stim_start': t_stim_start,
#         't_stim_end': t_stim_end
#     }


def run_spike_detection(df, df_pA, df_analysis, fs, bundle_path, pA_was_replaced=False, analysis_windows=None, sweep_config=None):
    # Use parameters from centralized config
    pre_threshold_window_ms = PRE_THRESHOLD_WINDOW_MS
    post_threshold_window_ms = POST_THRESHOLD_WINDOW_MS
    threshold_percent = THRESHOLD_PERCENT
    fast_trough_percent = FAST_TROUGH_PERCENT

    #threshold window in seconds
    pre_s = PRE_THRESHOLD_WINDOW_S
    post_s = POST_THRESHOLD_WINDOW_S

    # Detect if mixed protocol
    p = Path(bundle_path)
    try:
        manifest = json.loads((p / "manifest.json").read_text())
        is_mixed = "stimulus" in manifest["tables"] and "response" in manifest["tables"]
    except:
        is_mixed = False

    # Store relative time offsets for mixed protocol
    relative_t_stim_start = None
    relative_t_stim_end = None
    
    # Build analysis_windows from sweep_config if available and not already provided
    if analysis_windows is None and sweep_config is not None:
        # For mixed protocol, we'll calculate windows per-sweep in the loop
        # For single protocol, extract a reference window now
        try:
            # Find first valid sweep for reference
            valid_sweep = None
            for sweep_id, sweep_data in sweep_config.get("sweeps", {}).items():
                if sweep_data.get("valid", False):
                    valid_sweep = sweep_id
                    break
            
            if valid_sweep is not None:
                windows = sweep_config["sweeps"][valid_sweep]["windows"]
                relative_t_stim_start = windows["stimulus_start_s"]
                relative_t_stim_end = windows["stimulus_end_s"]
                
                # Simplified: Only store stimulus times and ISI bin range
                # Per-sweep filtering uses sweep-specific times from sweep_config
                analysis_windows = {
                    't_stim_start': relative_t_stim_start,
                    't_stim_end': relative_t_stim_end
                }
                print(f"Using analysis windows from sweep_config.json:")
                print(f"  Protocol type: {'MIXED' if is_mixed else 'SINGLE'}")
                print(f"  Reference stimulus period: [{relative_t_stim_start:.6f}, {relative_t_stim_end:.6f}] s")
                if is_mixed:
                    print(f"  Note: For mixed protocol, each sweep uses its own timing from sweep_config")
        except (KeyError, TypeError) as e:
            print(f"WARNING: Failed to extract windows from sweep_config: {e}")
            print("  Falling back to hardcoded defaults")

    # Main analysis windows (in seconds)
    # analysis_windows is required to define spike detection parameters
    if analysis_windows is None:
        raise ValueError("analysis_windows is required for spike detection. "
                        "Must be provided or calculated from sweep_config.")
    
    t_stim_start = analysis_windows['t_stim_start']
    t_stim_end = analysis_windows['t_stim_end']
    print(f"Using stimulus windows:")
    print(f"  Stimulus period: [{t_stim_start:.6f}, {t_stim_end:.6f}] s")

    # Handle sampling rate for mixed vs single protocol
    # For mixed protocol: fs might be a list like ['200000.0', '50000.0']
    # For single protocol: fs is a single number
    if isinstance(fs, list):
        # Mixed protocol: use the maximum sampling rate as default
        # (will be overridden per-sweep later if sweep_config has per-sweep rates)
        fs_default = max([float(f) for f in fs])
        print(f"  ⚠ Multiple sampling rates detected: {fs}")
        print(f"  Using maximum rate {fs_default} Hz as default")
    else:
        # Single protocol: use the single rate
        fs_default = float(fs)
    
    # This will be used for plotting later
    pre_samples = int(pre_s * fs_default)
    post_samples = int(post_s * fs_default)

    # Don't pre-filter all data; instead filter per-sweep to handle mixed protocol time offsets
    peak_level_data = {} #{ap_index: {} for ap_index in range(200)}
    max_peaks_overall = 0
    sweep_results = []
    filtered_peaks = {}
    print("RUNNING SPIKE DETECTION")
    
    # Get unique sweeps
    unique_sweeps = sorted(df["sweep"].unique())
    
    # Load manifest to get per-sweep sampling rates (ONLY for mixed protocol)
    per_sweep_rates = {}
    if is_mixed:
        try:
            protocols = manifest.get("protocols", {})
            for sweep_id, protocol_data in protocols.items():
                per_sweep_rates[int(sweep_id)] = float(protocol_data.get("rate", fs_default))
            print(f"  ✓ Loaded per-sweep sampling rates for {len(per_sweep_rates)} sweeps")
        except Exception as e:
            print(f"  ⚠ Could not load per-sweep rates: {e}")
            print(f"  Using default rate {fs_default} Hz for all sweeps")
    
    for sweep_number in unique_sweeps:
        # FOR DEBUGGING, IGNORE
        # if sweep_number != 10:
        #      continue
        print("SWEEP NUMBER:", sweep_number)
        
        # Get sampling rate for this sweep (ONLY different for mixed protocol)
        if is_mixed and sweep_number in per_sweep_rates:
            sweep_fs = per_sweep_rates[sweep_number]
            print(f"  Using sweep-specific rate: {sweep_fs} Hz")
            # Recalculate samples for this sweep's rate
            sweep_pre_samples = int(pre_s * sweep_fs)
            sweep_post_samples = int(post_s * sweep_fs)
        else:
            # Single protocol: use the default rate and pre-calculated samples
            sweep_fs = fs_default
            sweep_pre_samples = pre_samples
            sweep_post_samples = post_samples
        
        # Get data for this sweep
        group = df[df["sweep"] == sweep_number].sort_values("t_s")
        
        # For mixed protocol, get THIS SWEEP's specific time windows from sweep_config
        if is_mixed and sweep_config is not None:
            try:
                # Get this sweep's config
                sweep_cfg = sweep_config["sweeps"].get(str(sweep_number), {})
                if sweep_cfg and "windows" in sweep_cfg:
                    # For mixed protocol: sweep_config already contains ABSOLUTE times
                    # No need to add offset - use times directly
                    sweep_t_stim_start = sweep_cfg["windows"]["stimulus_start_s"]
                    sweep_t_stim_end = sweep_cfg["windows"]["stimulus_end_s"]
                    
                    # Use absolute times directly
                    sweep_t_min = sweep_t_stim_start
                    sweep_t_max = sweep_t_stim_end
                    sweep_t_plot_min = max(sweep_t_stim_start - 0.1, sweep_t_stim_start - 0.02)
                    sweep_t_plot_max = sweep_t_stim_end + 0.02
                    sweep_t_start = sweep_t_stim_start - pre_s
                    sweep_t_end = sweep_t_stim_end + post_s
                else:
                    # Fallback to stimulus times
                    sweep_t_min = t_stim_start
                    sweep_t_max = t_stim_end
                    sweep_t_plot_min = max(t_stim_start - 0.1, t_stim_start - 0.02)
                    sweep_t_plot_max = t_stim_end + 0.02
                    sweep_t_start = t_stim_start - pre_s
                    sweep_t_end = t_stim_end + post_s
            except (KeyError, TypeError):
                # If anything goes wrong, use stimulus times
                sweep_t_min = t_stim_start
                sweep_t_max = t_stim_end
                sweep_t_plot_min = max(t_stim_start - 0.1, t_stim_start - 0.02)
                sweep_t_plot_max = t_stim_end + 0.02
                sweep_t_start = t_stim_start - pre_s
                sweep_t_end = t_stim_end + post_s
        else:
            # Single protocol: use stimulus times
            sweep_t_min = t_stim_start
            sweep_t_max = t_stim_end
            sweep_t_plot_min = max(t_stim_start - 0.1, t_stim_start - 0.02)
            sweep_t_plot_max = t_stim_end + 0.02
            sweep_t_start = t_stim_start - pre_s
            sweep_t_end = t_stim_end + post_s
        
        # Filter this sweep's data to the analysis window
        group_window = group[(group["t_s"] >= sweep_t_min) & (group["t_s"] <= sweep_t_max)].reset_index(drop=True)
        
        # Skip if no data in this window
        if len(group_window) == 0:
            dbg(f"  WARNING: No data in analysis window for sweep {sweep_number}, skipping.")
            continue
        
        group = group_window.sort_values("t_s")
        time = group["t_s"].to_numpy()
        voltage = group["value"].to_numpy()

        # dV/dt in mV/ms
        #dvdt = np.gradient(voltage, time) * 1000

        # Detect peaks in this sweep
        peaks, props = find_peaks(
            voltage,
            height=PEAK_HEIGHT_THRESHOLD,
            prominence=PEAK_PROMINENCE
        )

        # Filter out peaks that are too close together (within 2ms of previous peak)
        # This removes duplicate detections from noise
        filtered_peaks_list = []
        
        for peak_idx in peaks:
            t_peak = time[int(peak_idx)]
            # Check if this peak is far enough from the last accepted peak
            if not filtered_peaks_list or (t_peak - time[int(filtered_peaks_list[-1])]) >= MIN_PEAK_DISTANCE_S:
                filtered_peaks_list.append(peak_idx)
        
        peaks = np.array(filtered_peaks_list)

        valid_peaks = []
        valid_spike_segments = []

        peak_voltages = []
        threshold_voltages = []
        threshold_to_peak_voltages = []
        threshold_to_peak_times_ms = []
        fast_trough_voltages = []
        upstroke_to_peak_times = []
        upstroke_to_peak_voltages = []
        peak_to_downstroke_times = []
        peak_to_trough_times_ms = [] 
        peak_to_trough_heights_mV = [] 
        half_heights = []
        heights = []
        ap_widths = []
        threshold_fast_trough_widths = []
        upstrokes = []
        downstrokes = []
        up_down_ratios = []
        isi_ms = []

        for i, peak in enumerate(peaks):
            # FOR DEBUGGING, IGNORE
            # if i !=42:
            #     continue
            peak = int(peak)
            t_peak = float(time[peak])
            v_peak = float(voltage[peak])
            print(f"  peak {i}: index {peak}, t={t_peak:.6f}s, v={v_peak:.2f}mV")

            # Exclude peaks that fall outside the stimulus window for THIS sweep
            # Note: Peaks are from the sweep-specific window, so should be within it
            if (t_peak < sweep_t_min) or (t_peak > sweep_t_max):
                print(f"    INFO: peak at {t_peak:.6f}s outside stimulus window [{sweep_t_min:.6f}, {sweep_t_max:.6f}], skipping.")
                continue

            # ---------------------------
            # Define Window 1 (upstroke)
            #   from max(t_peak - 4.5ms, sweep_start)
            #   up to the peak
            # Define Window 2 (downstroke)
            #   from peak to next peak
            #   or peak to t_peak + 20ms if last peak
            # ---------------------------
            t_start_w1 = max(time[peak] - pre_s, float(time[0])) # whichever one is higher

            if i < len(peaks) - 1:
                # Not last peak: window 2 ends at next peak
                next_peak_idx = int(peaks[i + 1])
                #t_end_w2 = time[peak] + post_s
                #w2_end_idx = int(np.searchsorted(time, t_end, side="right"))
                w2_end_idx = next_peak_idx
            else:
                # Last peak: window 2 ends 20 ms after peak or end of sweep
                t_end_w2 = min(time[peak] + post_s, float(time[-1]))
                w2_end_idx   = int(np.searchsorted(time, t_end_w2, side="right"))

            # Convert times to indices
            w1_start_idx = int(np.searchsorted(time, t_start_w1, side="left"))
            w1_end_idx   = peak + 1
            w2_start_idx = peak + 1

            if w1_start_idx >= w1_end_idx or w2_start_idx >= w2_end_idx:
                dbg("    WARNING: invalid window for this peak, skipping.")
                continue

            # Extract segments
            t_up = time[w1_start_idx:w1_end_idx]
            v_up = voltage[w1_start_idx:w1_end_idx]
            dvdt_up = np.gradient(v_up, t_up) * 1000
            #dvdt_up = dvdt[w1_start_idx:w1_end_idx]

            t_down = time[w2_start_idx:w2_end_idx]
            v_down = voltage[w2_start_idx:w2_end_idx]
            dvdt_down = np.gradient(v_down, t_down) * 1000
            #dvdt_down = dvdt[w2_start_idx:w2_end_idx]

            # Segment for averaging (use sweep-specific sample counts)
            plot_start = peak - sweep_pre_samples
            plot_end   = peak + sweep_post_samples
            
            segment = None
            if plot_start >= 0 and plot_end < len(voltage):
                segment = voltage[plot_start:plot_end]
            else:
                print(f"    WARNING: Cannot form segment for averaging - peak too close to sweep boundary.")
                print(f"             Peak needs {pre_threshold_window_ms}ms before and {post_threshold_window_ms}ms after for waveform extraction.")
                print(f"             This peak will still be analyzed for all other metrics (threshold, width, etc.).")

            # Initialize metrics for this peak
            t_threshold = np.nan
            v_threshold = np.nan
            t_upstroke  = np.nan
            v_upstroke  = np.nan
            t_fast_trough = np.nan
            v_fast_trough = np.nan
            t_downstroke  = np.nan
            v_downstroke  = np.nan
            height = np.nan
            half_height = np.nan
            width = np.nan
            threshold_fast_trough_width = np.nan
            peak_max_dvdt_time_difference = np.nan
            peak_max_dvdt_voltage_difference = np.nan
            min_dvdt_peak_time_difference = np.nan
            max_dvdt_value = np.nan
            min_dvdt_value = np.nan
            upstroke_downstroke_ratio = np.nan

            # -------------------------
            # Window 1: threshold & upstroke
            # -------------------------
            if len(dvdt_up) > 0:
                # Max upstroke in W1
                max_dvdt_value = float(np.max(dvdt_up))
                up_rel_idx = int(np.argmax(dvdt_up))
                up_idx = w1_start_idx + up_rel_idx
                t_upstroke = float(time[up_idx])
                v_upstroke = float(voltage[up_idx])

                peak_max_dvdt_time_difference = t_peak - t_upstroke
                peak_max_dvdt_voltage_difference = v_peak - v_upstroke

                # Threshold: last time dv/dt is still below threshold% of max
                thr_value = threshold_percent * max_dvdt_value
                below = np.where(dvdt_up >= thr_value)[0]
                if len(below) > 0:
                    thr_rel = int(below[0])
                    threshold_idx = w1_start_idx + thr_rel
                    t_threshold = float(time[threshold_idx])
                    v_threshold = float(voltage[threshold_idx])

                    if v_peak - v_threshold < MIN_PEAK_THRESHOLD_AMPLITUDE_MV:
                        print(f"    WARNING: Peak - Threshold difference for peak {i} is {v_peak - v_threshold:.2f} mV < {MIN_PEAK_THRESHOLD_AMPLITUDE_MV} mV, skipping.")
                        continue 
                else:
                    print("    WARNING: no dv/dt below threshold% of max in upstroke window; threshold=NaN.")
            else:
                print("    WARNING: empty dvdt_up for this peak.")

            # -------------------------
            # Window 2: downstroke & fast trough
            # -------------------------
            if len(dvdt_down) > 0:
                # Max downstroke (most negative dv/dt)
                min_dvdt_value = float(np.min(dvdt_down))
                #this will reset the range
                down_rel_idx = int(np.argmin(dvdt_down))
                #correct the range
                down_idx = w2_start_idx + down_rel_idx
                t_downstroke = float(time[down_idx])
                v_downstroke = float(voltage[down_idx])
                min_dvdt_peak_time_difference = t_downstroke - t_peak

                # Fast trough: first time dv/dt recovers above 1% of min downstroke
                fast_line = fast_trough_percent * min_dvdt_value

                # Narrow range for getting fast trough
                dvdt_after_downstroke = dvdt_down[down_rel_idx:]
                crossing = np.where(dvdt_after_downstroke >= fast_line)[0]
                if len(crossing) > 0:
                    trough_rel = int(crossing[0])
                    fast_trough_idx = down_idx + trough_rel 
                    t_fast_trough = float(time[fast_trough_idx])
                    v_fast_trough = float(voltage[fast_trough_idx])
                else:
                    next_best_idx = int(np.argmax(dvdt_after_downstroke))
                    fast_trough_idx = down_idx + next_best_idx
                    t_fast_trough = float(time[fast_trough_idx])
                    v_fast_trough = float(voltage[fast_trough_idx])
                    dbg(f"    WARNING: No 1% crossing found; using closest dv/dt value to threshold which is: {t_fast_trough} ms, {v_fast_trough} mV")
            else:
                dbg("    WARNING: empty dvdt_down for this peak.")

            # -------------------------
            # Amplitude, half-height & AP width
            # -------------------------
            if not np.isnan(v_fast_trough):
                height = v_peak - v_fast_trough
                half_height = v_fast_trough + 0.5 * height

                # Half-height crossings:
                # upstroke side (in W1)
                upstroke_segment = voltage[w1_start_idx:peak + 1]
                up_indices = np.where(upstroke_segment >= half_height)[0]
                t_start_hh = np.nan
                t_end_hh = np.nan

                if len(up_indices) > 0:
                    up_hh_idx = w1_start_idx + int(up_indices[0])
                    t_start_hh = float(time[up_hh_idx])
                else:
                    dbg("    WARNING: No upstroke half-height crossing.")

                # downstroke side (in W2)
                downstroke_segment = voltage[peak:w2_end_idx]
                down_indices = np.where(downstroke_segment <= half_height)[0]
                if len(down_indices) > 0:
                    down_hh_idx = peak + int(down_indices[0])
                    t_end_hh = float(time[down_hh_idx])
                else:
                    dbg("    WARNING: No downstroke half-height crossing.")

                if not np.isnan(t_start_hh) and not np.isnan(t_end_hh):
                    width = (t_end_hh - t_start_hh) * 1000.0  # ms
                else:
                    dbg("    WARNING: Could not compute AP width (half-height).")

            else:
                dbg("    WARNING: cannot compute height/half-height (fast trough NaN).")

            # Threshold → fast trough width
            if not np.isnan(t_threshold) and not np.isnan(t_fast_trough):
                threshold_fast_trough_width = (t_fast_trough - t_threshold) * 1000.0

            # Peak → fast trough distance (time in ms and voltage in mV)
            peak_to_trough_time_ms = np.nan
            peak_to_trough_height_mV = np.nan
            # print("time of fast trough:", t_fast_trough)
            # print("trough - peak time difference:", t_fast_trough - t_peak)
            if not np.isnan(t_peak) and not np.isnan(t_fast_trough):
                peak_to_trough_time_ms = (t_fast_trough - t_peak) * 1000.0  # convert to ms
            if not np.isnan(v_peak) and not np.isnan(v_fast_trough):
                peak_to_trough_height_mV = v_peak - v_fast_trough  # positive value: how far below peak is the trough

            # Upstroke / downstroke ratio
            if not np.isnan(max_dvdt_value) and not np.isnan(min_dvdt_value) and min_dvdt_value != 0:
                upstroke_downstroke_ratio = max_dvdt_value / abs(min_dvdt_value)

            # Collect per-peak metrics
            peak_voltages.append(v_peak)
            threshold_voltages.append(v_threshold)
            threshold_to_peak_voltages.append(v_peak - v_threshold if not np.isnan(v_threshold) else np.nan)
            threshold_to_peak_times_ms.append((t_peak - t_threshold) * 1000.0 if not np.isnan(t_threshold) else np.nan)  # NEW: time from threshold to peak in ms
            fast_trough_voltages.append(v_fast_trough)
            upstroke_to_peak_times.append(peak_max_dvdt_time_difference)
            upstroke_to_peak_voltages.append(peak_max_dvdt_voltage_difference)
            peak_to_downstroke_times.append(min_dvdt_peak_time_difference)
            peak_to_trough_times_ms.append(peak_to_trough_time_ms)  
            peak_to_trough_heights_mV.append(peak_to_trough_height_mV)  
            half_heights.append(half_height)
            heights.append(height)
            threshold_fast_trough_widths.append(threshold_fast_trough_width)
            ap_widths.append(width)
            upstrokes.append(max_dvdt_value)
            downstrokes.append(min_dvdt_value)
            up_down_ratios.append(upstroke_downstroke_ratio)
            # If we have gotten this far, it has passed all checks and is a valid peak
            valid_peaks.append(peak)  # Store original peak index for ISI calculation
            # Only peaks that survived the filters (valid peaks) contribute to the segment
            if segment is not None:
                valid_spike_segments.append(segment)

        # Store peak TIMES (in absolute seconds), not indices, for robust plotting
        # This avoids index confusion between filtered and full sweep data
        valid_peak_times = time[valid_peaks] if len(valid_peaks) > 0 else []
        filtered_peaks[sweep_number] = (valid_peaks, valid_peak_times)
        max_peaks_overall = max(max_peaks_overall, len(valid_peaks))

        # Morphology on how AP changes
        ratio_middle_first_width = np.nan
        ratio_middle_first_threshold_to_peak = np.nan
        ratio_middle_first_fast_trough = np.nan
        ratio_last_first_width = np.nan
        ratio_last_first_threshold_to_peak = np.nan
        ratio_last_first_fast_trough = np.nan

        if len(valid_peaks) >=3:
            def get_peak_metrics(idx: int):
                return (
                    ap_widths[idx],
                    threshold_to_peak_voltages[idx],
                    fast_trough_voltages[idx],
                )
            
            first_idx = 0
            middle_idx = floor(len(valid_peaks)/2)
            last_idx = -1

            first_width, first_thr2peak, first_trough = get_peak_metrics(first_idx)
            mid_width,   mid_thr2peak,   mid_trough   = get_peak_metrics(middle_idx)
            last_width,  last_thr2peak,  last_trough  = get_peak_metrics(last_idx)

            def safe_ratio(num, den):
                return num / den if (den is not None and den != 0 and not np.isnan(den)) else np.nan

            ratio_middle_first_width = safe_ratio(mid_width,first_width)
            ratio_middle_first_threshold_to_peak = safe_ratio(mid_thr2peak,first_thr2peak)
            ratio_middle_first_fast_trough = safe_ratio(mid_trough,first_trough)

            ratio_last_first_width = safe_ratio(last_width,first_width)
            ratio_last_first_threshold_to_peak = safe_ratio(last_thr2peak,first_thr2peak)
            ratio_last_first_fast_trough = safe_ratio(last_trough,first_trough)

        # Get the mean isi
        if len(valid_peaks) >= 2: 
            peak_times = time[valid_peaks] * 1000 # convert to ms 
            isi_ms = np.diff(peak_times) 
            mean_isi_ms = float(np.mean(isi_ms)) 
            if mean_isi_ms > 0:
                cv_isi = float(np.std(isi_ms) / mean_isi_ms)
            else:
                cv_isi = np.nan
        else: 
            isi_ms = np.array([]) 
            mean_isi_ms = np.nan
            cv_isi = np.nan  

        # Now get isi via binning
        bin_isi = []
        bin_cv_isi = []
        peak_times_full = time[valid_peaks] * 1000   # ms
        bin_width_ms = 50
        # Use sweep-specific stimulus window for binning
        t_start_ms = sweep_t_min * 1000 
        t_end_ms   = sweep_t_max * 1000
        window_ms = (t_end_ms - t_start_ms) 
        n_bins = int(np.ceil(window_ms / bin_width_ms)) 
        dbg(f"number of bins:{n_bins}")
        if len(valid_peaks) >= 2:
            bin_edges = np.arange(t_start_ms, t_start_ms + (n_bins + 1) * bin_width_ms, bin_width_ms)
            dbg(f"bin edges: {bin_edges}")
            bin_indices = np.digitize(peak_times_full, bin_edges) - 1
            dbg(f"bin indices: {bin_indices}")

            for b in range(n_bins):
                peaks_in_bin = peak_times[bin_indices == b]
                
                if len(peaks_in_bin) >= 2:
                    isi = np.diff(peaks_in_bin)
                    mean_isi = float(np.mean(isi))
                    cv = float(np.std(isi) / mean_isi) if mean_isi > 0 else np.nan
                else:
                    mean_isi = np.nan
                    cv = np.nan
                
                bin_isi.append(mean_isi)
                bin_cv_isi.append(cv)

        if len(bin_isi) == 0:
            bin_isi = [np.nan] * n_bins
            bin_cv_isi = [np.nan] * n_bins
        isi_cols = {f"bin_{i}_isi_ms": bin_isi[i] for i in range(n_bins)}
        cv_cols  = {f"bin_{i}_cv_isi": bin_cv_isi[i] for i in range(n_bins)}
     
        # ----------------------------------------------------
        # Construct averaged spike per sweep & plot with:
        #   - Window 1: -4.5 ms to 0 ms (pre-peak shaded)
        #   - Window 2: 0 ms to 20ms (post-peak shaded)
        #   - Mark threshold, upstroke, peak, fast trough, downstroke
        #   - Matching dV/dt plot
        # ----------------------------------------------------
        plot_dir = Path(bundle_path) / "Averaged_Peaks_Per_Sweep"
        plot_dir.mkdir(parents=True, exist_ok=True)

        if valid_spike_segments:
            average_spike = np.mean(valid_spike_segments, axis=0)
            spike_time = np.linspace(
                -pre_threshold_window_ms,
                post_threshold_window_ms,
                average_spike.shape[0],
            )  #This is in ms

            # Peak
            peak_idx = int(np.argmax(average_spike))
            peak_time_ms = spike_time[peak_idx]
            peak_val = average_spike[peak_idx]

            # Recenter time so that peak is at 0 ms
            spike_time = spike_time - peak_time_ms
            peak_time_ms = 0.0 

            #averaged_dvdt = np.gradient(average_spike, spike_time)

            # Upstroke: max dV/dt before peak
            # 1. Extract pre-peak region
            pre_peak_voltages = average_spike[:peak_idx + 1]
            pre_peak_times = spike_time[:peak_idx + 1]

            # 2. Compute dV/dt for pre-peak segment only
            if len(pre_peak_times) >= 2:
                up_region_dvdt = np.gradient(pre_peak_voltages, pre_peak_times)
            else:
                up_region_dvdt = np.array([])

            # 3. Initialize upstroke outputs
            up_idx = np.nan
            upstroke_time_ms = np.nan
            upstroke_val = np.nan
            max_upstroke = np.nan
            threshold_idx_avg = np.nan
            threshold_time_ms = np.nan
            threshold_val = np.nan
            threshold_dvdt_val = np.nan

            # 4. Find max upstroke (local max dV/dt before peak)
            if len(up_region_dvdt) > 0:
                up_idx = int(np.argmax(up_region_dvdt))
                upstroke_time_ms = pre_peak_times[up_idx]
                upstroke_val = pre_peak_voltages[up_idx]
                max_upstroke = up_region_dvdt[up_idx]

            # 5. Threshold (e.g., 5% of max upstroke)
            if not np.isnan(max_upstroke) and max_upstroke > 0:
                thr_line = threshold_percent * max_upstroke  # e.g., 0.05 * max_upstroke
                # Find *first* time the rising dv/dt crosses the threshold line
                cross = np.where(up_region_dvdt >= thr_line)[0]
                if len(cross) > 0:
                    threshold_idx_avg = int(cross[0])
                    threshold_time_ms = pre_peak_times[threshold_idx_avg]
                    threshold_val = pre_peak_voltages[threshold_idx_avg]
                    threshold_dvdt_val = up_region_dvdt[threshold_idx_avg]


            # Downstroke & fast trough on averaged spike
            # 1. Extract post-peak region
            post_peak_voltages = average_spike[peak_idx:]
            post_peak_times    = spike_time[peak_idx:]

            # 2. Compute segmented dV/dt for post-peak region
            if len(post_peak_times) >= 2:
                down_region_dvdt = np.gradient(post_peak_voltages, post_peak_times)
            else:
                down_region_dvdt = np.array([])

            # 3. Initialize fast-trough outputs
            min_downstroke_local_idx = np.nan
            min_downstroke_idx       = np.nan
            min_downstroke_val       = np.nan
            min_downstroke_time_ms   = np.nan
            min_downstroke_v_val     = np.nan
            fast_trough_local_idx    = np.nan
            fast_trough_idx          = np.nan
            fast_trough_time_ms      = np.nan
            fast_trough_val          = np.nan
            fast_trough_dvdt         = np.nan

            # 4. Minimum `dV/dt` (max downstroke) after peak
            if len(down_region_dvdt) > 0:

                min_downstroke_local_idx = int(np.argmin(down_region_dvdt))
                min_downstroke_val       = down_region_dvdt[min_downstroke_local_idx]

                # Convert to global index for plotting on full waveform
                min_downstroke_idx       = peak_idx + min_downstroke_local_idx

                # Time / voltage values at that point
                min_downstroke_time_ms   = post_peak_times[min_downstroke_local_idx]
                min_downstroke_v_val     = post_peak_voltages[min_downstroke_local_idx]

            # 5. Fast trough: first time `dV/dt` recovers above 1% of min downstroke
            if not np.isnan(min_downstroke_val):
                fast_line = fast_trough_percent * min_downstroke_val
                region_after_trough = down_region_dvdt[min_downstroke_local_idx:]
                post_cross = np.where(region_after_trough >= fast_line)[0]

                if len(post_cross) > 0:
                    fast_trough_local_idx = int(post_cross[0]) + min_downstroke_local_idx
                    fast_trough_idx       = peak_idx + fast_trough_local_idx
                    fast_trough_time_ms   = spike_time[fast_trough_idx]
                    fast_trough_val       = average_spike[fast_trough_idx]
                    fast_trough_dvdt      = down_region_dvdt[int(fast_trough_local_idx)]
                else:
                    dbg("WARNING: Fast trough undefined (no 1% upward crossing).")
                    if np.isnan(fast_trough_val):
                        dbg(f"Saving debug plot for fast-trough failure in sweep {sweep_number}")
                        fig_dbg, ax_dbg = plt.subplots(figsize=(4, 4))
                        ax_dbg.plot(post_peak_times, down_region_dvdt, '-k')
                        ax_dbg.axhline(fast_line, color="red", linestyle="--", label="1% line")
                        ax_dbg.axvline(min_downstroke_time_ms, color="purple", linestyle="--", label="Min downstroke")
                        ax_dbg.set_title(f"Sweep {sweep_number}: Fast Trough Debug")
                        ax_dbg.set_xlabel("Time (ms)")
                        ax_dbg.set_ylabel("dV/dt")
                        ax_dbg.legend(fontsize=6)
                        debug_dir = plot_dir / "debug_fasttrough"
                        debug_dir.mkdir(exist_ok=True)
                        fig_dbg.savefig(debug_dir / f"sweep_{sweep_number}_fasttrough_debug.png")
                        plt.close(fig_dbg)
                        # Getting next best value for fast trough
                        next_best_idx       = int(np.argmax(down_region_dvdt))
                        fast_trough_idx     = peak_idx + next_best_idx
                        fast_trough_time_ms = spike_time[fast_trough_idx]
                        fast_trough_val     = average_spike[fast_trough_idx]
                        fast_trough_dvdt    = down_region_dvdt[int(next_best_idx)]
                        dbg(f"Using next best value for fast trough, which is {fast_trough_time_ms} ms from the peak and at {fast_trough_val} mV")

            dbg(
                f"Sweep {sweep_number} summary:"
                f"\n  Peak: {peak_val:.2f} mV"
                f"\n  Threshold: {threshold_val if not np.isnan(threshold_val) else 'None'}"
                f"\n  Upstroke max dV/dt: {max_upstroke if not np.isnan(max_upstroke) else 'None'}"
                f"\n  Min downstroke dV/dt: {min_downstroke_val if not np.isnan(min_downstroke_val) else 'None'}"
                f"\n  Fast trough: {fast_trough_val if fast_trough_val is not None else 'None'}"
            )
            # Threshold must occur before peak
            if not np.isnan(threshold_time_ms) and threshold_time_ms > 0:
                dbg("WARNING: Threshold detected *after* peak (unexpected).")

            # Upstroke slope should be positive
            if not np.isnan(max_upstroke) and max_upstroke < 0:
                dbg("WARNING: Max upstroke dV/dt is negative - check derivative sign or sampling.")

            # Downstroke slope should be negative
            if not np.isnan(min_downstroke_val) and min_downstroke_val > 0:
                dbg("WARNING: Min downstroke dV/dt is positive - unexpected repolarization behavior.")

            # Fast trough should occur AFTER minimum downstroke
            if (not np.isnan(fast_trough_time_ms)
                and not np.isnan(min_downstroke_time_ms)
                and fast_trough_time_ms < min_downstroke_time_ms):
                dbg("WARNING: Fast trough occurs *before* min downstroke - check segmentation.")

            # Plot averaged spike & dV/dt with shaded windows
            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(6, 6),
                sharex=True,
                gridspec_kw={"height_ratios": [2, 1]},
            )

            # Window 1 & 2 in ms
            w1_start_ms = -pre_threshold_window_ms
            w1_end_ms   = 0.0
            w2_start_ms = 0.0
            w2_end_ms   = post_threshold_window_ms
            
            # Build full-length dV/dt for plotting only (NOT for analysis)
            full_dvdt = np.full_like(spike_time, np.nan, dtype=float)

            # Pre-peak dV/dt
            if len(up_region_dvdt) > 0:
                full_dvdt[:peak_idx+1] = up_region_dvdt

            # Post-peak dV/dt
            if len(down_region_dvdt) > 0:
                full_dvdt[peak_idx:] = down_region_dvdt

            # Vm plot
            ax1.plot(spike_time, average_spike, color="k", label="Averaged Spike")
            ax1.axvline(0, color="gray", linestyle="--", label="Peak align")
            ax1.axvspan(w1_start_ms, w1_end_ms, alpha=0.2, color="orange", label="Window 1 (pre-peak)")
            ax1.axvspan(w2_start_ms, w2_end_ms, alpha=0.2, color="cyan",   label="Window 2 (post-peak)")

            def safe_scatter(ax, x, y, color, label=None):
                if x is None or y is None: return
                if np.isnan(x) or np.isnan(y): return
                ax.scatter(x, y, color=color, label=label)

            safe_scatter(ax1, peak_time_ms, peak_val,              "r", label="Peak")
            safe_scatter(ax1, threshold_time_ms, threshold_val,    "orange", label="Threshold")
            safe_scatter(ax1, upstroke_time_ms, upstroke_val,      "g", label="Max Upstroke")
            safe_scatter(ax1, fast_trough_time_ms, fast_trough_val,"b", label="Fast Trough")
            safe_scatter(ax1, min_downstroke_time_ms, min_downstroke_v_val, "purple", label="Min Downstroke")

            ax1.set_ylabel("Voltage (mV)")
            ax1.set_title(f"Averaged Spike for Sweep {sweep_number}")
            ax1.legend(loc="upper right", fontsize=8, frameon=True)

            # dV/dt plot
            ax2.plot(spike_time, full_dvdt, color="purple", label="dV/dt")
            ax2.axvline(0, color="gray", linestyle="--")
            ax2.axvspan(w1_start_ms, w1_end_ms, alpha=0.2, color="orange")
            ax2.axvspan(w2_start_ms, w2_end_ms, alpha=0.2, color="cyan")


            safe_scatter(ax2, threshold_time_ms, threshold_dvdt_val,      "orange")
            safe_scatter(ax2, upstroke_time_ms, up_region_dvdt[int(up_idx)] if up_idx is not None else np.nan, "g")
            safe_scatter(ax2, peak_time_ms, 0,                             "r")   # dV/dt ≈ 0 at peak
            safe_scatter(ax2, fast_trough_time_ms, fast_trough_dvdt,       "b")
            safe_scatter(ax2, min_downstroke_time_ms, min_downstroke_val,  "purple")

            ax2.set_xlabel("Time Difference from Peak (ms)")
            ax2.set_ylabel("dV/dt (mV/ms)")
            ax2.legend(loc="best", fontsize=8)

            plt.tight_layout()
            out_dir = Path(bundle_path) / "Averaged_Peaks_Per_Sweep"
            out_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_dir / f"averaged_peaks_for_sweep_{sweep_number}.jpeg")
            plt.close()
            #plt.show()
        else:
            print(f"    WARNING: No valid spikes for sweep {sweep_number}, skipping averaged plot.")

        
        # ----------------- Aggregate per sweep -----------------
        if len(peak_voltages) > 0:
            peak_level_data[sweep_number] = {
                "AP_Location_ms": time[valid_peaks].tolist(),
                "AP_Height_mV": peak_voltages,
                "AP_Half_Height_mV": half_heights,
                "AP_Width_ms": ap_widths,
                "AP_Threshold_mV": threshold_voltages,
                "AP_Threshold_to_Peak_Voltage_mV": threshold_to_peak_voltages,
                "AP_Threshold_to_Peak_Time_ms": threshold_to_peak_times_ms,
                "AP_Fast_Trough_mV": fast_trough_voltages,
                "AP_Upstroke_to_Peak_Time_ms": upstroke_to_peak_times,
                "AP_Upstroke_to_Peak_Voltage_mV": upstroke_to_peak_voltages,
                "AP_Peak_to_Downstroke_Time_ms": peak_to_downstroke_times,
                "AP_Peak_to_Fast_Trough_Time_ms": peak_to_trough_times_ms,
                "AP_Peak_to_Fast_Trough_Height_mV": peak_to_trough_heights_mV,
                "AP_Threshold_to_Fast_Trough_Width_ms" : threshold_fast_trough_widths,
                "AP_Upstroke_Downstroke_Ratio": up_down_ratios,
                "AP_Upstroke_mVms":upstrokes,
                "AP_Downstroke_mVms": downstrokes,
                "ISI_ms": ([np.nan] + isi_ms.tolist())
            }
            sweep_results.append(
                {
                    "sweep": sweep_number,
                    "avg_peak_voltage_mV": np.nanmean(peak_voltages),
                    "avg_threshold_voltage_mV": np.nanmean(threshold_voltages),
                    "avg_threshold_to_peak_mV": np.nanmean(threshold_to_peak_voltages),
                    "avg_threshold_to_peak_ms": np.nanmean(threshold_to_peak_times_ms), 
                    "avg_fast_trough_at_hyperpolarization_mV": np.nanmean(fast_trough_voltages),
                    "avg_upstroke_to_peak_ms": np.nanmean(upstroke_to_peak_times),
                    "avg_upstroke_to_peak_mV": np.nanmean(upstroke_to_peak_voltages),
                    "avg_peak_to_downstroke_ms": np.nanmean(peak_to_downstroke_times),
                    "avg_peak_to_trough_ms": np.nanmean(peak_to_trough_times_ms), 
                    "avg_peak_to_trough_mV": np.nanmean(peak_to_trough_heights_mV),  
                    "avg_height_mV": np.nanmean(heights),
                    "avg_half_height_mV": np.nanmean(half_heights),
                    "avg_ap_width_ms": np.nanmean(ap_widths),
                    "avg_threshold_fast_trough_width_ms": np.nanmean(threshold_fast_trough_widths),
                    "avg_upstroke_downstroke_ratio": np.nanmean(up_down_ratios),
                    "avg_upstroke_mVms":np.nanmean(upstrokes),
                    "avg_downstroke_mVms": np.nanmean(downstrokes),
                    "spike_frequency_Hz": len(peak_voltages) / (sweep_t_max - sweep_t_min),
                    "mean_isi_ms": mean_isi_ms,
                    "cv_isi": cv_isi,
                    **isi_cols,
                    **cv_cols,
                    "middle_first_AP_width_ratio": ratio_middle_first_width,
                    "middle_first_AP_peak_threshold_ratio": ratio_middle_first_threshold_to_peak,
                    "middle_first_AP_fast_trough_ratio": ratio_middle_first_fast_trough,
                    "last_first_AP_width_ratio": ratio_last_first_width,
                    "last_first_AP_peak_threshold_ratio": ratio_last_first_threshold_to_peak,
                    "last_first_AP_fast_trough_ratio": ratio_last_first_fast_trough
                }

            )
        else:
            sweep_results.append(
                {
                    "sweep": sweep_number,
                    "avg_peak_voltage_mV": np.nan,
                    "avg_threshold_voltage_mV": np.nan,
                    "avg_threshold_to_peak_mV": np.nan,
                    "avg_threshold_to_peak_ms": np.nan,  
                    "avg_fast_trough_at_hyperpolarization_mV": np.nan,
                    "avg_upstroke_to_peak_ms": np.nan,
                    "avg_upstroke_to_peak_mV": np.nan,
                    "avg_peak_to_downstroke_ms": np.nan,
                    "avg_peak_to_trough_ms": np.nan,  
                    "avg_peak_to_trough_mV": np.nan,  
                    "avg_height_mV": np.nan,
                    "avg_half_height_mV": np.nan,
                    "avg_ap_width_ms": np.nan,
                    "avg_threshold_fast_trough_width_ms": np.nan,
                    "avg_upstroke_downstroke_ratio": np.nan,
                    "avg_upstroke_mVms":np.nan,
                    "avg_downstroke_mVms": np.nan,
                    "spike_frequency_Hz": 0.0,
                    "mean_isi_ms": mean_isi_ms,
                    "cv_isi": cv_isi,
                    **isi_cols,
                    **cv_cols,
                    "middle_first_AP_width_ratio": np.nan,
                    "middle_first_AP_peak_threshold_ratio": np.nan,
                    "middle_first_AP_fast_trough_ratio": np.nan,
                    "last_first_AP_width_ratio": np.nan,
                    "last_first_AP_peak_threshold_ratio": np.nan,
                    "last_first_AP_fast_trough_ratio": np.nan
                }
            )

    # # ----------------- Current per sweep -----------------
    # Use stimulus_level_pA from sweep_config for each sweep (most reliable method)
    # This avoids noise in the actual measured current
    valid_sweeps = df["sweep"].unique()
    
    if sweep_config is not None:
        # Extract stimulus_level_pA from sweep_config for each valid sweep
        avg_pa_per_sweep = []
        for sweep_id in valid_sweeps:
            try:
                stimulus_level = sweep_config["sweeps"][str(sweep_id)].get("stimulus_level_pA", np.nan)
                avg_pa_per_sweep.append({"sweep": sweep_id, "avg_injected_current_pA": stimulus_level})
            except (KeyError, ValueError):
                # If sweep not in config, compute from measured data
                # Get this sweep's stimulus window
                try:
                    sweep_windows = sweep_config["sweeps"][str(sweep_id)]["windows"]
                    sweep_t_stim_min = sweep_windows["stimulus_start_s"]
                    sweep_t_stim_max = sweep_windows["stimulus_end_s"]
                except:
                    sweep_t_stim_min = t_stim_start
                    sweep_t_stim_max = t_stim_end
                
                df_pA_protocol = df_pA[df_pA["sweep"] == sweep_id]
                if len(df_pA_protocol) > 0:
                    df_pA_filtered = df_pA_protocol[(df_pA_protocol["t_s"] >= sweep_t_stim_min) & (df_pA_protocol["t_s"] <= sweep_t_stim_max)]
                    if len(df_pA_filtered) > 0:
                        mean_current = df_pA_filtered["value"].mean()
                        avg_pa_per_sweep.append({"sweep": sweep_id, "avg_injected_current_pA": mean_current})
                    else:
                        avg_pa_per_sweep.append({"sweep": sweep_id, "avg_injected_current_pA": np.nan})
                else:
                    avg_pa_per_sweep.append({"sweep": sweep_id, "avg_injected_current_pA": np.nan})
        
        avg_pa_per_sweep = pd.DataFrame(avg_pa_per_sweep)
        dbg(f"Using stimulus_level_pA from sweep_config for {len(avg_pa_per_sweep)} sweeps")
    else:
        # Fallback: compute from measured data if no sweep_config
        dbg(f"No sweep_config available. Computing current from measured pA data.")
        df_pA_protocol = df_pA[df_pA["sweep"].isin(valid_sweeps)].copy()
        
        if len(df_pA_protocol) == 0:
            dbg("WARNING: No current data found. Filling with NaN.")
            avg_pa_per_sweep = pd.DataFrame({
                "sweep": valid_sweeps,
                "avg_injected_current_pA": [np.nan] * len(valid_sweeps)
            })
        else:
            # Calculate baseline offset if pA data wasn't already replaced
            if not pA_was_replaced:
                baseline_window = df_pA_protocol[df_pA_protocol["t_s"] < t_stim_start]
                baseline_offset = baseline_window["value"].mean() if len(baseline_window) > 0 else 0.0
                df_pA_protocol["value"] = df_pA_protocol["value"] - baseline_offset
                dbg(f"Baseline offset correction: {baseline_offset:.2f} pA")
            
            # Compute mean current in stimulus window (use reference times)
            df_pA_filtered = df_pA_protocol[(df_pA_protocol["t_s"] >= t_stim_start) & (df_pA_protocol["t_s"] <= t_stim_end)]
            avg_pa_per_sweep = df_pA_filtered.groupby("sweep")["value"].mean().reset_index()
            avg_pa_per_sweep.rename(columns={"value": "avg_injected_current_pA"}, inplace=True)
    
    # Step 4: Round per-sweep means to nearest 5 pA (or 0) 
    if not avg_pa_per_sweep.empty:
        avg_pa_per_sweep["avg_injected_current_pA_rounded"] = (np.round(avg_pa_per_sweep["avg_injected_current_pA"] / 5) * 5).astype(float)
        avg_pa_per_sweep["avg_injected_current_pA"] = avg_pa_per_sweep["avg_injected_current_pA_rounded"]
        avg_pa_per_sweep = avg_pa_per_sweep.drop(columns=["avg_injected_current_pA_rounded"])
        dbg(f"Rounded per-sweep currents to nearest 5 pA (preview first 10):")
        print(avg_pa_per_sweep.head(10).to_string())

    # #-------------Create Dataframes-----------------
    results_df = pd.DataFrame(sweep_results)
    
    # Defensive check: ensure results_df has "sweep" column
    if results_df.empty:
        dbg("WARNING: No spike detection results. Creating empty results DataFrame.")
        # Create a DataFrame with all expected columns filled with NaN
        all_sweeps = avg_pa_per_sweep["sweep"].tolist()
        results_df = pd.DataFrame({"sweep": all_sweeps})
        # Add all the other columns as NaN
        for col in ["avg_peak_voltage_mV", "avg_threshold_voltage_mV", "avg_threshold_to_peak_mV",
                    "avg_fast_trough_at_hyperpolarization_mV", "avg_upstroke_to_peak_ms",
                    "avg_upstroke_to_peak_mV", "avg_peak_to_downstroke_ms", "avg_height_mV",
                    "avg_half_height_mV", "avg_ap_width_ms", "avg_threshold_fast_trough_width_ms",
                    "avg_upstroke_downstroke_ratio", "spike_frequency_Hz", "mean_isi_ms", "cv_isi",
                    "middle_first_AP_width_ratio", "middle_first_AP_peak_threshold_ratio",
                    "middle_first_AP_fast_trough_ratio", "last_first_AP_width_ratio",
                    "last_first_AP_peak_threshold_ratio", "last_first_AP_fast_trough_ratio"]:
            results_df[col] = np.nan
    elif "sweep" not in results_df.columns:
        dbg("ERROR: 'sweep' column missing from results_df. This should not happen.")
    
    results_df = results_df.merge(avg_pa_per_sweep, on="sweep", how="left")

    # Create AP_results_df for peak-level data
    rows = []
    for ap_index in range(max_peaks_overall):
        row = {"AP_Index": ap_index}
        for sweep, metrics in peak_level_data.items():
            for key, arr in metrics.items():
                col_name = f"Sweep_{sweep}_{key}"
                
                if ap_index < len(arr):
                    row[col_name] = arr[ap_index]
                else:
                    row[col_name] = np.nan
        
        rows.append(row)

    df_peak_level = pd.DataFrame(rows)

    print("PEAK LEVEL DATAFRAME",df_peak_level)

    # ----------------- Plot AP per sweep with labeled peaks -----------------
    plot_dir = Path(bundle_path) / "AP_Per_Sweep"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for sweep_number in sorted(filtered_peaks.keys()):
        # Get the FULL sweep data for this sweep (unfiltered to get true sweep offset)
        group_full = df[(df["sweep"] == sweep_number)].sort_values("t_s")
        sweep_offset = group_full["t_s"].min()  # True start of the sweep
        time_full = group_full["t_s"].to_numpy()  # Full sweep times for peak lookup
        volt_full = group_full["value"].to_numpy()  # Full sweep voltages for peak lookup
        
        # For mixed protocol, get THIS sweep's specific time windows and convert to relative times
        # For single protocol, use absolute times
        if is_mixed and sweep_config is not None:
            try:
                sweep_cfg = sweep_config["sweeps"].get(str(sweep_number), {})
                if sweep_cfg and "windows" in sweep_cfg:
                    # For mixed protocol: sweep_config contains ABSOLUTE times
                    sweep_t_start_abs = sweep_cfg["windows"]["stimulus_start_s"]
                    sweep_t_end_abs = sweep_cfg["windows"]["stimulus_end_s"]
                    
                    # Get sweep's starting time for converting to relative plotting
                    sweep_offset = group_full["t_s"].min()
                    
                    # Define plot window in ABSOLUTE times
                    plot_t_min = max(sweep_offset, sweep_t_start_abs - 0.02)
                    plot_t_max = sweep_t_end_abs + 0.02
                    
                    # Will convert to relative times for plotting
                    use_relative_times = True
                    sweep_relative_t_start = sweep_t_start_abs - sweep_offset
                    sweep_relative_t_end = sweep_t_end_abs - sweep_offset
                else:
                    use_relative_times = False
                    plot_t_min = max(t_stim_start - 0.1, t_stim_start - 0.02)
                    plot_t_max = t_stim_end + 0.02
            except (KeyError, TypeError):
                use_relative_times = False
                plot_t_min = max(t_stim_start - 0.1, t_stim_start - 0.02)
                plot_t_max = t_stim_end + 0.02
        else:
            # Single protocol: use absolute times
            use_relative_times = False
            plot_t_min = max(t_stim_start - 0.1, t_stim_start - 0.02)
            plot_t_max = t_stim_end + 0.02
        
        # Filter to the plot window
        if use_relative_times:
            # For mixed: filter using absolute times, then convert to relative for plotting
            group_plot = group_full[(group_full["t_s"] >= plot_t_min) & 
                                   (group_full["t_s"] <= plot_t_max)]
            # Convert times to RELATIVE (0-based from sweep start) for plotting
            time_plot = (group_plot["t_s"].to_numpy() - sweep_offset) * 1000  # Convert to ms relative
            time_offset_for_peaks = sweep_offset
        else:
            # For single: use absolute times
            group_plot = group_full[(group_full["t_s"] >= plot_t_min) & (group_full["t_s"] <= plot_t_max)]
            time_plot = group_plot["t_s"].to_numpy()  # Absolute times
            time_offset_for_peaks = 0
        
        volt_plot = group_plot["value"].to_numpy()
        peaks_plot = filtered_peaks.get(sweep_number, [])

        plt.figure(figsize=(10, 4))
        plt.plot(time_plot, volt_plot, label="Voltage (mV)")
        
        # Add stimulus window markers if analysis_windows available
        if analysis_windows is not None:
            if use_relative_times:
                # For mixed protocol: stimulus window in RELATIVE times (ms) for plotting
                t_stim_start = sweep_relative_t_start * 1000
                t_stim_end = sweep_relative_t_end * 1000
            else:
                # For single protocol: stimulus window in ABSOLUTE times
                t_stim_start = analysis_windows.get('t_stim_start')
                t_stim_end = analysis_windows.get('t_stim_end')
            
            if t_stim_start is not None and t_stim_end is not None:
                # Add vertical lines at stimulus boundaries
                plt.axvline(x=t_stim_start, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Stimulus start')
                plt.axvline(x=t_stim_end, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Stimulus end')
                # Optional: shade the stimulus region
                plt.axvspan(t_stim_start, t_stim_end, alpha=0.1, color='yellow', label='Stimulus window')
        
        if peaks_plot:
            # filtered_peaks stores (indices, times) tuple
            if isinstance(peaks_plot, tuple):
                peak_indices_in_filtered = np.array(peaks_plot[0])  # Indices into filtered window
                peak_times_abs = np.array(peaks_plot[1])  # Absolute times (already converted during detection)
            else:
                # Fallback for old format
                peak_indices_in_filtered = np.array(peaks_plot)
                peak_times_abs = time_full[peak_indices_in_filtered]
            
            if len(peak_times_abs) > 0 and len(peak_indices_in_filtered) > 0:
                # Get stimulus window for filtering (in absolute times)
                if use_relative_times:
                    # For mixed protocol: convert relative times back to absolute for comparison
                    stim_t_min = sweep_offset + sweep_relative_t_start
                    stim_t_max = sweep_offset + sweep_relative_t_end
                else:
                    stim_t_min = t_stim_start
                    stim_t_max = t_stim_end
                
                # Filter to peaks within stimulus window
                stim_mask = (peak_times_abs >= stim_t_min) & (peak_times_abs <= stim_t_max)
                peak_times_stim = peak_times_abs[stim_mask]
                peak_indices_stim = peak_indices_in_filtered[stim_mask]
                
                if len(peak_times_stim) > 0 and len(peak_indices_stim) > 0:
                    # Get peak voltages from peak_level_data (already have actual voltage values)
                    if sweep_number in peak_level_data:
                        peak_volt_array = np.array(peak_level_data[sweep_number]["AP_Height_mV"])
                        # Get voltages for the filtered peaks
                        peak_volts_stim = peak_volt_array[stim_mask]
                        
                        # Convert peak times to plot coordinate system
                        if use_relative_times:
                            peak_times_plot = (peak_times_stim - sweep_offset) * 1000  # Relative in ms
                        else:
                            peak_times_plot = peak_times_stim  # Absolute times
                        
                        # Plot peaks directly with their times and voltages
                        plt.plot(peak_times_plot, peak_volts_stim, "r*", markersize=12, label="Valid Peaks")

        plt.title(f"Sweep {sweep_number}")
        if use_relative_times:
            plt.xlabel("Time (ms) - Relative to Sweep Start")
        else:
            plt.xlabel("Time (s)")
        plt.ylabel("Voltage (mV)")
        plt.legend(loc="upper right", frameon=True)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_dir / f"AP_sweep_{sweep_number}.jpeg")
        plt.close()
        #plt.show()

    # ----------------- Merge with analysis + manifest -----------------
    # Drop old spike detection columns from df_analysis to avoid duplicate columns
    # Keep only the sweep and resting VM columns
    spike_cols_to_drop = [col for col in df_analysis.columns 
                         if col not in ["sweep", "resting_vm_mean_mV"]]
    df_analysis_clean = df_analysis.drop(columns=spike_cols_to_drop)
    
    # Now merge with fresh spike detection results
    updated_analysis = pd.merge(df_analysis_clean, results_df, on="sweep", how="left")
    
    # Sort by avg_injected_current_pA (ascending) before saving
    if "avg_injected_current_pA" in updated_analysis.columns:
        updated_analysis = updated_analysis.sort_values(by="avg_injected_current_pA", ascending=True).reset_index(drop=True)
    
    updated_analysis.to_parquet(Path(bundle_path) / "analysis.parquet", index=False)
    updated_analysis.to_csv(Path(bundle_path) / "analysis.csv", index=False)

    # =========================================================================
    # Generate min_frequency.csv - minimum current required to produce APs
    # =========================================================================
    rows_with_spikes_for_min_freq = updated_analysis[updated_analysis["spike_frequency_Hz"] > 0]
    
    if not rows_with_spikes_for_min_freq.empty:
        # Find the minimum current that produces spikes
        min_current_with_spikes = rows_with_spikes_for_min_freq["avg_injected_current_pA"].min()
        min_freq_row = rows_with_spikes_for_min_freq[
            rows_with_spikes_for_min_freq["avg_injected_current_pA"] == min_current_with_spikes
        ]
        min_freq_row.to_csv(Path(bundle_path) / "min_frequency.csv", index=False)
        print(f"\n✓ Saved min_frequency.csv (min current with APs: {min_current_with_spikes:.2f} pA)")
    else:
        # No spikes detected - save empty file with headers
        updated_analysis.iloc[:0].to_csv(Path(bundle_path) / "min_frequency.csv", index=False)
        print("\n⚠ No spikes detected - min_frequency.csv saved with headers only")

    # =========================================================================
    # Generate max_frequency.csv - current with maximum AP frequency
    # =========================================================================
    if not rows_with_spikes_for_min_freq.empty:
        # Find the maximum spike frequency
        max_freq = rows_with_spikes_for_min_freq["spike_frequency_Hz"].max()
        max_freq_row = rows_with_spikes_for_min_freq[
            rows_with_spikes_for_min_freq["spike_frequency_Hz"] == max_freq
        ]
        max_freq_row.to_csv(Path(bundle_path) / "max_frequency.csv", index=False)
        max_freq_current = max_freq_row["avg_injected_current_pA"].iloc[0]
        print(f"✓ Saved max_frequency.csv (max freq: {max_freq:.2f} Hz at {max_freq_current:.2f} pA)")
    else:
        # No spikes detected - save empty file with headers
        updated_analysis.iloc[:0].to_csv(Path(bundle_path) / "max_frequency.csv", index=False)
        print("⚠ No spikes detected - max_frequency.csv saved with headers only")

    # =========================================================================
    # Generate mean_frequency.csv - mean of all columns up to max frequency row
    # =========================================================================
    if not rows_with_spikes_for_min_freq.empty:
        # Find the row index of maximum frequency
        max_freq = rows_with_spikes_for_min_freq["spike_frequency_Hz"].max()
        max_freq_idx = rows_with_spikes_for_min_freq[
            rows_with_spikes_for_min_freq["spike_frequency_Hz"] == max_freq
        ].index[0]
        
        # Get all rows up to and including the max frequency row
        # (using the original updated_analysis to include all sweeps in order)
        rows_up_to_max = updated_analysis.loc[:max_freq_idx]
        
        # Calculate mean of numeric columns only
        numeric_cols = rows_up_to_max.select_dtypes(include=[np.number]).columns
        mean_values = rows_up_to_max[numeric_cols].mean()
        
        # Create a single-row DataFrame with mean values
        mean_df = pd.DataFrame([mean_values])
        
        # Add non-numeric columns as NaN or first value (for reference)
        non_numeric_cols = rows_up_to_max.select_dtypes(exclude=[np.number]).columns
        for col in non_numeric_cols:
            mean_df[col] = rows_up_to_max[col].iloc[0] if len(rows_up_to_max) > 0 else np.nan
        
        # Reorder columns to match original
        mean_df = mean_df[updated_analysis.columns]
        mean_df.to_csv(Path(bundle_path) / "mean_frequency.csv", index=False)
        print(f"✓ Saved mean_frequency.csv (mean of {len(rows_up_to_max)} rows up to max freq)")
    else:
        # No spikes detected - save empty file with headers
        updated_analysis.iloc[:0].to_csv(Path(bundle_path) / "mean_frequency.csv", index=False)
        print("⚠ No spikes detected - mean_frequency.csv saved with headers only")

    # Current threshold = first sweep with any spikes
    print("\n--- Calculating current threshold ---")
    dbg(f"Total sweeps: {len(updated_analysis)}")
    dbg(f"Sweeps with spikes: {(updated_analysis['spike_frequency_Hz'] > 0).sum()}")
    dbg(f"Current values (first 10): {updated_analysis['avg_injected_current_pA'].head(10).tolist()}")
    
    rows_with_spikes = updated_analysis[updated_analysis["spike_frequency_Hz"] > 0]

    if rows_with_spikes.empty:
        first_spike_current = np.nan 
        dbg("WARNING: No spikes detected in any sweep. Current threshold is NaN")
    else:
        # Get the first sweep with spikes and its current value
        first_spike_row = rows_with_spikes.iloc[0]
        first_spike_current = first_spike_row["avg_injected_current_pA"]
        
        # Check if it's NaN
        if pd.isna(first_spike_current):
            dbg(f"WARNING: First spike sweep {first_spike_row['sweep']} has NaN current value")
            first_spike_current = np.nan
        else:
            first_spike_current = float(first_spike_current)
            dbg(f"Current threshold (first spike at sweep {first_spike_row['sweep']}): {first_spike_current:.2f} pA")

    bundle_path = Path(bundle_path)
    manifest_path = bundle_path / "manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    manifest.setdefault("analysis", {})["current_threshold_pA"] = first_spike_current
    manifest.setdefault("analysis", {})["per_sweep_ap_metrics"] = "analysis.parquet"

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Update per peak data
    file_num = manifest["meta"].get("fileNum", None)
    cell_num = manifest["meta"].get("cellNum", None)
    df_peak_level["fileNum"] = file_num
    df_peak_level["cellNum"] = cell_num
    df_peak_level.to_csv(Path(bundle_path) / "AP_analysis.csv", index=False)

    # # =====================================================================
    # # Generate GIF from AP_Per_Sweep and Averaged_Peaks_Per_Sweep figures
    # # =====================================================================
    try:
        from PIL import Image
        import os
        
        print("GENERATING GIFs from analysis plots...")
        
        # Generate GIF for AP_Per_Sweep
        ap_sweep_dir = bundle_path / "AP_Per_Sweep"
        if ap_sweep_dir.exists():
            ap_files = sorted(ap_sweep_dir.glob("AP_sweep_*.jpeg"))
            if ap_files:
                print(f"  Creating GIF from {len(ap_files)} AP_Per_Sweep figures...")
                images = [Image.open(f) for f in ap_files]
                gif_path = bundle_path / "AP_Per_Sweep_animation.gif"
                images[0].save(
                    gif_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=500,  # 500ms per frame
                    loop=0
                )
                print(f"  [OK] Saved AP_Per_Sweep GIF to {gif_path.name}")
        
        # Generate GIF for Averaged_Peaks_Per_Sweep
        avg_peaks_dir = bundle_path / "Averaged_Peaks_Per_Sweep"
        if avg_peaks_dir.exists():
            avg_files = sorted(avg_peaks_dir.glob("Averaged_*.jpeg"))
            if avg_files:
                print(f"  Creating GIF from {len(avg_files)} Averaged_Peaks_Per_Sweep figures...")
                images = [Image.open(f) for f in avg_files]
                gif_path = bundle_path / "Averaged_Peaks_Per_Sweep_animation.gif"
                images[0].save(
                    gif_path,
                    save_all=True,
                    append_images=images[1:],
                    duration=500,  # 500ms per frame
                    loop=0
                )
                print(f"  [OK] Saved Averaged_Peaks_Per_Sweep GIF to {gif_path.name}")
                
    except ImportError:
        dbg("  WARNING: PIL (Pillow) not installed. Skipping GIF generation.")
        print("  Install with: pip install Pillow")

# FOR TESTING:
# bundle_dir = Path("/Users/snehajaikumar/30_09_25 Culture_48/2025_09_30_0027_517")
# mv_parquet = pd.read_parquet(bundle_dir / "mv_517.parquet")  # adjust filename if different
# pa_parquet = pd.read_parquet(bundle_dir / "pa_517.parquet")
# run_spike_detection(mv_parquet, pa_parquet, pa_parquet, 20000, bundle_dir)