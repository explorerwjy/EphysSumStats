"""
Central configuration for analysis parameters.
These are shared across spike_detection_new.py, sav_gol_filter.py, input_resistance.py, etc.
"""

# ============================================================================
# Spike Detection Window Parameters
# ============================================================================
# These define the time windows used for spike analysis

# Pre-spike analysis window (how far before peak to look)
PRE_THRESHOLD_WINDOW_MS = 4.5

# Post-spike analysis window (how far after peak to look)
POST_THRESHOLD_WINDOW_MS = 20.0

# Convert to seconds for convenience
PRE_THRESHOLD_WINDOW_S = PRE_THRESHOLD_WINDOW_MS / 1000.0
POST_THRESHOLD_WINDOW_S = POST_THRESHOLD_WINDOW_MS / 1000.0

# ============================================================================
# Spike Detection Parameters
# ============================================================================

# Threshold for detecting upstroke: percentage of max dV/dt
THRESHOLD_PERCENT = 0.05  # 5% of max upstroke

# Threshold for detecting fast trough: percentage of min dV/dt
FAST_TROUGH_PERCENT = 0.01  # 1% of max downstroke

# Peak detection parameters
PEAK_HEIGHT_THRESHOLD = -10  # mV
PEAK_PROMINENCE = 20  # mV

# Minimum distance between peaks to avoid detecting noise as multiple peaks
MIN_PEAK_DISTANCE_MS = 2.0
MIN_PEAK_DISTANCE_S = MIN_PEAK_DISTANCE_MS / 1000.0

# Minimum peak-threshold amplitude to be considered a valid spike
MIN_PEAK_THRESHOLD_AMPLITUDE_MV = 15.0

# ============================================================================
# Savitzky-Golay Filter Parameters
# ============================================================================

# Reference smoothing window for 2-second recording
REFERENCE_WINDOW_S = 2.0
REFERENCE_SMOOTH_MS = 200.05

# Savitzky-Goyal polynomial order
SAV_GOL_POLY_ORDER = 3

# Baseline window size for drift analysis and RMP derivative calculation
BASELINE_WINDOW_MS = 25
BASELINE_WINDOW_S = BASELINE_WINDOW_MS / 1000.0

# ============================================================================
# Sweep Classification Parameters
# ============================================================================

# Current thresholds for stimulus detection
BASELINE_THRESHOLD_PA = 0.01      # Current below this is considered "no injection"
STIMULUS_THRESHOLD_PA = 5.0       # Current above this is considered "injection"

# Stimulus window detection
MIN_STIMULUS_DURATION_S = 0.300   # Minimum 300ms stimulus duration
MIN_FLAT_RATIO = 0.70             # 70% of segment must be at stable current (square wave)
RESPONSE_PADDING_S = 0.10         # Extra 100ms after stimulus for response window
BASELINE_FALLBACK_S = 0.01        # Last resort: use first 10ms if baseline detection fails

# ============================================================================
# Artifact Detection Parameters
# ============================================================================

# Second derivative threshold for detecting sharp corners/artifacts
# Normal AP: d²V/dt² up to ~2 billion mV/s²
# True artifacts: 20+ billion mV/s²
SECOND_DERIV_THRESHOLD = 10e9     # 10 billion mV/s²

# Voltage jump threshold for detecting discontinuities
VOLTAGE_JUMP_THRESHOLD = 10.0     # mV - sudden voltage jump in single sample

# ============================================================================
# Helper Functions
# ============================================================================

def get_analysis_window_bounds(sweep_config=None):
    """
    Get the time bounds for analysis based on response/stimulus windows.
    
    Args:
        sweep_config: Dict from sweep_config.json containing response windows (required)
        
    Returns:
        tuple: (t_min, t_max) in seconds for analysis window
               Used by input_resistance.py and other analysis functions
               
    Raises:
        ValueError: If sweep_config is not provided
    """
    # sweep_config is required to determine proper analysis windows
    if sweep_config is None:
        raise ValueError("sweep_config is required to determine analysis window bounds")
    
    try:
        # Find first valid sweep
        valid_sweep = None
        for sweep_id, sweep_data in sweep_config.get("sweeps", {}).items():
            if sweep_data.get("valid", False):
                valid_sweep = sweep_id
                break
        
        if valid_sweep is not None:
            windows = sweep_config["sweeps"][valid_sweep]["windows"]
            # The analysis window is the stimulus period (when stimulus is applied)
            t_min = windows["stimulus_start_s"]
            t_max = windows["stimulus_end_s"]
            print(f"Using analysis window from sweep_config: [{t_min:.6f}, {t_max:.6f}] s")
        else:
            # No valid sweep found in config
            raise ValueError("No valid sweep found in sweep_config")
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"Failed to extract response window from sweep_config: {e}")
    
    # Expand by threshold windows to capture complete spike morphology
    t_min_expanded = t_min - PRE_THRESHOLD_WINDOW_S
    t_max_expanded = t_max + POST_THRESHOLD_WINDOW_S
    
    return t_min_expanded, t_max_expanded


def get_smoothing_proportion():
    """
    Calculate the smoothing proportion for adaptive filtering.
    
    The proportion is: (reference_smooth_ms) / (reference_window_s * 1000)
    This ensures consistent smoothing across different recording durations.
    
    Returns:
        float: Smoothing proportion (typically ~0.1 for 200.05ms in 2s window)
    """
    return REFERENCE_SMOOTH_MS / (REFERENCE_WINDOW_S * 1000)
