# Why Sampling Rate is Required for Savitzky-Golay Filter

## Overview
The Savitzky-Golay (Sav-Gol) filter in our pipeline requires sampling rate information to properly configure the filter window size. This document explains why this is necessary and how it's used.

## The Core Problem

**The Sav-Gol filter operates on discrete samples, but we want to specify smoothing in terms of time.**

- **Our goal**: Smooth voltage data over a specific time duration (e.g., 600 milliseconds)
- **Filter requirement**: Needs to know the number of samples in the smoothing window
- **The gap**: Time duration → Number of samples conversion requires sampling rate

## Mathematical Relationship

```
window_length (samples) = desired_smoothing_time (seconds) × sampling_rate (Hz)
```

### Example Calculations

For a 600 ms smoothing window:

| Sampling Rate | Calculation | Window Length |
|--------------|-------------|---------------|
| 50 kHz | 0.6 s × 50,000 Hz | 30,000 samples |
| 200 kHz | 0.6 s × 200,000 Hz | 120,000 samples |

**Key insight**: Same time duration results in different sample counts based on sampling rate.

## Why This Matters for Mixed Protocol Data

Our Allen Brain data contains **mixed protocols** with different sampling rates:

- **CurrentClamp protocol**: 50 kHz (50,000 samples/second)
- **VoltageClamp protocol**: 200 kHz (200,000 samples/second)

### Without Per-Sweep Sampling Rates:
If we used a fixed sample count (e.g., 30,000 samples):
- At 50 kHz: 30,000 samples = 600 ms ✓
- At 200 kHz: 30,000 samples = 150 ms ✗ (inconsistent smoothing!)

### With Per-Sweep Sampling Rates:
Using the actual rate for each sweep:
- At 50 kHz: 30,000 samples = 600 ms ✓
- At 200 kHz: 120,000 samples = 600 ms ✓ (consistent smoothing!)

## Implementation in Code

### 1. Window Length Calculation (Line 163)
```python
window_length = int((desired_smooth_ms / 1000) * sweep_fs)
```

**Purpose**: Convert desired smoothing time to appropriate sample count for each sweep

**Example**:
- `desired_smooth_ms = 609.94 ms`
- `sweep_fs = 50000 Hz`
- `window_length = 30,497 samples`

### 2. Histogram Binning (Line 285)
```python
samples_per_window = int(fs_default * window_s)  # 50ms bins
```

**Purpose**: Create time-consistent bins for analyzing filtered voltage distribution

**Example**:
- 50 ms bins at 50 kHz = 2,500 samples/bin
- Ensures consistent temporal resolution across analyses

### 3. Filter Application
The Savitzky-Golay filter function (`savgol_filter`) requires:
- `window_length`: Number of samples in the smoothing window (must be odd)
- `poly_order`: Polynomial order for fitting

Without correct `window_length`, the filter would apply inconsistent smoothing across different sampling rates.

## Benefits of Our Approach

### ✅ Temporal Consistency
- Same smoothing duration across all sweeps regardless of sampling rate
- Consistent filtering behavior for comparable results

### ✅ Protocol-Aware Processing
- Each sweep uses its actual sampling rate from manifest
- No assumptions about uniform sampling across protocols

### ✅ Quality Control
- Proper smoothing preserves signal characteristics
- Avoids over-smoothing (high-rate data) or under-smoothing (low-rate data)

## Real Data Example

From our test run on sub-1000610030:

```
Adaptive smoothing window: 609.95 ms

SWEEP 4:
  Using sweep-specific rate: 50000.0 Hz
  Window length: 30497 samples (609.94 ms)
  Baseline period: [883.216000, 883.260980] s (2250 samples)
```

**Result**: 
- Filter window = 609.94 ms consistently
- Actual sample count adjusted based on sweep's 50 kHz rate
- Proper smoothing applied to baseline period

## Conclusion

**Sampling rate is essential for the Sav-Gol filter because:**

1. It converts time-based specifications → sample counts
2. It ensures consistent temporal smoothing across different protocols
3. It enables protocol-aware, per-sweep filter configuration
4. It maintains signal quality and comparability across sweeps

Without sampling rate information, we cannot properly configure the filter for mixed-protocol electrophysiology data.
