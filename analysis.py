# analysis.py
import pandas as pd
import numpy as np
import json
from pathlib import Path
import re
from typing import Optional


def sweep_sort_key(s: str):
    """
    Natural-sort key for sweep names like Vm_0, Vm_10, I_2, etc.
    Sorts by trailing integer if present; otherwise falls back to string.
    """
    if s is None:
        return -1
    m = re.search(r"(\d+)$", str(s))
    return int(m.group(1)) if m else str(s)


#resting membrane potential - looks at average mV for first 2 seconds per sweep (total 40 sweeps)
def resting_vm_per_sweep(df_mv: pd.DataFrame, sweep_config: Optional[dict] = None, bundle_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Input: long mV table with columns ['sweep','t_s','value', 'channel_name', ...]
           sweep_config: optional dict from sweep_config.json with baseline window per sweep
           bundle_dir: optional path to bundle (used to detect if mixed protocol)
    Returns: one row per sweep with mean resting Vm (mV).
    
    IMPORTANT: For MIXED PROTOCOL files only:
    sweep_config.json uses RELATIVE times per sweep (0-27s)
    but the parquet files use ABSOLUTE times (all sweeps concatenated, e.g., 278-1856s)
    This function converts relative times to absolute times for mixed protocol files.
    """
    baseline = df_mv
    
    # sweep_config is required to determine baseline windows
    if not sweep_config:
        raise ValueError("sweep_config is required to determine baseline windows for each sweep")
    #change
    # Detect if mixed protocol
    is_mixed = False
    if bundle_dir:
        try:
            p = Path(bundle_dir)
            man = json.loads((p / "manifest.json").read_text())
            is_mixed = "stimulus" in man["tables"] and "response" in man["tables"]
        except:
            is_mixed = False
    
    # For mixed protocol, pre-compute absolute time offsets for each sweep
    sweep_offsets = {}
    if is_mixed:
        for sweep_id in df_mv["sweep"].unique():
            sweep_data = df_mv[df_mv["sweep"] == sweep_id]
            if len(sweep_data) > 0:
                sweep_offsets[int(sweep_id)] = sweep_data["t_s"].min()
    
    # VECTORIZED APPROACH: Filter by baseline window for each sweep
    # Instead of using slow apply(), build a list of filtered DataFrames per sweep
    baseline_dfs = []
    
    for sweep_id in df_mv["sweep"].unique():
        # Get config key (try both int and str)
        try:
            key = int(sweep_id)
            if key not in sweep_config["sweeps"]:
                key = str(key)
        except (ValueError, TypeError):
            key = str(sweep_id)
        
        # Skip if sweep not in config
        if key not in sweep_config["sweeps"]:
            continue
        
        # Get baseline window for this sweep
        try:
            windows = sweep_config["sweeps"][key]["windows"]
            t_baseline_start = windows["baseline_start_s"]
            t_baseline_end = windows["baseline_end_s"]
        except KeyError:
            continue
        
        # For mixed protocol, times are already absolute (use directly)
        # For single protocol, times are relative (also use directly)
        # No conversion needed!
        
        # Filter this sweep's data to baseline window
        sweep_data = df_mv[df_mv["sweep"] == sweep_id]
        sweep_baseline = sweep_data[
            (sweep_data["t_s"] >= t_baseline_start) & 
            (sweep_data["t_s"] <= t_baseline_end)
        ]
        
        if len(sweep_baseline) > 0:
            baseline_dfs.append(sweep_baseline)
    
    # Combine all baseline data
    if not baseline_dfs:
        # No baseline data found, return empty DataFrame
        return pd.DataFrame(columns=["sweep", "resting_vm_mean_mV"])
    
    baseline_filtered = pd.concat(baseline_dfs, ignore_index=True)

    agg = (baseline_filtered
           .groupby("sweep", as_index=False)["value"]
           .mean()
           .rename(columns={"value": "resting_vm_mean_mV"}))

    agg = agg.sort_values(
        by="sweep",
        key=lambda s: s.map(sweep_sort_key)
    ).reset_index(drop=True)

    return agg

#Consolidate metadata with results
def attach_manifest_to_analysis(bundle_dir: str, df_analysis):
    """
    For a single bundle directory, load manifest.json and analysis.parquet,
    and add:
      - file_id
      - all manifest["meta"] fields
      - selected manifest["analysis"] fields
    as constant columns to every row (each sweep).
    Then overwrite analysis.parquet and analysis.csv.
    """
    p = Path(bundle_dir)

    # --- load manifest ---
    man_path = p / "manifest.json"
    with man_path.open("r") as f:
        manifest = json.load(f)

    # --- base fields ---
    file_id = manifest.get("file_id")

    meta = manifest.get("meta", {})
    manifest_analysis  = manifest.get("analysis", {})

    # pick the bundle-level analysis fields you care about
    selected_analysis = {
        "grand_average_resting_vm": manifest_analysis.get("resting_vm_mean"),
        "filtered_grand_average_resting_vm": manifest_analysis.get("filtered_grand_average_resting_vm_mean"),
        "input_resistance": manifest_analysis.get("input_resistance"),
        "current_threshold_pA": manifest_analysis.get("current_threshold_pA"),
    }

    # prefix meta keys so you don't collide with column names
    # but skip protocol-specific fields that shouldn't be in analysis CSV
    skip_keys = {
        'protocols',
        'protocol_info', 
        'samplingRates_detected',
        'sampleRate_Hz',
    }
    meta_prefixed = {f"meta_{k}": v for k, v in meta.items() if k not in skip_keys}

    # build a dict of constant columns to broadcast to all rows
    extra_cols = {
        **selected_analysis,
        "file_id": file_id,
        **meta_prefixed
    }

    # broadcast into df (each row gets the same values)
    for col_name, value in extra_cols.items():
        # Skip columns that are lists/arrays - they can't be broadcast
        if isinstance(value, (list, tuple)):
            continue
        df_analysis[col_name] = value

    # --- save back ---
    out_parq = p / "analysis.parquet"
    out_csv  = p / "analysis.csv"

    # Sort by avg_injected_current_pA (ascending) before saving
    if "avg_injected_current_pA" in df_analysis.columns:
        df_analysis = df_analysis.sort_values(by="avg_injected_current_pA", ascending=True).reset_index(drop=True)

    df_analysis.to_parquet(out_parq, index=False)
    df_analysis.to_csv(out_csv, index=False)

    print(f"[OK] Combined analysis for {bundle_dir} with manifest metadata.")

