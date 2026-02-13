"""
Helper script to manually replace pA parquet values in a faulty bundle.

Usage:
    python replace_pA_manual.py
    
    Then follow the prompts to select:
    1. The FAULTY bundle (with empty/corrupted pA data)
    2. The REFERENCE bundle (with good pA data)
"""

import pandas as pd
from pathlib import Path
import json

def replace_pA_manual():
    """
    Interactively replace pA parquet values from a reference bundle into a faulty bundle.
    """
    
    print("\n" + "="*70)
    print("pA PARQUET VALUE REPLACEMENT TOOL")
    print("="*70)
    
    # Step 1: Get faulty bundle path
    print("\n[STEP 1] Enter path to FAULTY bundle (the one with empty/corrupted pA)")
    print("Example: C:\\Users\\manol\\Version 01-06\\2025_12_02_0001_660")
    faulty_bundle = input("Faulty bundle path: ").strip()
    
    if not Path(faulty_bundle).exists():
        print(f"ERROR: Path does not exist: {faulty_bundle}")
        return False
    
    # Step 2: Get reference bundle path
    print("\n[STEP 2] Enter path to REFERENCE bundle (the one with GOOD pA data)")
    print("Example: C:\\Users\\manol\\Version 01-06\\2025_12_02_0002_668")
    ref_bundle = input("Reference bundle path: ").strip()
    
    if not Path(ref_bundle).exists():
        print(f"ERROR: Path does not exist: {ref_bundle}")
        return False
    
    # Step 3: Load manifests
    print("\n[STEP 3] Loading manifest files...")
    p_faulty = Path(faulty_bundle)
    p_ref = Path(ref_bundle)
    
    try:
        man_faulty = json.loads((p_faulty / "manifest.json").read_text())
        man_ref = json.loads((p_ref / "manifest.json").read_text())
    except Exception as e:
        print(f"ERROR: Could not load manifest files: {e}")
        return False
    
    # Get pA filenames
    pa_faulty_name = man_faulty["tables"]["pa"]
    pa_ref_name = man_ref["tables"]["pa"]
    
    print(f"  Faulty pA file: {pa_faulty_name}")
    print(f"  Reference pA file: {pa_ref_name}")
    
    # Step 4: Load pA parquets
    print("\n[STEP 4] Loading pA parquet files...")
    try:
        df_pa_faulty = pd.read_parquet(p_faulty / pa_faulty_name)
        df_pa_ref = pd.read_parquet(p_ref / pa_ref_name)
    except Exception as e:
        print(f"ERROR: Could not load pA files: {e}")
        return False
    
    print(f"  Faulty pA shape: {df_pa_faulty.shape}")
    print(f"  Reference pA shape: {df_pa_ref.shape}")
    print(f"  Faulty sweeps: {sorted(df_pa_faulty['sweep'].unique())}")
    print(f"  Reference sweeps: {sorted(df_pa_ref['sweep'].unique())}")
    
    # Step 5: Map sweeps
    print("\n[STEP 5] Mapping sweep numbers...")
    target_sweeps = sorted(df_pa_faulty["sweep"].unique())
    ref_sweeps = sorted(df_pa_ref["sweep"].unique())

    # If the faulty pA file has no sweeps (empty), we'll write the reference sweeps
    # into the target filename and use the reference sweep numbering.
    if len(target_sweeps) == 0:
        print("  Note: Faulty pA contains no sweeps. Will write reference sweeps into target file.")
        target_sweeps = list(ref_sweeps)

    n_map = min(len(ref_sweeps), len(target_sweeps))
    if n_map == 0:
        print("ERROR: No sweeps to map!")
        return False
    
    if len(ref_sweeps) != len(target_sweeps):
        print(f"  ⚠ WARNING: Sweep count mismatch")
        print(f"    Reference has {len(ref_sweeps)} sweeps")
        print(f"    Faulty has {len(target_sweeps)} sweeps")
        print(f"    Will map first {n_map} sweeps")
    
    sweep_mapping = {ref_sweeps[i]: target_sweeps[i] for i in range(n_map)}
    print(f"  Mapping: {sweep_mapping}")
    
    # Step 6: Remap and replace
    print("\n[STEP 6] Remapping reference data...")
    df_pa_ref_remapped = df_pa_ref.copy()
    df_pa_ref_remapped["sweep"] = df_pa_ref_remapped["sweep"].map(sweep_mapping)
    
    # Drop unmapped rows
    before = len(df_pa_ref_remapped)
    df_pa_ref_remapped = df_pa_ref_remapped.dropna(subset=["sweep"])
    after = len(df_pa_ref_remapped)
    
    if after < before:
        print(f"  Dropped {before - after} rows that could not be mapped")
    
    df_pa_ref_remapped["sweep"] = df_pa_ref_remapped["sweep"].astype(int)
    
    # Step 7: Preview
    print(f"\n[STEP 7] Preview of remapped data:")
    print(f"  New shape: {df_pa_ref_remapped.shape}")
    print(f"  New sweeps: {sorted(df_pa_ref_remapped['sweep'].unique())}")
    print(f"\n  First few rows:")
    print(df_pa_ref_remapped.head())
    # Apply baseline offset correction + per-sweep averaging + rounding to 5 pA increments
    try:
        import numpy as _np
        # Step 1: Calculate baseline offset during quiet period (beginning of recording)
        # Using first 10% as baseline (older versions used 1.6-1.8s for mouse data)
        t_max = df_pa_ref_remapped['t_s'].max()
        baseline_window = df_pa_ref_remapped[df_pa_ref_remapped['t_s'] < (t_max * 0.1)]
        baseline_offset = baseline_window['value'].mean() if len(baseline_window) > 0 else 0.0
        print(f"\nBaseline offset (first 10% of recording): {baseline_offset:.2f} pA")

        # Step 2: Subtract baseline offset from all values
        df_pa_ref_remapped['value'] = df_pa_ref_remapped['value'] - baseline_offset

        # Step 3: Compute mean current in the stimulus window per sweep (after offset correction)
        # Using middle 50% of recording (older versions used 2.1-2.55s for mouse data)
        t_min = df_pa_ref_remapped['t_s'].min()
        t_window_min = t_min + (t_max - t_min) * 0.2
        t_window_max = t_min + (t_max - t_min) * 0.7
        df_window = df_pa_ref_remapped[(df_pa_ref_remapped['t_s'] >= t_window_min) & (df_pa_ref_remapped['t_s'] <= t_window_max)]
        avg_pa = df_window.groupby('sweep')['value'].mean().reset_index(name='avg_injected_current_pA')
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
    except Exception as e:
        print(f"Warning: could not apply per-sweep rounding to remapped data: {e}")

    # Step 8: Confirm and save
    print(f"\n[STEP 8] Ready to replace values in {pa_faulty_name}")
    confirm = input("  Proceed with replacement? (yes/no): ").strip().lower()
    
    if confirm != "yes":
        print("  Cancelled.")
        return False
    
    try:
        pa_faulty_path = p_faulty / pa_faulty_name
        df_pa_ref_remapped.to_parquet(pa_faulty_path, index=False)
        print(f"\n✓ SUCCESS! Replaced values in {pa_faulty_name}")
        print(f"  Location: {pa_faulty_path}")
        print(f"  Rows: {len(df_pa_ref_remapped)}")
        return True
    except Exception as e:
        print(f"\n✗ ERROR: Could not save parquet: {e}")
        return False


if __name__ == "__main__":
    success = replace_pA_manual()
    
    if success:
        print("\n" + "="*70)
        print("Next steps:")
        print("  1. Run: python -c \"from run_analysis import run_for_bundle; run_for_bundle(r'<faulty_bundle_path>')\"")
        print("  2. When prompted for reference, press Enter (already replaced)")
        print("  3. Analysis will run with correct current data")
        print("="*70 + "\n")
    else:
        print("\nReplacement failed. Please check the paths and try again.\n")
