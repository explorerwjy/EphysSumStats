"""
NWB to Analysis Pipeline
========================

Main driver for analyzing NWB files through the complete analysis pipeline.

Two-Step Workflow:
  STEP 1: Data Preparation (process_human_data.py)
    • Takes NWB files from a directory
    • Extracts voltage and current traces
    • Converts to parquet format
    • Creates manifest.json with metadata
    • Output: bundle directory with parquets
  
  STEP 2: Analysis & Classification (this script)
    • Takes bundle directory from STEP 1
    • Classifies sweeps (kept vs dropped)
    • Creates sweep_config.json
    • (Optional) Visualizes sweep classification
    • Runs full analysis (resting_vm, spike_detection, sav_gol, input_resistance)

Usage:
    # Run the full pipeline (both steps)
    python main.py
    
    # Or run analysis on an existing bundle
    python bundle_analyzer.py /path/to/bundle_dir [--visualize] [--skip-analysis]
"""

import sys
from pathlib import Path
import argparse
from pynwb import NWBHDF5IO

# Add Version 01-06 to path for run_analysis imports
# NOTE: REMOVED - This was importing old analysis_config.py with wrong signature
# sys.path.insert(0, '/Users/snehajaikumar/Version 01-06')

from sweep_classifier import process_bundle, visualize_sweeps_from_parquet, visualize_mixed_protocol_sweeps
from run_analysis import run_for_bundle


def main():
    """
    Main entry point for STEP 2: Analysis pipeline
    
    Takes a bundle directory (created by process_human_data.py)
    and runs sweep classification + full analysis.
    """
    parser = argparse.ArgumentParser(
        description="Run analysis on bundle directory (created by process_human_data.py)"
    )
    parser.add_argument(
        "bundle_dir",
        type=str,
        help="Path to bundle directory (contains mv_*.parquet, pa_*.parquet, manifest.json)"
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Only create sweep_config.json, skip spike detection"
    )
    
    args = parser.parse_args()
    
    bundle_path = Path(args.bundle_dir)
    
    # Validate bundle directory exists
    if not bundle_path.exists():
        print(f"ERROR: Bundle directory not found: {bundle_path}")
        sys.exit(1)
    
    # Validate required files exist
    required_files = ["manifest.json"]
    for fname in required_files:
        if not (bundle_path / fname).exists():
            print(f"ERROR: Missing required file: {bundle_path / fname}")
            print(f"This doesn't look like a valid bundle directory.")
            sys.exit(1)
    
    print("\n" + "="*70)
    print("STEP 2: ANALYSIS PIPELINE")
    print("="*70)
    
    # ========================================================================
    # STEP 1: Process bundle → Create sweep_config.json
    # ========================================================================
    print(f"\n[STEP 1] Analyzing bundle...")
    print(f"  Bundle: {bundle_path}")
    
    try:
        sweep_config = process_bundle(str(bundle_path))
        print(f"  ✓ sweep_config.json created")
    except Exception as e:
        print(f"  ✗ ERROR processing bundle: {e}")
        sys.exit(1)
    
    # ========================================================================
    # STEP 2: Visualize sweep classification (optional user prompt)
    # ========================================================================
    print(f"\n[STEP 2] Sweep Classification Visualization")
    print(f"  Would you like to visualize the kept vs dropped sweeps?")
    print(f"  This will create PNG plots showing which sweeps were kept and rejected.")
    
    # Auto-yes visualization
    viz_choice = "yes"
    if viz_choice in ["yes", "y"]:
        try:
            kept_sweeps = [int(k) for k, v in sweep_config.get('sweeps', {}).items() if v.get('valid')]
            dropped_sweeps = [int(k) for k, v in sweep_config.get('sweeps', {}).items() if not v.get('valid')]
            print(f"\n  Creating visualization ({len(kept_sweeps)} kept, {len(dropped_sweeps)} dropped)...")
            
            # Check if mixed protocol and use appropriate visualization function
            import json
            manifest_path = bundle_path / "manifest.json"
            with open(manifest_path) as f:
                manifest = json.load(f)
            is_mixed = "stimulus" in manifest["tables"] and "response" in manifest["tables"]
            
            if is_mixed:
                print(f"  Detected mixed protocol - using specialized visualization")
                visualize_mixed_protocol_sweeps(str(bundle_path), kept_sweeps, dropped_sweeps)
            else:
                visualize_sweeps_from_parquet(str(bundle_path), kept_sweeps, dropped_sweeps)
            print(f"  ✓ Visualization saved to bundle directory")
        except Exception as e:
            print(f"  ⚠ WARNING: Could not create visualization: {e}")
    
    # ========================================================================
    # CHECKPOINT: Before analysis - verify sweep_config
    # ========================================================================
    print(f"\n[CHECKPOINT] Sweep classification complete!")
    print(f"  sweep_config.json has been created with {sweep_config.get('valid_sweeps', 0)} valid sweeps")
    print(f"  Location: {bundle_path / 'sweep_config.json'}")
    
    # Auto-yes proceed with analysis
    checkpoint_ok = True
    print(f"\n  Auto-proceeding with analysis...")
    
    # ========================================================================
    # STEP 3: Run full analysis (if not skipped)
    # ========================================================================
    if args.skip_analysis:
        print(f"\n[STEP 3] Skipped (--skip-analysis flag set)")
        print(f"\n✓ sweep_config.json created. Ready for manual review/editing.")
        print(f"  Location: {bundle_path / 'sweep_config.json'}")
        sys.exit(0)
    
    print(f"\n[STEP 3] Running full analysis pipeline...")
    print(f"  Bundle: {bundle_path}")
    
    try:
        run_for_bundle(str(bundle_path))
        print(f"  ✓ Analysis complete")
    except Exception as e:
        import traceback
        print(f"  ✗ ERROR during analysis: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)
    
    # ========================================================================
    # DONE
    # ========================================================================
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {bundle_path}")
    print(f"  - sweep_config.json   (sweep metadata + windows)")
    print(f"  - analysis.parquet    (spike metrics)")
    print()


if __name__ == "__main__":
    main()
