#!/usr/bin/env python3
"""
ELECTROPHYSIOLOGY ANALYSIS PIPELINE
====================================

Main entry point for analyzing electrophysiology data.
Supports both ABF (patch clamp) and NWB (Neurodata Without Borders) formats.

This script will guide you through the analysis process step-by-step.
"""

import sys
import os
import subprocess
from pathlib import Path
import h5py
import json
import gc
from typing import Tuple, Dict, Any

def detect_nwb_protocol_type(nwb_file: str) -> Tuple[str, Dict[str, Any]]:
    """
    Detect if NWB file has single or mixed protocols.
    
    Returns:
        Tuple of (protocol_type, protocol_info) where:
        - protocol_type: "single" or "mixed"
        - protocol_info: dict with protocol details
    """
    try:
        with h5py.File(nwb_file, 'r') as f:
            # Check if this is an intracellular ephys recording
            if 'general' not in f or 'intracellular_ephys' not in f['general']:
                return None, None
            
            ice = f['general']['intracellular_ephys']
            protocols_found = set()
            rates_found = set()
            
            # Check sweep table for protocol information
            if 'sweep_table' in ice:
                sweep_table = ice['sweep_table']
                
                # Look at the intracellular_electrode column
                if 'intracellular_electrode' in sweep_table:
                    electrodes = sweep_table['intracellular_electrode']
                    
                    for i, electrode_ref in enumerate(electrodes[:]):
                        # Get protocol from stimulus description or other attributes
                        if 'stimulus_description' in electrodes.attrs:
                            stim = electrodes.attrs['stimulus_description']
                            protocols_found.add(stim.decode() if isinstance(stim, bytes) else stim)
            
            # Check stimulus data for protocol types (VoltageClamp vs CurrentClamp)
            stimulus_data = f['stimulus/presentation'] if 'stimulus' in f and 'presentation' in f['stimulus'] else None
            
            if stimulus_data:
                for key in stimulus_data.keys():
                    series = stimulus_data[key]
                    series_type = series.attrs.get('neurodata_type', b'').decode() if isinstance(series.attrs.get('neurodata_type'), bytes) else series.attrs.get('neurodata_type', '')
                    if 'VoltageClampStimulusSeries' in series_type or 'VoltageClamp' in key:
                        protocols_found.add('VoltageClamp')
                    elif 'CurrentClampStimulusSeries' in series_type or 'CurrentClamp' in key:
                        protocols_found.add('CurrentClamp')
            
            # Check acquisition data for response types (which tells us about stimulus protocols)
            if 'acquisition' in f:
                acq = f['acquisition']
                for key in acq.keys():
                    series = acq[key]
                    series_type = series.attrs.get('neurodata_type', b'').decode() if isinstance(series.attrs.get('neurodata_type'), bytes) else series.attrs.get('neurodata_type', '')
                    if 'VoltageClampSeries' in series_type or 'voltage' in key.lower():
                        protocols_found.add('VoltageClamp')
                    elif 'CurrentClampSeries' in series_type or 'current' in key.lower():
                        protocols_found.add('CurrentClamp')
            
            # Determine if mixed or single protocol
            protocol_type = 'mixed' if len(protocols_found) > 1 else 'single'
            
            protocol_info = {
                'protocols': list(protocols_found),
                'count': len(protocols_found),
                'file': nwb_file
            }
            
            return protocol_type, protocol_info
    
    except Exception as e:
        print(f"⚠ Warning: Could not detect protocol type: {e}")
        # Default to single protocol if detection fails
        return 'single', {'error': str(e)}


def print_header():
    """Print welcome banner"""
    print("\n" + "="*70)
    print("  ELECTROPHYSIOLOGY ANALYSIS PIPELINE")
    print("="*70)
    print("\nSupported formats:")
    print("  • ABF files (.abf) - Patch clamp electrophysiology")
    print("  • NWB files (.nwb) - Neurodata Without Borders format")
    print()


def get_file_type():
    """Prompt user to select file type"""
    print("What type of data are you analyzing?")
    print("  1) ABF files (.abf)")
    print("  2) NWB files (.nwb)")
    print()
    
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice in ["1", "2"]:
            return choice
        print("  ✗ Invalid input. Please enter 1 or 2.")


def run_abf_pipeline():
    """Run ABF analysis pipeline"""
    print("\n" + "="*70)
    print("ABF PIPELINE")
    print("="*70)
    print("""
This pipeline will:
  1. Read all .abf files in a directory
  2. Parse ABF metadata (recording date, sweep info, etc.)
  3. Bundle sweeps into parquet format
  4. Create sweep_config.json for classification
  5. Run full analysis:
     - Resting membrane potential
     - Spike detection
     - Savitzky-Golay filtering
     - Input resistance calculation

Launching zuckerman-abf.py...
""")
    
    try:
        import zuckerman_abf
        # Note: This assumes the file is in Version 01-06
        # If it's in the current directory, adjust the import
        #COMMENT OUT AFTERWARD:
        sys.path.insert(0, str(Path(__file__).parent / 'Version 01-06'))
        from zuckerman_abf import main as abf_main
        abf_main()
    except ImportError as e:
        print(f"✗ ERROR: Could not import zuckerman-abf.py")
        print(f"  Make sure zuckerman-abf.py is in Version 01-06 directory")
        print(f"  Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ ERROR running ABF pipeline: {e}")
        sys.exit(1)


def get_parent_directory() -> Path:
    """Prompt user for parent directory containing subject folders with NWB files"""
    while True:
        parent_dir = input("\nEnter parent directory path (contains subject subfolders): ").strip()
        parent_path = Path(parent_dir)
        
        if not parent_path.exists():
            print(f"✗ Directory not found: {parent_path}")
            continue
        
        if not parent_path.is_dir():
            print(f"✗ Path is not a directory: {parent_path}")
            continue
        
        # Check if there are any NWB files in subdirectories
        nwb_files = list(parent_path.rglob("*.nwb"))
        if not nwb_files:
            print(f"✗ No .nwb files found in {parent_path} or its subdirectories")
            continue
        
        print(f"\n✓ Found {len(nwb_files)} NWB files")
        
        # Show sample of what will be processed
        print("\nSample of NWB files found:")
        for nwb in sorted(nwb_files)[:5]:
            print(f"  • {nwb.relative_to(parent_path)}")
        if len(nwb_files) > 5:
            print(f"  ... and {len(nwb_files) - 5} more")
        
        confirm = input("\nProcess all NWB files in this directory? (y/n): ").strip().lower()
        if confirm == 'y':
            return parent_path
        
        print("Please try another directory.")


def run_nwb_data_preparation():
    """Run STEP 1: Data preparation with automatic protocol detection"""
    print("\n" + "="*70)
    print("STEP 1: DATA PREPARATION (Automatic Protocol Detection)")
    print("="*70)
    print("""
This step extracts NWB data and creates bundle directories.

For each NWB file in the directory, this will:
  • Detect if single or mixed protocol
  • Extract voltage and current traces
  • Convert units (V → mV, A → pA)
  • Save as parquet files (fast I/O)
  • Create manifest.json with metadata

Single protocol: Uses standard process_human_data.py
Mixed protocol:  Uses process_human_data_mixed_protocol.py with per-sweep rates
""")
    
    # Get parent directory containing subject folders with NWB files
    parent_dir = get_parent_directory()
    
    # Find all NWB files
    nwb_files = sorted(parent_dir.rglob("*.nwb"))
    print(f"\n📊 Found {len(nwb_files)} NWB file(s) to process")
    
    # Analyze protocol type for each file
    protocol_analysis = {}
    print("\n📊 Analyzing NWB file structures...")
    
    for nwb_file in nwb_files:
        filename = nwb_file.stem
        subject_id = filename.split('_')[0] if '_' in filename else 'unknown'
        
        protocol_type, protocol_info = detect_nwb_protocol_type(str(nwb_file))
        
        if protocol_info is None:
            protocol_type = 'single'
            protocol_info = {}
        
        protocol_analysis[str(nwb_file)] = {
            'subject_id': subject_id,
            'type': protocol_type,
            'info': protocol_info
        }
        
        protocols_str = ', '.join(protocol_info.get('protocols', ['unknown']))
        print(f"  • {filename}: {protocol_type.upper()} ({protocols_str})")
    
    # Call appropriate processing script(s)
    script_dir = Path(__file__).parent
    
    try:
        print(f"\n🔄 Starting extraction process...")
        
        # Validate template file ONCE before processing any files
        template_path = script_dir / 'ePhys_log_sheet.xlsx'
        if not template_path.is_file():
            print(f"\n⚠ Template file not found: {template_path}")
            print("Please provide the path to the Excel metadata template:")
            template_path_input = input("Path to Excel metadata template: ").strip()
            template_path = Path(template_path_input).expanduser()
            
            # Validate the user-provided path
            if not template_path.is_file():
                print(f"\n✗ ERROR: Template file not found: {template_path}")
                print("Cannot proceed without metadata template.")
                sys.exit(1)
            
            print(f"✓ Using template: {template_path}\n")
        
        # Separate files by protocol type
        single_protocol_files = []
        mixed_protocol_files = []
        
        for nwb_file_path, analysis in protocol_analysis.items():
            if analysis['type'] == 'mixed':
                mixed_protocol_files.append(nwb_file_path)
            else:
                single_protocol_files.append(nwb_file_path)
        
        # Process mixed protocol files (one at a time)
        if mixed_protocol_files:
            print(f"\n📊 Processing {len(mixed_protocol_files)} mixed protocol file(s)...\n")
            
            # Track cell count per subject to pass to script
            subject_cell_counts = {}
            
            for nwb_file in mixed_protocol_files:
                # Extract subject ID and get count (starts at 0)
                subject_id = Path(nwb_file).stem.split('_')[0]  # e.g., "sub-1000610030"
                cell_count = subject_cell_counts.get(subject_id, 0)
                subject_cell_counts[subject_id] = cell_count + 1  # Increment for next file
                
                print(f"\n{'='*70}")
                print(f"Processing: {Path(nwb_file).name}")
                print(f"Protocol: MIXED")
                print(f"Cell: {subject_id}_{cell_count}")
                print('='*70)
                print(f"🔄 Mixed protocol detected - Using enhanced extraction...")
                print(f"   Launching process_human_data_mixed_protocol.py...\n")
                
                # Use generic mixed protocol script for all mixed protocol files
                mixed_script = script_dir / 'process_human_data_mixed_protocol.py'
                if mixed_script.exists():
                    # Pass: parent_dir, log_output_dir, template_path, nwb_file, cell_count
                    result = subprocess.run(
                        [sys.executable, str(mixed_script), 
                         str(parent_dir), str(parent_dir), str(template_path),
                         str(nwb_file), str(cell_count)],
                        check=False
                    )
                    if result.returncode != 0:
                        print(f"\n⚠ Processing script exited with code {result.returncode}")
                        print(f"  (Continuing with next file if available)")
                else:
                    # Warn if script not found
                    print(f"   ⚠ Mixed protocol script not found: {mixed_script}")
                    print(f"   Skipping this file.\n")
                
                # Force garbage collection after each file to free memory
                gc.collect()
        
        # Process single protocol files (all at once)
        if single_protocol_files:
            print(f"\n📊 Processing {len(single_protocol_files)} single protocol file(s)...\n")
            print(f"\n{'='*70}")
            print(f"Processing: {len(single_protocol_files)} single protocol files")
            print(f"Protocol: SINGLE")
            print('='*70)
            print(f"🔄 Single protocol detected - Using standard extraction...")
            print(f"   Launching process_human_data.py...\n")
            result = subprocess.run(
                [sys.executable, str(script_dir / 'process_human_data.py'), 
                 str(parent_dir), str(parent_dir), str(template_path)],
                check=False
            )
            if result.returncode != 0:
                print(f"\n⚠ Processing script exited with code {result.returncode}")
        
        print(f"\n{'='*70}")
        print(f"✓ Data preparation complete!")
        print('='*70)
            
    except Exception as e:
        print(f"✗ ERROR running data preparation: {e}")
        sys.exit(1)


def run_nwb_analysis():
    """Run STEP 2: Analysis on bundle directories"""
    print("\n" + "="*70)
    print("STEP 2: ANALYSIS & CLASSIFICATION (bundle_analyzer.py)")
    print("="*70)
    print("""
This step analyzes all bundles and runs spike detection.

For each bundle, this will:
  • Load parquet files (voltage + current)
  • Classify sweeps (kept vs dropped)
  • Create sweep_config.json with metadata
  • Run full analysis:
    - Resting membrane potential
    - Spike detection
    - Savitzky-Golay filtering
    - Input resistance calculation
""")
    
    # Get parent directory containing bundle directories
    parent_dir = get_parent_directory()
    parent_path = Path(parent_dir)
    
    # Look for directories that contain manifest.json (they are bundle directories)
    bundle_dirs = []
    for subfolder in parent_path.glob("*/"):
        # Look for bundle directories inside each subject subfolder
        for bundle_candidate in subfolder.glob("*"):
            if bundle_candidate.is_dir() and (bundle_candidate / "manifest.json").exists():
                bundle_dirs.append(bundle_candidate)
    
    bundle_dirs = sorted(bundle_dirs)
    
    if not bundle_dirs:
        print(f"✗ ERROR: No bundle directories found in {parent_path}")
        print(f"  Looking for directories containing manifest.json files")
        print(f"  Make sure you've completed STEP 1 (data preparation)")
        sys.exit(1)
    
    print(f"\n📊 Found {len(bundle_dirs)} bundle(s) to analyze:")
    for bundle in bundle_dirs:
        print(f"  • {bundle.relative_to(parent_path)}")
    
    # Ask about optional flags
    print("\nAnalysis options:")
    print("  1) Run full analysis (default)")
    print("  2) Only create sweep_config.json (skip spike detection)")
    
    while True:
        analysis_choice = input("Enter 1 or 2: ").strip()
        if analysis_choice in ["1", "2"]:
            break
        print("  ✗ Invalid input. Please enter 1 or 2.")
    
    try:
        from bundle_analyzer import main as analyzer_main
        
        # Process each bundle
        for idx, bundle_path in enumerate(bundle_dirs, 1):
            print(f"\n{'='*70}")
            print(f"[{idx}/{len(bundle_dirs)}] Analyzing: {bundle_path.relative_to(parent_path)}")
            print('='*70)
            
            # Set up sys.argv for argparse
            argv = ["bundle_analyzer.py", str(bundle_path)]
            
            if analysis_choice == "2":
                argv.append("--skip-analysis")
            
            sys.argv = argv
            
            try:
                analyzer_main()
                print(f"✓ Analysis complete for {bundle_path.name}")
            except Exception as e:
                print(f"⚠ ERROR analyzing {bundle_path.name}: {e}")
                print(f"  Continuing with next bundle...")
                continue
        
        print(f"\n{'='*70}")
        print(f"✓ Analysis complete!")
        print('='*70)
            
    except ImportError as e:
        print(f"✗ ERROR: Could not import bundle_analyzer.py")
        print(f"  Make sure bundle_analyzer.py is in the current directory")
        print(f"  Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ ERROR running analysis: {e}")
        sys.exit(1)


def run_nwb_pipeline():
    """Run full NWB analysis pipeline (both steps)"""
    print("\n" + "="*70)
    print("NWB ANALYSIS PIPELINE")
    print("="*70)
    print("""
This pipeline has two steps:

STEP 1: DATA PREPARATION (process_human_data.py)
  NWB files → Extract traces → Parquet files + Bundles

STEP 2: ANALYSIS & CLASSIFICATION (bundle_analyzer.py)
  Bundle directory → Classify sweeps → Run spike detection

Workflow options:
  1) Full pipeline (both steps) - for new NWB files
  2) Data preparation only (STEP 1) - prepare bundles first
  3) Analysis only (STEP 2) - for existing bundles
""")
    
    while True:
        pipeline_choice = input("Enter 1, 2, or 3: ").strip()
        if pipeline_choice in ["1", "2", "3"]:
            break
        print("  ✗ Invalid input. Please enter 1, 2, or 3.")
    
    if pipeline_choice == "1":
        # Full pipeline: both steps
        print("\n" + "="*70)
        print("FULL PIPELINE: DATA PREP + ANALYSIS")
        print("="*70)
        print("""
This will run both steps in sequence:
  1. Extract NWB files to parquets
  2. Run analysis on all created bundles

Note: You will need to provide paths for:
  - Parent directory with NWB files
  - Output directory for bundles
  - Metadata Excel template
""")
        
        run_nwb_data_preparation()
        
        # Automatically run analysis on all bundles
        print("\n" + "="*70)
        print("Data preparation complete!")
        print("="*70)
        analyze_now = input("\nRun analysis on all bundles now? (y/n): ").strip().lower()
        
        if analyze_now == "y":
            run_nwb_analysis()  # Will ask for parent directory and process all bundles
            print("\n" + "="*70)
            print("✓ PIPELINE COMPLETE")
            print("="*70)
            print()
        else:
            print("\n" + "="*70)
            print("✓ DATA PREPARATION COMPLETE")
            print("="*70)
            print("\nNext step:")
            print("  Run analysis: python main.py  (select option 2 → NWB → Analysis only)")
            print()
    
    elif pipeline_choice == "2":
        # Data preparation only
        run_nwb_data_preparation()
        
        print("\n" + "="*70)
        print("✓ DATA PREPARATION COMPLETE")
        print("="*70)
        print("\nNext step:")
        print("  Run analysis: python main.py  (select option 2 → NWB → Analysis only)")
        print()
    
    else:  # pipeline_choice == "3"
        # Analysis only
        run_nwb_analysis()
        print("\n" + "="*70)
        print("✓ ANALYSIS COMPLETE")
        print("="*70)
        print()


def main():
    """Main entry point"""
    print_header()
    
    file_type = get_file_type()
    
    if file_type == "1":
        run_abf_pipeline()
        print("\n" + "="*70)
        print("✓ PIPELINE COMPLETE")
        print("="*70)
        print()
    else:  # file_type == "2"
        run_nwb_pipeline()
        # Success message printed by run_nwb_pipeline()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
