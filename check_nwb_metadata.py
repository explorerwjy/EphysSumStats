#!/usr/bin/env python3
"""Quick script to inspect NWB file metadata"""

from pynwb import NWBHDF5IO
from pathlib import Path

import sys

if len(sys.argv) > 1:
    nwb_path = sys.argv[1]
else:
    nwb_path = "/Users/snehajaikumar/Sneha_NWB_Allen_brain/Test_NWB_Subjects/sub-1000610030/sub-1000610030_ses-1002181694_icephys.nwb"

print(f"Inspecting: {Path(nwb_path).name}\n")
print("="*70)

with NWBHDF5IO(nwb_path, 'r') as io:
    nwb = io.read()
    
    print("TOP-LEVEL ATTRIBUTES:")
    print("-"*70)
    
    # List all top-level attributes
    for attr in dir(nwb):
        if not attr.startswith('_'):
            try:
                val = getattr(nwb, attr, None)
                if val is not None and not callable(val):
                    # Truncate long values
                    val_str = str(val)
                    if len(val_str) > 100:
                        val_str = val_str[:100] + "..."
                    print(f"  {attr}: {val_str}")
            except Exception as e:
                print(f"  {attr}: <Error accessing: {e}>")
    
    print("\n" + "="*70)
    print("KEY METADATA FIELDS:")
    print("-"*70)
    
    # Check specific fields we care about
    fields = [
        'lab', 'institution', 'experimenter', 
        'session_description', 'experiment_description',
        'identifier', 'session_start_time', 'keywords'
    ]
    
    for field in fields:
        val = getattr(nwb, field, None)
        print(f"  {field}: {val}")
    
    print("\n" + "="*70)
    print("SUBJECT INFO:")
    print("-"*70)
    
    if nwb.subject:
        for attr in ['subject_id', 'age', 'sex', 'genotype', 'species', 'description']:
            val = getattr(nwb.subject, attr, None)
            print(f"  {attr}: {val}")
    else:
        print("  No subject information available")
    
    print("\n" + "="*70)
    print("STIMULUS/ACQUISITION INFO:")
    print("-"*70)
    
    if hasattr(nwb, 'stimulus') and nwb.stimulus:
        print(f"  Stimulus series: {list(nwb.stimulus.keys())[:5]}")
    
    if hasattr(nwb, 'acquisition') and nwb.acquisition:
        print(f"  Acquisition series: {list(nwb.acquisition.keys())[:5]}")

print("\n" + "="*70)
print("DONE")
print("="*70)
