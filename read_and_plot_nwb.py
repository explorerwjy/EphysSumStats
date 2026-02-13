"""
Script to read and plot NWB file with all trials and metadata extraction
"""

import matplotlib.pyplot as plt
import numpy as np
from pynwb import NWBHDF5IO
import warnings
import os
from datetime import datetime

# Suppress deprecation warnings
warnings.filterwarnings('ignore')

def read_and_plot_nwb(nwb_path):
    """
    Read an NWB file and plot all trials for acquisition and stimulus data
    
    Parameters:
    -----------
    nwb_path : str
        Path to the NWB file
    """
    
    # Read the NWB file
    with NWBHDF5IO(nwb_path, 'r') as io:
        nwbfile = io.read()
        
        # ===== Extract Metadata =====
        print("="*60)
        print("NWB FILE METADATA")
        print("="*60)
        print(f"Identifier: {nwbfile.identifier}")
        print(f"Session Description: {nwbfile.session_description}")
        print(f"Session Start Time: {nwbfile.session_start_time}")
        print(f"Institution: {nwbfile.institution}")
        
        print("\n--- Devices ---")
        for device_name, device in nwbfile.devices.items():
            print(f"  {device_name}")
        
        print("\n--- Intracellular Electrodes ---")
        if nwbfile.icephys_electrodes is not None:
            print(f"  Number of electrodes: {len(nwbfile.icephys_electrodes)}")
        
        print("\n--- Acquisition Data ---")
        acq_names = list(nwbfile.acquisition.keys())
        print(f"  Number of acquisition traces: {len(acq_names)}")
        print(f"  First few: {acq_names[:5]}")
        
        print("\n--- Stimulus Data ---")
        stim_names = list(nwbfile.stimulus.keys())
        print(f"  Number of stimulus traces: {len(stim_names)}")
        print(f"  First few: {stim_names[:5]}")
        
        # ===== Extract and Plot Data =====
        print("\n" + "="*60)
        print("EXTRACTING AND PLOTTING DATA")
        print("="*60)
        
        # ===== Extract and Plot Acquisition Data (all 97 trials) =====
        print("\n" + "="*60)
        print("EXTRACTING ACQUISITION DATA (ALL 97 TRIALS)")
        print("="*60)
        
        acq_data_list = []
        acq_times_list = []
        
        for acq_name in acq_names:
            acq_container = nwbfile.acquisition[acq_name]
            
            if hasattr(acq_container, 'data'):
                data = np.array(acq_container.data)
                
                # Get timestamps
                timestamps = None
                if hasattr(acq_container, 'timestamps'):
                    ts_array = np.array(acq_container.timestamps)
                    if ts_array.size == len(data):
                        timestamps = ts_array
                    elif ts_array.size == 1:
                        if hasattr(acq_container, 'rate'):
                            rate = acq_container.rate
                            timestamps = np.arange(len(data)) / rate
                
                if timestamps is None and hasattr(acq_container, 'rate'):
                    rate = acq_container.rate
                    timestamps = np.arange(len(data)) / rate
                
                if timestamps is None:
                    timestamps = np.arange(len(data))
                
                acq_data_list.append(data)
                acq_times_list.append(timestamps)
        
        print(f"Total acquisition traces extracted: {len(acq_data_list)}")
        
        # ===== Extract Stimulus Data (all 97 trials) =====
        print("\n" + "="*60)
        print("EXTRACTING STIMULUS DATA (ALL 97 TRIALS)")
        print("="*60)
        
        stim_data_list = []
        stim_times_list = []
        
        for stim_name in stim_names:
            stim_container = nwbfile.stimulus[stim_name]
            
            if hasattr(stim_container, 'data'):
                data = np.array(stim_container.data)
                
                # Get timestamps
                timestamps = None
                if hasattr(stim_container, 'timestamps'):
                    ts_array = np.array(stim_container.timestamps)
                    if ts_array.size == len(data):
                        timestamps = ts_array
                    elif ts_array.size == 1:
                        if hasattr(stim_container, 'rate'):
                            rate = stim_container.rate
                            timestamps = np.arange(len(data)) / rate
                
                if timestamps is None and hasattr(stim_container, 'rate'):
                    rate = stim_container.rate
                    timestamps = np.arange(len(data)) / rate
                
                if timestamps is None:
                    timestamps = np.arange(len(data))
                
                stim_data_list.append(data)
                stim_times_list.append(timestamps)
        
        print(f"Total stimulus traces extracted: {len(stim_data_list)}")
        
        # ===== Create Acquisition Plot =====
        print("\n" + "="*60)
        print("CREATING ACQUISITION PLOT (97 TRIALS)")
        print("="*60)
        
        # Calculate grid layout for all 97 traces
        num_traces = len(acq_data_list)
        cols = 7
        rows = (num_traces + cols - 1) // cols
        
        # Create figure with appropriate size
        fig_acq_height = max(20, 2.5 * rows)
        fig_acq = plt.figure(figsize=(22, fig_acq_height))
        fig_acq.suptitle('Acquisition Data - All 97 Trials', fontsize=16, fontweight='bold', y=0.995)
        
        # Plot all acquisition traces
        for idx, (data, times, acq_name) in enumerate(zip(acq_data_list, acq_times_list, acq_names)):
            ax = fig_acq.add_subplot(rows, cols, idx + 1)
            
            # Handle 1D and 2D data
            if data.ndim == 1:
                ax.plot(times, data, linewidth=0.5, color='steelblue')
            else:
                ax.plot(times, data[:, 0], linewidth=0.5, color='steelblue')
            
            ax.set_title(f'{idx}: {acq_name}', fontsize=7, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=6)
            ax.set_ylabel('Signal', fontsize=6)
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=5)
            
            # Add data statistics
            if data.ndim == 1:
                data_vals = data
            else:
                data_vals = data[:, 0]
            
            unit = acq_container = nwbfile.acquisition[acq_name]
            unit_str = acq_container.unit if hasattr(acq_container, 'unit') else 'N/A'
            
            stats_text = f"μ: {np.mean(data_vals):.1e}\nσ: {np.std(data_vals):.1e}"
            
            ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
                   fontsize=5, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        
        # Save the figure
        output_path_acq = nwb_path.replace('.nwb', '_acquisition_all_97_trials.png')
        plt.savefig(output_path_acq, dpi=150, bbox_inches='tight')
        print(f"\nAcquisition plot saved to: {output_path_acq}")
        
        # ===== Create Stimulus Plot =====
        print("\n" + "="*60)
        print("CREATING STIMULUS PLOT (97 TRIALS)")
        print("="*60)
        
        # Calculate grid layout for all 97 traces
        num_stim_traces = len(stim_data_list)
        cols = 7
        rows = (num_stim_traces + cols - 1) // cols
        
        # Create figure with appropriate size
        fig_stim_height = max(20, 2.5 * rows)
        fig_stim = plt.figure(figsize=(22, fig_stim_height))
        fig_stim.suptitle('Stimulus Data - All 97 Trials', fontsize=16, fontweight='bold', y=0.995)
        
        # Plot all stimulus traces
        for idx, (data, times, stim_name) in enumerate(zip(stim_data_list, stim_times_list, stim_names)):
            ax = fig_stim.add_subplot(rows, cols, idx + 1)
            
            # Handle 1D and 2D data
            if data.ndim == 1:
                ax.plot(times, data, linewidth=0.5, color='darkgreen')
            else:
                ax.plot(times, data[:, 0], linewidth=0.5, color='darkgreen')
            
            ax.set_title(f'{idx}: {stim_name}', fontsize=7, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=6)
            ax.set_ylabel('Signal', fontsize=6)
            ax.grid(True, alpha=0.2)
            ax.tick_params(labelsize=5)
            
            # Add data statistics
            if data.ndim == 1:
                data_vals = data
            else:
                data_vals = data[:, 0]
            
            stats_text = f"μ: {np.mean(data_vals):.1e}\nσ: {np.std(data_vals):.1e}"
            
            ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
                   fontsize=5, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        
        # Save the figure
        output_path_stim = nwb_path.replace('.nwb', '_stimulus_all_97_trials.png')
        plt.savefig(output_path_stim, dpi=150, bbox_inches='tight')
        print(f"\nStimulus plot saved to: {output_path_stim}")
        
        # ===== Extract and Save Comprehensive Metadata =====
        print("\n" + "="*60)
        print("EXTRACTING COMPREHENSIVE METADATA")
        print("="*60)
        
        metadata_output = []
        metadata_output.append("="*80)
        metadata_output.append("NWB FILE COMPREHENSIVE METADATA REPORT")
        metadata_output.append("="*80)
        metadata_output.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        metadata_output.append(f"File: {os.path.basename(nwb_path)}")
        metadata_output.append(f"Path: {nwb_path}")
        
        # File-level metadata
        metadata_output.append("\n" + "-"*80)
        metadata_output.append("BASIC FILE INFORMATION")
        metadata_output.append("-"*80)
        metadata_output.append(f"Identifier: {nwbfile.identifier}")
        metadata_output.append(f"Session Description: {nwbfile.session_description}")
        metadata_output.append(f"Session Start Time: {nwbfile.session_start_time}")
        metadata_output.append(f"File Create Date: {nwbfile.file_create_date}")
        metadata_output.append(f"Experimenter: {nwbfile.experimenter if nwbfile.experimenter else 'Not specified'}")
        metadata_output.append(f"Lab: {nwbfile.lab if nwbfile.lab else 'Not specified'}")
        metadata_output.append(f"Institution: {nwbfile.institution if nwbfile.institution else 'Not specified'}")
        metadata_output.append(f"Related Publications: {nwbfile.related_publications if nwbfile.related_publications else 'None'}")
        metadata_output.append(f"Keywords: {nwbfile.keywords if nwbfile.keywords else 'None'}")
        metadata_output.append(f"Protocol: {nwbfile.protocol if nwbfile.protocol else 'Not specified'}")
        metadata_output.append(f"Subject: {nwbfile.subject if nwbfile.subject else 'Not specified'}")
        metadata_output.append(f"Notes: {nwbfile.notes if nwbfile.notes else 'None'}")
        
        # Devices information
        metadata_output.append("\n" + "-"*80)
        metadata_output.append("DEVICES")
        metadata_output.append("-"*80)
        if nwbfile.devices:
            for device_name, device in nwbfile.devices.items():
                metadata_output.append(f"\nDevice: {device_name}")
                metadata_output.append(f"  Description: {device.description if hasattr(device, 'description') else 'N/A'}")
        else:
            metadata_output.append("No devices recorded")
        
        # Intracellular Electrodes information
        metadata_output.append("\n" + "-"*80)
        metadata_output.append("INTRACELLULAR ELECTRODES")
        metadata_output.append("-"*80)
        if nwbfile.icephys_electrodes:
            metadata_output.append(f"Total Electrodes: {len(nwbfile.icephys_electrodes)}\n")
            for idx, electrode in enumerate(nwbfile.icephys_electrodes):
                # Check if it's a string or object
                if isinstance(electrode, str):
                    metadata_output.append(f"Electrode {idx}: {electrode}")
                else:
                    metadata_output.append(f"Electrode: {electrode.name if hasattr(electrode, 'name') else str(electrode)}")
                    if hasattr(electrode, 'description'):
                        metadata_output.append(f"  Description: {electrode.description}")
                    if hasattr(electrode, 'location'):
                        metadata_output.append(f"  Location: {electrode.location}")
                    if hasattr(electrode, 'device'):
                        metadata_output.append(f"  Device: {electrode.device.name if hasattr(electrode.device, 'name') else str(electrode.device)}")
                    if hasattr(electrode, 'filtering'):
                        metadata_output.append(f"  Filtering: {electrode.filtering}")
                    if hasattr(electrode, 'resistance'):
                        metadata_output.append(f"  Resistance: {electrode.resistance}")
                    if hasattr(electrode, 'init_amp'):
                        metadata_output.append(f"  Initial Membrane Potential: {electrode.init_amp}")
                metadata_output.append("")
        else:
            metadata_output.append("No intracellular electrodes recorded")
        
        # Acquisition Data Summary
        metadata_output.append("\n" + "-"*80)
        metadata_output.append("ACQUISITION DATA SUMMARY")
        metadata_output.append("-"*80)
        metadata_output.append(f"Total Acquisition Traces: {len(acq_names)}\n")
        
        # Analyze acquisition data by unit
        acq_units = {}
        acq_rates = {}
        acq_sizes = {}
        
        for acq_name in acq_names:
            acq_container = nwbfile.acquisition[acq_name]
            unit = acq_container.unit if hasattr(acq_container, 'unit') else 'unknown'
            rate = acq_container.rate if hasattr(acq_container, 'rate') else 'unknown'
            
            if unit not in acq_units:
                acq_units[unit] = 0
                acq_rates[unit] = []
                acq_sizes[unit] = []
            
            acq_units[unit] += 1
            if rate != 'unknown':
                acq_rates[unit].append(rate)
            
            if hasattr(acq_container, 'data'):
                acq_sizes[unit].append(len(np.array(acq_container.data)))
        
        for unit, count in acq_units.items():
            metadata_output.append(f"Unit: {unit}")
            metadata_output.append(f"  Count: {count} traces")
            if acq_rates.get(unit):
                metadata_output.append(f"  Sampling Rates: {set(acq_rates.get(unit, []))}")
            if acq_sizes.get(unit):
                metadata_output.append(f"  Data Points: {set(acq_sizes.get(unit, []))}")
            metadata_output.append("")
        
        # Detailed Acquisition Traces
        metadata_output.append("\nDetailed Acquisition Traces:")
        for idx, acq_name in enumerate(acq_names):
            acq_container = nwbfile.acquisition[acq_name]
            metadata_output.append(f"\n  [{idx:03d}] {acq_name}")
            
            if hasattr(acq_container, 'unit'):
                metadata_output.append(f"        Unit: {acq_container.unit}")
            if hasattr(acq_container, 'rate'):
                metadata_output.append(f"        Sampling Rate: {acq_container.rate} Hz")
            if hasattr(acq_container, 'description'):
                metadata_output.append(f"        Description: {acq_container.description}")
            if hasattr(acq_container, 'data'):
                data = np.array(acq_container.data)
                metadata_output.append(f"        Data Shape: {data.shape}")
                metadata_output.append(f"        Data Type: {data.dtype}")
                metadata_output.append(f"        Min Value: {np.min(data):.6e}")
                metadata_output.append(f"        Max Value: {np.max(data):.6e}")
                metadata_output.append(f"        Mean Value: {np.mean(data):.6e}")
                metadata_output.append(f"        Std Dev: {np.std(data):.6e}")
        
        # Stimulus Data Summary
        metadata_output.append("\n" + "-"*80)
        metadata_output.append("STIMULUS DATA SUMMARY")
        metadata_output.append("-"*80)
        metadata_output.append(f"Total Stimulus Traces: {len(stim_names)}\n")
        
        # Analyze stimulus data by unit
        stim_units = {}
        stim_rates = {}
        stim_sizes = {}
        
        for stim_name in stim_names:
            stim_container = nwbfile.stimulus[stim_name]
            unit = stim_container.unit if hasattr(stim_container, 'unit') else 'unknown'
            rate = stim_container.rate if hasattr(stim_container, 'rate') else 'unknown'
            
            if unit not in stim_units:
                stim_units[unit] = 0
                stim_rates[unit] = []
                stim_sizes[unit] = []
            
            stim_units[unit] += 1
            if rate != 'unknown':
                stim_rates[unit].append(rate)
            
            if hasattr(stim_container, 'data'):
                stim_sizes[unit].append(len(np.array(stim_container.data)))
        
        for unit, count in stim_units.items():
            metadata_output.append(f"Unit: {unit}")
            metadata_output.append(f"  Count: {count} traces")
            if stim_rates.get(unit):
                metadata_output.append(f"  Sampling Rates: {set(stim_rates.get(unit, []))}")
            if stim_sizes.get(unit):
                metadata_output.append(f"  Data Points: {set(stim_sizes.get(unit, []))}")
            metadata_output.append("")
        
        # Detailed Stimulus Traces
        metadata_output.append("\nDetailed Stimulus Traces:")
        for idx, stim_name in enumerate(stim_names):
            stim_container = nwbfile.stimulus[stim_name]
            metadata_output.append(f"\n  [{idx:03d}] {stim_name}")
            
            if hasattr(stim_container, 'unit'):
                metadata_output.append(f"        Unit: {stim_container.unit}")
            if hasattr(stim_container, 'rate'):
                metadata_output.append(f"        Sampling Rate: {stim_container.rate} Hz")
            if hasattr(stim_container, 'description'):
                metadata_output.append(f"        Description: {stim_container.description}")
            if hasattr(stim_container, 'data'):
                data = np.array(stim_container.data)
                metadata_output.append(f"        Data Shape: {data.shape}")
                metadata_output.append(f"        Data Type: {data.dtype}")
                metadata_output.append(f"        Min Value: {np.min(data):.6e}")
                metadata_output.append(f"        Max Value: {np.max(data):.6e}")
                metadata_output.append(f"        Mean Value: {np.mean(data):.6e}")
                metadata_output.append(f"        Std Dev: {np.std(data):.6e}")
        
        metadata_output.append("\n" + "="*80)
        metadata_output.append("END OF METADATA REPORT")
        metadata_output.append("="*80)
        
        # Write metadata to text file
        metadata_text = "\n".join(metadata_output)
        metadata_file = nwb_path.replace('.nwb', '_metadata.txt')
        
        with open(metadata_file, 'w') as f:
            f.write(metadata_text)
        
        print(f"\nMetadata report saved to: {metadata_file}")
        
        plt.show()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)

if __name__ == '__main__':
    # Get NWB file path from user
    print("\n" + "="*60)
    print("NWB FILE ANALYZER")
    print("="*60)
    
    nwb_file = input("\nEnter the path to the NWB file: ").strip()
    
    # Check if file exists
    if not os.path.exists(nwb_file):
        print(f"\nError: File '{nwb_file}' not found!")
        exit(1)
    
    if not nwb_file.lower().endswith('.nwb'):
        print(f"\nError: File must be an NWB file (.nwb extension)!")
        exit(1)
    
    print(f"\nReading NWB file: {nwb_file}\n")
    read_and_plot_nwb(nwb_file)