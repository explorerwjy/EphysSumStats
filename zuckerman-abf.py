# Version 01-09
# Original: Sneha Jaikumar
# Adapted by: Manos
# Date: 2026-01-08
import json
import os
import re
import pyabf
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
from run_analysis import run_for_bundle

def parse_abf_filename(fname: str):
    """
    Parse ABF filename like 2025_06_10_0003.abf
    Return (recDate, fileNum) where:
      recDate = YYYYMMDD
      fileNum = last 2 digits, matching Excel (e.g. '03')
    """
    base = os.path.basename(fname)
    m = re.match(r"(\d{4})[_-]?(\d{2})[_-]?(\d{2})[_-]?(\d{3,4})", base)
    if not m:
        raise ValueError(f"Unexpected filename format: {base}")

    year, month, day, num = m.groups()
    rec_date = f"{year}{month}{day}"
    file_num = num[-2:]   # keep only last 2 digits

    # Construct file id
    run_4d   = str(int(num)).zfill(4) # 4-digit run number for display/folders
    file_id  = f"{year}_{month}_{day}_{run_4d}"
    
    return rec_date, file_num, file_id

def build_long_tables_from_abf(abf_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (df_mV, df_pA) as tidy/long DataFrames with columns:
      ['sweep','kind','channel_index','channel_name','unit','t_s','value']
    kind is 'ADC' or 'DAC'
    """
    abf = pyabf.ABF(abf_path)
    rows = []
    fs_hz_sample = getattr(abf, "sampleRate", None)

    # extra ABF-derived metadata 
    abf_meta = {
        "sampleRate_Hz": int(fs_hz_sample),
        "sweepCount": int(getattr(abf, "sweepCount", len(abf.sweepList))),
        "sweepLength_sec": float(getattr(abf, "sweepLengthSec", 0.0)),
        "protocol": getattr(abf, "protocol", None),
    }

    # loop ADC channels
    for sweep in abf.sweepList:
        for ch in range(abf.channelCount):
            abf.setSweep(sweepNumber=sweep, channel=ch)
            t = abf.sweepX
            y = abf.sweepY
            name = (abf.adcNames[ch] if hasattr(abf, "adcNames") else f"ADC{ch}").strip()
            unit = abf.adcUnits[ch] if hasattr(abf, "adcUnits") else ""
            rows.append(pd.DataFrame({
                "sweep": sweep,
                "kind": "ADC",
                "channel_index": ch,
                "channel_name": name,
                "unit": unit,
                "t_s": t,
                "value": y
            }))

    long_df = pd.concat(rows, ignore_index=True)

    # split by unit
    df_pA = long_df[long_df["unit"] == "pA"].copy()
    df_mV = long_df[long_df["unit"] == "mV"].copy()

    # remove the I_Clamp rows from pA
    # if not df_pA.empty and "channel_name" in df_pA.columns:
    #     df_pA = df_pA[~df_pA["channel_name"].str.fullmatch(r"I_clamp", case=False, na=False)].copy()

    return (df_mV.reset_index(drop=True), df_pA.reset_index(drop=True), abf_meta)


# ----------------------------
# Excel metadata handling
# ----------------------------

def load_excel_meta(excel_path: str) -> Dict[Tuple[str, str], dict]:
    """
    Load Excel and return a dict keyed by (recDate, fileNum) -> normalized row-dict
    - recDate should be like 20250923 (string or number)
    - fileNum should be like 001 (zero-padded to 3)
    All columns are normalized to snake_case keys.
    """
    df = pd.read_excel(excel_path, header=2)

    df = df.loc[:, ~df.columns.str.match(r"Unnamed")]

    # ensure recDate and fileNum exist
    if "recDate" not in df.columns or "fileNum" not in df.columns:
        raise ValueError("Excel must contain columns 'recDate' and 'fileNum'.")

    meta_map = {}
    df["recDate"] = (
        df["recDate"].astype(str)
                     .str.replace(r"\.0$","",regex=True)
                     .str.strip()
    ) 

    # filenum in Excel is 2 digits; ensure exactly 2 digits (keep last 2 if someone typed 003, 0018, etc.)
    df["fileNum"] = (
        df["fileNum"].astype(str)
                     .str.replace(r"\.0$","",regex=True)
                     .str.strip()
                     .str.zfill(2)                              
    )

    meta_map = { (row["recDate"], row["fileNum"]): dict(row)
                 for _, row in df.iterrows() }

    return meta_map


# ----------------------------
# Save bundle
# ----------------------------
def save_bundle(file_id: str,
                cell_num: str,
                abf_path: str,
                df_mV: pd.DataFrame,
                df_pA: pd.DataFrame,
                meta_full: dict,
                out_root: str):
    out = Path(out_root) / f"{file_id}_{cell_num}"
    out.mkdir(parents=True, exist_ok=True)

    mv_path = out / f"mv_{cell_num}.parquet"
    pa_path = out /  f"pa_{cell_num}.parquet"
    # mv_csv_path = out / "mV.csv"
    # pa_csv_path = out / "pA.csv"
    df_mV.to_parquet(mv_path, index=False)
    # df_mV.to_csv(mv_csv_path, index = False)
    df_pA.to_parquet(pa_path, index=False)
    # df_pA.to_csv(pa_csv_path, index = False)

    manifest = {
        "file_id": file_id,
        "abf_path": os.path.abspath(abf_path),
        "tables": {"mv": mv_path.name, "pa": pa_path.name},
        "meta": meta_full       # ALL Excel columns (future-proof) # sex, recGoal, dyeFill, cellNum, fileNum, cellType, remarks (snake_case)
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)


# ----------------------------
# Main driver
# ----------------------------
def process_mouse_folder(
    mouse_dir: str,
    excel_path: str,
    out_root: str
):
    """
    Walk mouse_dir for *.abf, parse each, match to Excel by (recDate, fileNum),
    and write bundles with mv/parquet, pa/parquet, and manifest.json.
    """
    meta_map = load_excel_meta(excel_path)
    Path(out_root).mkdir(parents=True, exist_ok=True)

    abf_files = sorted(
        str(p) for p in Path(mouse_dir).glob("*.abf")
    )

    if not abf_files:
        print(f"[WARN] No ABF files found in {mouse_dir}")
        return

    for abf_path in abf_files:
        try:
            recDate, fileNum, file_id = parse_abf_filename(abf_path)
            if not recDate or not fileNum:
                print(f"[SKIP] Unrecognized ABF name format: {abf_path}")
                continue

            key = (recDate, fileNum)
            print("KEY",key)
            if key not in meta_map:
                print(f"[MISS] No Excel metadata for {file_id} (recDate={recDate}, fileNum={fileNum})")
                continue

            meta_full = meta_map[key]

            df_mV, df_pA, abf_meta = build_long_tables_from_abf(abf_path)

            meta_full.update(abf_meta)
            print("updated", meta_full)

            if df_mV.empty and df_pA.empty:
                print(f"[SKIP] No mV/pA data found in {abf_path}")
                continue

            cell_num = str(meta_full.get("cellNum")).strip()
            cell_num = re.sub(r"\.0+$", "", cell_num)

            save_bundle(
                file_id=file_id,
                cell_num = cell_num,
                abf_path=abf_path,
                df_mV=df_mV,
                df_pA=df_pA,
                meta_full=meta_full,
                out_root=out_root
            )
            print(f"[OK] {file_id} -> {out_root}/{file_id}_{cell_num}")

        except Exception as e:
            print(f"[ERROR] {abf_path}: {e}")


# ----------------------------
# Run it (edit paths)
# ----------------------------
if __name__ == "__main__":
    
    input_dir = input("Please input the path to the directory: ")
    # Excel file with recDate/fileNum
    excel_path = input("Please input the path to the excel metadata: ")

    #Type of data
    data_type = input("Enter 1 for mice data or 2 for human: ")

    #Only applicable when processing mice data, comment out otherwise
    # process_mouse_folder(
    #     mouse_dir= input_dir,      
    #     excel_path= excel_path,   
    #     out_root= input_dir             
    # )

    input_dir = Path(input_dir)
    # This logic assumes there are NO OTHER SUBFOLDERS in the mouse folder
    bundle_dirs = sorted(
        [d for d in input_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name
    )
    for bundle in bundle_dirs:
        # if bundle.name == "2025_10_01_0008_525":
        manifest_path = bundle / "manifest.json"
        print(manifest_path)
        # Skip directories that don't have a manifest.json (e.g., Data/ folders, other non-bundle dirs)
        if not manifest_path.exists():
            print(f"Skipping {bundle.name} (no manifest.json found)")
            continue
        try:
            if data_type == "1":
                with open(manifest_path, "r") as f:
                    man = json.load(f)
                protocol = str(man.get("meta", {}).get("protocol", "")).lower()
                if "step" in protocol:
                    print(f"Running bundle {bundle.name}")
                    run_for_bundle(bundle)
                else:
                    print(f"Skipping Bundle: {bundle.name}")
            else:
                print(f"Running bundle {bundle.name}")
                run_for_bundle(bundle)

        except Exception as e:
            print(f"ERROR processing bundle {bundle.name}: {e}")
