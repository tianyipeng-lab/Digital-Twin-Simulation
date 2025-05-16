#!/usr/bin/env python3
"""
Generate a column‑mapping table between the LLM‑imputed response file and the
Qualtrics export (wave4_anonymized.csv).

The script produces a CSV with three columns:
    responses_column, wave4_column, description

How it works
------------
1.  Reads *all* rows of the Qualtrics file as raw text because the first few
    rows hold metadata and JSON blobs rather than data.
2.  Detects the row that contains the `{"ImportId": "QID…"}` JSON strings.
3.  Assumes the row **above** that contains the full question wording
    (Qualtrics places it there).
4.  Builds a lookup table   ImportId → (display_name, description)
5.  For each column in the LLM‑imputed file, looks up its display name and
    description—leaving blanks if not found.
6.  Writes the mapping table to `--output` (default: question_mapping.csv).

Usage
-----
$ python generate_question_mapping.py \
      --wave wave4_anonymized.csv \
      --responses responses_llm_imputed.csv \
      --output question_mapping.csv

Alternative command-line interfaces are provided for compatibility with the simulation pipeline:
$ python generate_question_mapping.py \
      --template template.json \
      --template-mapping-output output.csv \
      --wave4-csv wave4_anonymized.csv

$ python generate_question_mapping.py \
      --convert-csv \
      --csv-dir csv_comparison \
      --mapping-csv question_mapping.csv \
      --wave4-csv wave4_anonymized.csv
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import csv
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import pandas as pd


# ---------------------------------------------------------------------------
# Core mapping functionality
# ---------------------------------------------------------------------------

def detect_import_row(df: pd.DataFrame, max_scan: int = 5) -> int:
    """Return the index of the first row that contains `"ImportId"` JSON blobs.

    Parameters
    ----------
    df : pd.DataFrame
        The raw Qualtrics CSV loaded *without* headers.
    max_scan : int, default 5
        How many initial rows to inspect. Qualtrics usually stores metadata in
        the first two or three, so scanning five is safe.

    Raises
    ------
    ValueError
        If no such row is found within the scan window.
    """
    for i in range(min(max_scan, len(df))):
        if df.iloc[i].str.contains(r"\"ImportId\"", na=False).any():
            return i
    raise ValueError("Could not locate a row with ImportId JSON strings.")


def build_lookup(df: pd.DataFrame, import_row_idx: int) -> dict[str, tuple[str, str]]:
    """Create a mapping: ImportId → (display_name, description).

    Assumptions
    -----------
    * Row 0     : Qualtrics *variable names* (display names).
    * Row n     : JSON strings containing ImportId (detected above).
    * Row n - 1 : Full question wording / description.

    Returns
    -------
    dict
        Keys   : ImportId strings (e.g., "QID287_1").
        Values : Tuple(display_name, description).
    """
    if import_row_idx == 0:
        raise ValueError("ImportId row is at index 0—cannot determine description row.")

    import_row = df.iloc[import_row_idx]
    description_row = df.iloc[import_row_idx - 1]

    lookup: dict[str, tuple[str, str]] = {}
    for col_idx, cell in import_row.items():
        try:
            record = json.loads(cell)
        except (TypeError, json.JSONDecodeError):
            continue  # not a JSON cell

        imp_id = record.get("ImportId")
        if not imp_id:
            continue

        display_name = str(df.iloc[0, col_idx])
        description = str(description_row[col_idx]) if pd.notna(description_row[col_idx]) else ""
        lookup[imp_id] = (display_name, description)

    return lookup


def create_mapping(responses: pd.DataFrame, lookup: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Build the final mapping DataFrame."""
    data = [
        {
            "responses_column": col,
            "wave4_column": lookup.get(col, ("", ""))[0],
            "description": lookup.get(col, ("", ""))[1],
        }
        for col in responses.columns
    ]
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Template-based mapping for compatibility with pipeline
# ---------------------------------------------------------------------------

def extract_importid_mapping(wave4_csv_path: str) -> dict[str, str]:
    """
    Extract mapping from ImportId to column name from wave4 CSV.
    
    Parameters
    ----------
    wave4_csv_path : str
        Path to the wave4 anonymized CSV file
        
    Returns
    -------
    Dict[str, str]
        Mapping from QID (with 'QID' prefix) to column name
    """
    # Use CSV reader to properly handle the file
    import_id_mapping = {}
    
    try:
        with open(wave4_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
            # Check if we have at least 3 rows
            if len(rows) >= 3:
                # Column headers (row 1)
                headers = rows[0]
                # Find ImportId row
                import_row_idx = None
                for i in range(len(rows)):
                    if any('"ImportId"' in str(cell) for cell in rows[i]):
                        import_row_idx = i
                        break
                
                if import_row_idx is None:
                    print("Could not find ImportId row in wave4 CSV")
                    return import_id_mapping
                
                import_id_row = rows[import_row_idx]
                
                for col_idx, import_id_value in enumerate(import_id_row):
                    if col_idx < len(headers):
                        # Extract ImportId using regex
                        match = re.search(r'"ImportId"\s*:\s*"([^"]+)"', import_id_value)
                        if match:
                            qid = match.group(1)
                            column_name = headers[col_idx]
                            import_id_mapping[qid] = column_name
    except Exception as e:
        print(f"Error processing ImportId row: {e}")
    
    print(f"Extracted {len(import_id_mapping)} ImportId mappings from {wave4_csv_path}")
    return import_id_mapping


def generate_question_mapping_from_template(template_path: str, output_csv: str, wave4_csv_path: str = None):
    """
    For pipeline compatibility: Generate a mapping CSV for Qualtrics questions using the wave4 CSV directly.
    
    Parameters
    ----------
    template_path : str
        Path to the template JSON file (not used in the new implementation)
    output_csv : str
        Path to save the mapping CSV file
    wave4_csv_path : str, optional
        Path to wave4 anonymized CSV file for ImportId mapping
    """
    if not wave4_csv_path or not os.path.exists(wave4_csv_path):
        print("Error: wave4_csv_path is required for mapping generation")
        return
    
    # Load wave4 CSV with no header to preserve metadata
    wave_df = pd.read_csv(wave4_csv_path, dtype=str, header=None)
    
    # Generate empty responses DataFrame with columns matching ImportIds
    try:
        # Detect ImportId row
        import_row_idx = detect_import_row(wave_df)
        
        # Build lookup table from ImportId to column name and description
        lookup = build_lookup(wave_df, import_row_idx)
                
        # Create a DataFrame with ImportIds as columns to generate mapping
        if lookup:
            import_ids = list(lookup.keys())
            dummy_df = pd.DataFrame(columns=import_ids)
            
            # Create mapping and write out
            mapping = create_mapping(dummy_df, lookup)
            
            # Rename columns to match expected output format
            mapping = mapping.rename(columns={
                "responses_column": "column", 
                "wave4_column": "wave4_column", 
                "description": "description"
            })
        
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            
            # Save to CSV
            mapping.to_csv(output_csv, index=False)
            print(f"Generated mapping with {len(mapping)} entries to {output_csv}")
        else:
            print("No ImportIds found in wave4 CSV")
    except Exception as e:
        print(f"Error generating mapping from wave4 CSV: {e}")


# ---------------------------------------------------------------------------
# CSV conversion functions for compatibility with pipeline
# ---------------------------------------------------------------------------

def convert_csv_to_wave4_format(input_csv_path: str, output_csv_path: str = None, mapping_csv_path: str = None, wave4_csv_path: str = None):
    """
    Convert a CSV file to match the column structure of wave4_anonymized.csv
    
    Parameters
    ----------
    input_csv_path : str
        Path to the input CSV file
    output_csv_path : str, optional
        Path where the reformatted CSV should be saved. If None, will use the input filename with '_formatted' appended.
    mapping_csv_path : str, optional
        Path to the mapping CSV file
    wave4_csv_path : str, optional
        Path to the wave4 anonymized CSV file
    """
    # Set default output path if none provided
    if output_csv_path is None:
        filename = os.path.basename(input_csv_path)
        base, ext = os.path.splitext(filename)
        output_csv_path = os.path.join(os.path.dirname(input_csv_path), f"{base}_formatted{ext}")
    
    # Load the input CSV
    input_df = pd.read_csv(input_csv_path)
    print(f"Loaded {input_csv_path} with {len(input_df)} rows and {len(input_df.columns)} columns")
    
    # Load wave4_anonymized.csv to get its column structure
    if not wave4_csv_path or not os.path.exists(wave4_csv_path):
        wave4_csv_path = 'Data/wave_csv/wave4_anonymized.csv'
    
    try:
        wave4_df = pd.read_csv(wave4_csv_path, nrows=0)  # Only load headers
        wave4_columns = wave4_df.columns.tolist()
        print(f"Loaded column structure from {wave4_csv_path} ({len(wave4_columns)} columns)")
    except Exception as e:
        print(f"Error loading wave4_anonymized.csv: {e}")
        return
    
    # Load mapping if available
    column_mapping = {}
    if mapping_csv_path and os.path.exists(mapping_csv_path):
        try:
            mapping_df = pd.read_csv(mapping_csv_path)
            print(f"Loaded mapping from {mapping_csv_path} with {len(mapping_df)} entries")
            
            # Check for column/responses_column and wave4_column fields
            source_col = 'column' if 'column' in mapping_df.columns else 'responses_column'
            target_col = 'wave4_column'
            
            if source_col in mapping_df.columns and target_col in mapping_df.columns:
                for _, row in mapping_df.iterrows():
                    if pd.notna(row[source_col]) and pd.notna(row[target_col]) and row[target_col]:
                        column_mapping[row[source_col]] = row[target_col]
                        else:
                print(f"Warning: Required columns not found in mapping file. Looking for {source_col} and {target_col}")
        except Exception as e:
            print(f"Error loading mapping file: {e}")
    
    # Apply column renaming where applicable
    renamed_df = input_df.copy()
    renamed_columns = {}
    
    for col in input_df.columns:
        if col in column_mapping:
            renamed_columns[col] = column_mapping[col]
    
    if renamed_columns:
        renamed_df = renamed_df.rename(columns=renamed_columns)
        print(f"Renamed {len(renamed_columns)} columns based on mapping")
    
    # Order columns as requested
    input_columns = renamed_df.columns.tolist()
    
    # Identify columns unique to input_df and columns in both
    unique_columns = [col for col in input_columns if col not in wave4_columns]
    common_columns = [col for col in input_columns if col in wave4_columns]
    missing_columns = [col for col in wave4_columns if col not in input_columns]
    
    print(f"Found {len(unique_columns)} columns unique to input, {len(common_columns)} common columns, "
          f"and {len(missing_columns)} columns missing from input")
    
    # Create a dictionary of all column data at once to avoid fragmentation
    result_data = {}
    
    # Add unique columns first
    for col in unique_columns:
        result_data[col] = renamed_df[col].values
    
    # Add common and missing columns in wave4's order
    for col in wave4_columns:
        if col in common_columns:
            result_data[col] = renamed_df[col].values
        else:
            # Create empty column for those in wave4 but not in input
            result_data[col] = [None] * len(renamed_df)
    
    # Create DataFrame in one step to avoid fragmentation
    result_df = pd.DataFrame(result_data, index=renamed_df.index)
    
    # Save the reformatted DataFrame
    result_df.to_csv(output_csv_path, index=False)
    print(f"Saved reformatted CSV to {output_csv_path} with {len(result_df)} rows and {len(result_df.columns)} columns")


def convert_all_comparison_csvs(comparison_dir: str = 'csv_comparison', mapping_csv_path: str = None, wave4_csv_path: str = None):
    """
    Convert all CSV files in the comparison directory to match wave4_anonymized.csv format
    
    Parameters
    ----------
    comparison_dir : str
        Path to the directory containing the CSV files to convert
    mapping_csv_path : str, optional
        Path to the mapping CSV file
    wave4_csv_path : str, optional
        Path to the wave4 anonymized CSV file
    """
    target_files = [
        'responses_llm_imputed.csv', 
        'responses_wave1_3.csv', 
        'responses_wave4.csv'
    ]
    
    for filename in target_files:
        input_path = os.path.join(comparison_dir, filename)
        if os.path.exists(input_path):
            print(f"\nProcessing {filename}...")
            output_path = os.path.join(comparison_dir, f"{os.path.splitext(filename)[0]}_formatted.csv")
            convert_csv_to_wave4_format(input_path, output_path, mapping_csv_path, wave4_csv_path)
        else:
            print(f"File not found: {input_path}")


# ---------------------------------------------------------------------------
# Main CLI functions
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate mapping between responses and Qualtrics metadata."
    )
    
    # Core functionality arguments
    p.add_argument("--wave", type=Path, help="Path to wave4_anonymized.csv")
    p.add_argument("--responses", type=Path, help="Path to responses file")
    p.add_argument("--output", type=Path, default=Path("question_mapping.csv"),
                  help="Where to save the mapping CSV (default: question_mapping.csv)")
    
    # Pipeline compatibility arguments - Template-based generation
    p.add_argument("--template", help="Path to template JSON file")
    p.add_argument("--template-mapping-output", help="Output path for template-based mapping")
    p.add_argument("--wave4-csv", help="Path to wave4_anonymized.csv for ImportId mapping")
    
    # Pipeline compatibility arguments - CSV conversion
    p.add_argument("--convert-csv", action="store_true", help="Convert CSVs to match wave4 format")
    p.add_argument("--csv-dir", default="csv_comparison", help="Directory with CSVs to convert")
    p.add_argument("--mapping-csv", help="Path to mapping CSV file for conversion")
    
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    # Handle template-based generation (for pipeline compatibility)
    if args.template and args.template_mapping_output:
        print(f"Generating mapping from wave4 CSV using ImportId approach...")
        generate_question_mapping_from_template(
            args.template, 
            args.template_mapping_output,
            args.wave4_csv
        )
        return
    
    # Handle CSV conversion (for pipeline compatibility)
    if args.convert_csv:
        print(f"Converting CSV files to match wave4 format...")
        convert_all_comparison_csvs(
            args.csv_dir,
            args.mapping_csv,
            args.wave4_csv
        )
        return
    
    # Handle single CSV conversion (for pipeline compatibility)
    if args.mapping_csv and args.responses and not args.wave:
        print(f"Converting single CSV file...")
        output_path = args.output if args.output else None
        convert_csv_to_wave4_format(
            args.responses,
            output_path,
            args.mapping_csv,
            args.wave4_csv
        )
        return
    
    # Handle core mapping functionality
    if args.wave and args.responses:
        print(f"Generating mapping from responses to wave4 columns...")
        # Load files (all as strings to preserve exact content)
        wave_df = pd.read_csv(args.wave, dtype=str, header=None)
        resp_df = pd.read_csv(args.responses, dtype=str)

        # Build lookup table
        import_row_idx = detect_import_row(wave_df)
        lookup = build_lookup(wave_df, import_row_idx)

        # Create mapping and write out
        mapping = create_mapping(resp_df, lookup)
        mapping.to_csv(args.output, index=False)
        print(f"[✔] Wrote {len(mapping)} rows to {args.output}")
        return
    
    # If no recognized command pattern, show help
    if not any([
        args.template and args.template_mapping_output,
        args.convert_csv,
        args.wave and args.responses,
        args.mapping_csv and args.responses and not args.wave
    ]):
        print("No recognized command pattern. Use one of the following:")
        print("1. --wave <wave_csv> --responses <responses_csv> [--output <output_csv>]")
        print("2. --template <template_json> --template-mapping-output <output_csv> [--wave4-csv <wave4_csv>]")
        print("3. --convert-csv [--csv-dir <dir>] [--mapping-csv <mapping_csv>] [--wave4-csv <wave4_csv>]")
        print("4. --mapping-csv <mapping_csv> --responses <input_csv> [--output <output_csv>] [--wave4-csv <wave4_csv>]")


if __name__ == "__main__":
    main() 