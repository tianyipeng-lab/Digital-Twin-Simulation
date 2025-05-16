#!/usr/bin/env python3
"""
Convert CSV files to match wave4_anonymized.csv column format using a simple column mapping.

This script:
1. Can extract ImportId to column name mappings from wave4_anonymized.csv
2. Reads a manual column mapping file (if provided)
3. Applies mapping to rename columns in the input CSV files
4. Creates formatted CSV files using wave4's structure as a template
5. Preserves additional columns not found in the mapping
"""
import argparse
import os
import re
import pandas as pd
import csv
import numpy as np
import glob

def extract_importid_mapping(wave_csv_path):
    """
    Extract mapping from ImportId to column name from wave4 CSV.
    
    Parameters
    ----------
    wave_csv_path : str
        Path to the wave4 CSV file
        
    Returns
    -------
    dict
        Mapping from ImportId to column name and description
    dict
        Mapping from column name to description
    dict
        Raw descriptions from wave4 CSV by column name
    """
    # Use CSV reader to properly handle the file format
    with open(wave_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    # Get column names from the first row
    if not rows:
        print(f"Warning: CSV file {wave_csv_path} is empty")
        return {}, {}, {}
    
    column_names = rows[0]
    
    # Find the row with ImportId strings - these typically appear in the 3rd row (index 2)
    import_row_idx = None
    for i in range(min(5, len(rows))):
        for cell in rows[i]:
            if 'ImportId' in cell:
                import_row_idx = i
                break
        if import_row_idx is not None:
            break
    
    # If we couldn't find it, fall back to the traditional location (row 3)
    if import_row_idx is None:
        if len(rows) >= 3:
            import_row_idx = 2  # Traditionally in the 3rd row (0-indexed)
            print(f"Warning: Using default ImportId row (3rd row)")
        else:
            print(f"Warning: Could not find ImportId row in CSV and not enough rows for default")
            return {}, {}, {}
    
    # Description row is typically the row above the ImportId row
    description_row_idx = import_row_idx - 1
    if description_row_idx < 0:
        print(f"Warning: No description row available")
        descriptions = [""] * len(column_names)
    else:
        descriptions = rows[description_row_idx]
        # Ensure descriptions list is the same length as column_names
        if len(descriptions) < len(column_names):
            descriptions.extend([""] * (len(column_names) - len(descriptions)))
    
    # Extract ImportIds from the identified row
    mapping = {}
    descriptions_by_column = {}
    raw_wave4_descriptions = {}  # Store raw wave4 descriptions by column name
    import_row = rows[import_row_idx]
    
    for i, cell in enumerate(import_row):
        if i < len(column_names):
            column_name = column_names[i]
            
            # Save the raw description for this column
            if i < len(descriptions):
                raw_wave4_descriptions[column_name] = descriptions[i]
            
            # Save the description for this column
            if i < len(descriptions):
                descriptions_by_column[column_name] = descriptions[i]
            else:
                descriptions_by_column[column_name] = ""
                
            # Extract ImportId using regex
            match = re.search(r'"ImportId"\s*:\s*"([^"]+)"', cell)
            if match:
                qid = match.group(1)
                # Store the mapping from wave4 column name to ImportId
                mapping[column_name] = qid
    
    print(f"Extracted {len(mapping)} ImportId mappings from wave4 CSV")
    
    return mapping, descriptions_by_column, raw_wave4_descriptions

def load_column_mapping(mapping_file_path):
    """
    Load column mapping from a CSV file.
    
    Expected CSV format: 
    wave4_column_name,input_column_name
    Q156_1,QID156
    Form A _1,QID154
    
    This creates a mapping FROM wave4 column names TO input column names
    Example: 17_Q295 -> QID9_17
    
    Parameters
    ----------
    mapping_file_path : str
        Path to the mapping CSV file
    
    Returns
    -------
    dict
        Mapping from wave4 column names to input column names and descriptions
    """
    if not os.path.exists(mapping_file_path):
        print(f"Warning: Mapping file not found: {mapping_file_path}")
        return {}, {}
    
    mapping = {}
    descriptions = {}
    
    try:
        mapping_df = pd.read_csv(mapping_file_path)
        
        # Check for our expected column names or fall back to standard format
        if 'wave4_column_name' in mapping_df.columns and 'input_column_name' in mapping_df.columns:
            source_col = 'wave4_column_name'
            target_col = 'input_column_name'
        elif 'source_id' in mapping_df.columns and 'target_id' in mapping_df.columns:
            source_col = 'source_id'
            target_col = 'target_id'
        else:
            print(f"Warning: Mapping file must contain either columns: wave4_column_name, input_column_name OR source_id, target_id")
            return {}, {}
        
        # Build mapping dictionary (wave4 column name -> input column name)
        for _, row in mapping_df.iterrows():
            if pd.notna(row[source_col]) and pd.notna(row[target_col]):
                wave4_col = str(row[source_col])  # This is the wave4 column name (e.g. 17_Q295)
                input_col = str(row[target_col])  # This is the input column name (e.g. QID9_17)
                mapping[wave4_col] = input_col
                
                # Store description if available
                if 'description' in mapping_df.columns and pd.notna(row['description']):
                    descriptions[wave4_col] = row['description']
                else:
                    descriptions[wave4_col] = ""
        
        print(f"Loaded {len(mapping)} column mappings from {mapping_file_path}")
        return mapping, descriptions
    
    except Exception as e:
        print(f"Error loading mapping file: {e}")
        return {}, {}

def convert_csv_files(input_dir, output_dir, mapping, descriptions, wave4_csv_path, input_file=None, process_standard_files=False):
    """
    Convert CSV files in the input directory using the column mapping.
    
    Parameters
    ----------
    input_dir : str
        Directory containing CSV files to convert
    output_dir : str
        Base directory to save converted CSV files (subdirectories will be created here)
    mapping : dict
        Mapping from wave4 column names to input column names
    descriptions : dict
        Mapping from wave4 column names to descriptions
    wave4_csv_path : str
        Path to wave4 CSV file to use as template for column order
    input_file : str, optional
        Specific file to process (if None, process all matching files)
    process_standard_files : bool, optional
        If True, process standard numeric and text files in a predefined pattern
    """
    # Create base output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define and create specific subdirectories for formatted files
    csv_formatted_dir = os.path.join(output_dir, "csv_formatted")
    csv_formatted_label_dir = os.path.join(output_dir, "csv_formatted_label")
    os.makedirs(csv_formatted_dir, exist_ok=True)
    os.makedirs(csv_formatted_label_dir, exist_ok=True)
    
    # Get wave4 column structure for ordering
    wave4_df = pd.read_csv(wave4_csv_path, nrows=0)
    wave4_columns = wave4_df.columns.tolist()
    
    # Ensure TWIN_ID will be the first column in the output
    # First remove it from wave4_columns if it exists
    if 'TWIN_ID' in wave4_columns:
        wave4_columns.remove('TWIN_ID')
    # Then add it back at the beginning
    wave4_columns = ['TWIN_ID'] + wave4_columns
    
    # Reverse the mapping (from wave4 names to input names)
    # This is because our mapping file has wave4_column_name -> input_column_name
    # But we need to map from input column names to wave4 column names
    reverse_mapping = {}
    for wave4_name, input_name in mapping.items():
        reverse_mapping[input_name] = wave4_name
    
    # Define the specific files to process
    if process_standard_files:
        numeric_files = [
            "responses_wave1_3.csv",
            "responses_wave4.csv",
            "responses_llm_imputed.csv"
        ]
        
        text_files = [ # These are already label files
            "responses_wave1_3_label.csv",
            "responses_wave4_label.csv", 
            "responses_llm_imputed_label.csv"
        ]
        
        # Process numeric files
        print("Processing numeric format files:")
        for filename in numeric_files:
            input_path = os.path.join(input_dir, filename)
            if os.path.exists(input_path):
                # Numeric files go to csv_formatted, their labels go to csv_formatted_label
                process_file(input_path, csv_formatted_dir, csv_formatted_label_dir, wave4_columns, reverse_mapping, descriptions)
            else:
                print(f"  Warning: File not found - {filename}")
        
        # Process text files (which are label files)
        print("Processing text format files:")
        for filename in text_files:
            input_path = os.path.join(input_dir, filename)
            if os.path.exists(input_path):
                # Label files are processed and their formatted versions go to csv_formatted_label_dir
                # No secondary label file is generated for these.
                process_file(input_path, csv_formatted_label_dir, None, wave4_columns, reverse_mapping, descriptions)
            else:
                print(f"  Warning: File not found - {filename}")
        
        return
    
    # Process a specific file if specified
    if input_file:
        input_path = os.path.join(input_dir, input_file)
        if os.path.exists(input_path):
            base_filename = os.path.splitext(os.path.basename(input_file))[0]
            is_label_file = base_filename.endswith('_label')
            if is_label_file:
                # Label files are processed and their formatted versions go to csv_formatted_label_dir
                process_file(input_path, csv_formatted_label_dir, None, wave4_columns, reverse_mapping, descriptions)
            else:
                # Numeric files go to csv_formatted, their labels go to csv_formatted_label
                process_file(input_path, csv_formatted_dir, csv_formatted_label_dir, wave4_columns, reverse_mapping, descriptions)
        else:
            print(f"Error: Input file not found: {input_path}")
        return
    
    # Process all CSV files in the input directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv') and not f.endswith('_formatted.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return
    
    for filename in csv_files:
        input_path = os.path.join(input_dir, filename)
        base_filename = os.path.splitext(filename)[0]
        is_label_file = base_filename.endswith('_label')
        if is_label_file:
            process_file(input_path, csv_formatted_label_dir, None, wave4_columns, reverse_mapping, descriptions)
        else:
            process_file(input_path, csv_formatted_dir, csv_formatted_label_dir, wave4_columns, reverse_mapping, descriptions)

def process_file(input_path, primary_output_dir, label_output_dir_if_applicable, wave4_columns, reverse_mapping, descriptions):
    """
    Process a single CSV file and convert it to match wave4 format.
    
    Parameters
    ----------
    input_path : str
        Path to the input CSV file
    primary_output_dir : str
        Directory to save the primary converted CSV file 
        (e.g., csv_formatted for numeric, csv_formatted_label for label files)
    label_output_dir_if_applicable : str or None
        Directory to save the corresponding formatted label CSV file (if applicable, e.g., for numeric files).
        None if the input is already a label file or no label version is needed.
    wave4_columns : list
        List of column names from wave4 CSV
    reverse_mapping : dict
        Mapping from input column names to wave4 column names
    descriptions : dict
        Mapping from wave4 column names to descriptions
    """
    filename = os.path.basename(input_path)
    base_filename = os.path.splitext(filename)[0]
    
    # Determine if this is an original label file (e.g. ends with _label.csv)
    is_original_label_file = base_filename.endswith('_label')

    # Define the output path for the primary processed file
    # If it's an original label file, it goes to primary_output_dir (which should be csv_formatted_label_dir)
    # If it's a numeric file, it goes to primary_output_dir (which should be csv_formatted_dir)
    output_path = os.path.join(primary_output_dir, f"{base_filename}_formatted.csv")
    
    print(f"Processing {filename}...")
    
    # Load input CSV
    input_df = pd.read_csv(input_path)
    
    # Look for the TWIN_ID column in the input data
    twin_id_col = None
    for col in input_df.columns:
        # Check for exact match or case insensitive match
        if col == 'TWIN_ID' or col.upper() == 'TWIN_ID':
            twin_id_col = col
            break
        # Check if this column maps to TWIN_ID in our mapping
        elif col in reverse_mapping and reverse_mapping[col] == 'TWIN_ID':
            twin_id_col = col
            break
    
    # Initialize output structure using wave4's columns
    result_data = {col: [None] * len(input_df) for col in wave4_columns}
    additional_columns = {}  # For columns not in wave4
    
    # If we found a TWIN_ID column, make sure it gets mapped to the right place
    if twin_id_col:
        result_data['TWIN_ID'] = input_df[twin_id_col].values
    
    # Track mapping statistics
    mapped_count = 0
    unmapped_count = 0
    
    # Process each column in the input file
    for col in input_df.columns:
        # Skip TWIN_ID as we've already handled it
        if col == twin_id_col:
            continue
            
        # Check if this input column maps to a wave4 column
        if col in reverse_mapping:
            wave4_col = reverse_mapping[col]
            
            # Skip if it's TWIN_ID (already handled)
            if wave4_col == 'TWIN_ID':
                continue
            
            # If wave4 column is in our expected structure, place data there
            if wave4_col in wave4_columns:
                result_data[wave4_col] = input_df[col].values
                mapped_count += 1
            # Otherwise add to additional columns
            else:
                additional_columns[wave4_col] = input_df[col].values
                mapped_count += 1
        else:
            # No mapping found, keep original column name
            # Don't duplicate TWIN_ID
            if col.upper() != 'TWIN_ID':
                additional_columns[col] = input_df[col].values
                unmapped_count += 1
                print(f"  - Unmapped: {col}")
    
    # Add unmapped columns to result data
    for col, values in additional_columns.items():
        if col not in result_data:  # Only add if not already present
            result_data[col] = values
    
    # Create final DataFrame
    result_df = pd.DataFrame(result_data)
    
    # Report mapping statistics
    print(f"  - Mapped columns: {mapped_count}")
    print(f"  - Unmapped columns: {unmapped_count}")
    print(f"  - Total wave4 columns: {len(wave4_columns)}")
    print(f"  - Additional columns: {len(additional_columns)}")
    
    # Save the CSV with descriptions as the second row
    headers = list(result_df.columns)
    
    # Prepare description row
    description_row = []
    for col in headers:
        description_row.append(descriptions.get(col, ""))
    
    # Write CSV with description row
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header row
        writer.writerow(headers)
        # Write description row
        writer.writerow(description_row)
        # Write data rows, replacing NaN with empty string
        for _, row in result_df.iterrows():
            # Convert NaN values to empty strings
            processed_row = []
            for value in row:
                if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
                    processed_row.append("")
                else:
                    processed_row.append(value)
            writer.writerow(processed_row)
    
    print(f"  - Saved to {output_path}")
    
    # If this isn't an original label file AND a directory for label outputs is provided,
    # try to find and process its corresponding label file.
    if not is_original_label_file and label_output_dir_if_applicable:
        # Look for the label version of the input file (e.g., input_file_label.csv)
        corresponding_label_input_filename = f"{base_filename}_label.csv"
        label_input_path = os.path.join(os.path.dirname(input_path), corresponding_label_input_filename)
        
        if os.path.exists(label_input_path):
            # The output name for this formatted label file will be like base_filename_label_formatted.csv
            # and it will be saved in label_output_dir_if_applicable
            formatted_label_output_filename = f"{base_filename}_label_formatted.csv"
            label_version_output_path = os.path.join(label_output_dir_if_applicable, formatted_label_output_filename)

            print(f"Creating formatted label version from {corresponding_label_input_filename}...")
            
            # Load label CSV
            label_df = pd.read_csv(label_input_path)
            
            # Use the same mapping approach as above but with the label file
            label_result_data = {col: [None] * len(label_df) for col in wave4_columns}
            label_additional_columns = {}
            
            # Map TWIN_ID from the label file
            # We need to re-check for twin_id_col in label_df as it might be named differently or absent
            label_twin_id_col = None
            for col_label in label_df.columns:
                if col_label == 'TWIN_ID' or col_label.upper() == 'TWIN_ID':
                    label_twin_id_col = col_label
                    break
                elif col_label in reverse_mapping and reverse_mapping[col_label] == 'TWIN_ID':
                    label_twin_id_col = col_label
                    break
            
            if label_twin_id_col:
                label_result_data['TWIN_ID'] = label_df[label_twin_id_col].values
            
            # Process each column in the label file
            for col in label_df.columns:
                if col == label_twin_id_col: # Use the twin_id_col specific to label_df
                    continue
                    
                if col in reverse_mapping:
                    wave4_col = reverse_mapping[col]
                    
                    if wave4_col == 'TWIN_ID':
                        continue
                    
                    if wave4_col in wave4_columns:
                        label_result_data[wave4_col] = label_df[col].values
                    else:
                        label_additional_columns[wave4_col] = label_df[col].values
                else:
                    if col.upper() != 'TWIN_ID': # Ensure not to add original TWIN_ID again if unmapped
                        label_additional_columns[col] = label_df[col].values
            
            # Add unmapped columns
            for col, values in label_additional_columns.items():
                if col not in label_result_data:
                    label_result_data[col] = values
            
            # Create label DataFrame
            label_result_df = pd.DataFrame(label_result_data)
            
            # Write label CSV with descriptions
            with open(label_version_output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Write header row
                writer.writerow(list(label_result_df.columns))
                # Write description row
                description_row = [descriptions.get(col, "") for col in label_result_df.columns]
                writer.writerow(description_row)
                # Write data rows
                for _, row in label_result_df.iterrows():
                    processed_row = []
                    for value in row:
                        if pd.isna(value) or (isinstance(value, float) and np.isnan(value)):
                            processed_row.append("")
                        else:
                            processed_row.append(value)
                    writer.writerow(processed_row)
            
            print(f"  - Saved label version to {label_version_output_path}")
        else:
            print(f"  - Warning: No corresponding label file found at {label_input_path}. Formatted label version not created for {base_filename}.")

def main():
    parser = argparse.ArgumentParser(description="Convert CSV files to match wave4 format using a column mapping")
    parser.add_argument("--csv-dir", required=True, help="Directory containing CSV files to convert")
    parser.add_argument("--output-dir", help="Directory to save converted CSV files (defaults to csv-dir)")
    parser.add_argument("--wave4-csv", default="data/wave_csv/wave4_anonymized.csv",
                        help="Path to wave4_anonymized.csv (default: data/wave_csv/wave4_anonymized.csv)")
    parser.add_argument("--manual-mapping", default="data/wave_csv/column_mapping.csv",
                        help="Path to column mapping CSV (default: data/wave_csv/column_mapping.csv)")
    parser.add_argument("--save-mapping", help="Path to save the mapping for reference")
    parser.add_argument("--use-importid", action="store_true", 
                        help="Extract ImportId mapping from wave4_csv (for backward compatibility)")
    parser.add_argument("--input-file", help="Process only a specific file in the input directory")
    parser.add_argument("--process-standard-files", action="store_true",
                        help="Process standard numeric and text files")
    parser.add_argument("--merge-mapping", action="store_true",
                        help="Merge automatic and manual mappings (manual takes precedence only for conflicts)")
    args = parser.parse_args()
    
    # Set output directory to input directory if not specified
    output_dir = args.output_dir if args.output_dir else args.csv_dir
    
    # Initialize mapping
    mapping = {}
    descriptions = {}
    automatic_mapping = {}
    automatic_descriptions = {}
    raw_wave4_descriptions = {}  # Store raw descriptions from wave4
    
    # Extract ImportId mapping from wave4 CSV if requested
    if args.use_importid:
        print(f"Extracting ImportId mapping from {args.wave4_csv}")
        automatic_mapping, automatic_descriptions, raw_wave4_descriptions = extract_importid_mapping(args.wave4_csv)
        print(f"Using {len(automatic_mapping)} mappings from wave4 ImportId")
        
        # Store automatic mappings
        mapping.update(automatic_mapping)
        descriptions.update(automatic_descriptions)
    else:
        # Even if not using ImportId mapping, get the raw descriptions from wave4
        _, _, raw_wave4_descriptions = extract_importid_mapping(args.wave4_csv)
    
    # Load column mapping from file 
    manual_mapping, manual_descriptions = load_column_mapping(args.manual_mapping)
    if manual_mapping:
        print(f"Using {len(manual_mapping)} mappings from manual mapping file")
        
        # If we're merging mappings, only override automatic mappings for columns that exist in manual mapping
        if args.merge_mapping and automatic_mapping:
            # Identify overlapping columns
            overlap = set(automatic_mapping.keys()).intersection(set(manual_mapping.keys()))
            print(f"Found {len(overlap)} overlapping column mappings")
            
            # Update only overlapping columns with manual values
            for col in overlap:
                mapping[col] = manual_mapping[col]
                # Use wave4 description if available, otherwise use manual description
                descriptions[col] = raw_wave4_descriptions.get(col, manual_descriptions.get(col, ""))
            
            # Add non-overlapping manual mappings
            for col in set(manual_mapping.keys()) - overlap:
                mapping[col] = manual_mapping[col]
                descriptions[col] = raw_wave4_descriptions.get(col, manual_descriptions.get(col, ""))
        else:
            # Otherwise completely replace with manual mappings (old behavior)
            mapping.update(manual_mapping)
            # For each manual mapping, use wave4 description if available
            for col in manual_mapping:
                descriptions[col] = raw_wave4_descriptions.get(col, manual_descriptions.get(col, ""))
    
    if not mapping:
        print("No valid column mapping found. Please check your mapping files or use --use-importid.")
        return
    
    # Run the conversion
    if os.path.exists(args.csv_dir):
        convert_csv_files(args.csv_dir, output_dir, mapping, descriptions, args.wave4_csv, 
                         args.input_file, args.process_standard_files)
    else:
        print(f"Error: CSV directory not found: {args.csv_dir}")
    
    # Save mapping for reference if requested
    if args.save_mapping:
        # Create a DataFrame from the mapping and descriptions
        mapping_rows = []
        # Read wave4 column order
        wave4_order_list = []
        try:
            wave4_df_for_order = pd.read_csv(args.wave4_csv, nrows=0)
            wave4_order_list = wave4_df_for_order.columns.tolist()
        except Exception as e:
            print(f"Warning: Could not read wave4 column order: {e}")
            wave4_order_list = []
        
        for wave4_col, input_col in mapping.items():
            # Determine the source of this mapping
            source = "manual" if wave4_col in manual_mapping else "automatic"
            # Determine the order index in wave4 columns
            wave4_order = wave4_order_list.index(wave4_col) if wave4_col in wave4_order_list else -1
            # Create a simpler mapping record with just the essential columns
            mapping_rows.append({
                'source_id': wave4_col,           # wave4 column name (e.g. 17_Q295)
                'target_id': input_col,           # input column name (e.g. QID9_17)
                'description': descriptions.get(wave4_col, ""),  # Now includes wave4 descriptions
                'QuestionID': wave4_col,          # Kept for compatibility with evaluate_responses.py
                'mapping_source': source,         # Added to track mapping source
                'wave4_order': wave4_order        # New: order in wave4_anonymized.csv
            })
        
        # Save to CSV
        mapping_df = pd.DataFrame(mapping_rows)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.save_mapping), exist_ok=True)
        
        # Save the mapping
        mapping_df.to_csv(args.save_mapping, index=False)
        print(f"Saved reference mapping to {args.save_mapping} with {len(mapping_df)} entries")
        
        # Print statistics
        auto_count = len([r for r in mapping_rows if r['mapping_source'] == 'automatic'])
        manual_count = len([r for r in mapping_rows if r['mapping_source'] == 'manual'])
        print(f"  - Automatic mappings: {auto_count}")
        print(f"  - Manual mappings: {manual_count}")

if __name__ == "__main__":
    main() 