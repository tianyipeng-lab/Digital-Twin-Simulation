import os
import json
import pandas as pd
import glob
import argparse
import yaml
import re
from pathlib import Path
import sys

# Add the evaulation directory to the path so we can import from it
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.extract_mega_persona_csv import process_mega_persona, extract_answers_from_mega_persona_json

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def identify_question_type(file_path, question_id):
    """
    Identify the type of a specific question from the answer block.
    
    Parameters:
    -----------
    file_path : str
        Path to the sample answer block
    question_id : str
        Question ID to look for
    
    Returns:
    --------
    str
        Question type (MC, Matrix, TE, etc.)
    dict
        Additional information about the question
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            blocks = json.load(f)
        
        # Ensure blocks is a list
        blocks_list = blocks if isinstance(blocks, list) else [blocks]
        
        for block in blocks_list:
            if not isinstance(block, dict):
                continue
                
            # Look in Questions
            if 'Questions' in block:
                for question in block['Questions']:
                    if question.get('QuestionID') == question_id:
                        return question.get('QuestionType', ''), {
                            'settings': question.get('Settings', {}),
                            'rows': question.get('Rows', []),
                            'columns': question.get('Columns', []),
                            'choices': question.get('Choices', []),
                            'statements': question.get('Statements', []),
                            'statements_id': question.get('StatementsID', []),
                            'rows_id': question.get('RowsID', []),
                            'choices_id': question.get('ChoicesID', []),
                            'columns_id': question.get('ColumnsID', [])
                        }
            
            # Look in Elements
            if 'Elements' in block:
                for element in block['Elements']:
                    if isinstance(element, dict) and 'Questions' in element:
                        for question in element['Questions']:
                            if question.get('QuestionID') == question_id:
                                return question.get('QuestionType', ''), {
                                    'settings': question.get('Settings', {}),
                                    'rows': question.get('Rows', []),
                                    'columns': question.get('Columns', []),
                                    'choices': question.get('Choices', []),
                                    'statements': question.get('Statements', []),
                                    'statements_id': question.get('StatementsID', []),
                                    'rows_id': question.get('RowsID', []),
                                    'choices_id': question.get('ChoicesID', []),
                                    'columns_id': question.get('ColumnsID', [])
                                }
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
    
    return "", {}

def get_all_related_columns(column_prefix, df):
    """
    Find all columns that start with the given prefix in the DataFrame.
    
    Parameters:
    -----------
    column_prefix : str
        Prefix to search for (usually a QID)
    df : DataFrame
        DataFrame to search in
    
    Returns:
    --------
    list
        List of column names that match the prefix
    """
    related_columns = []
    # Using regex to match exact prefix to avoid matching similar QIDs
    pattern = rf'^{re.escape(column_prefix)}(_|$)'
    
    for col in df.columns:
        if re.match(pattern, col):
            related_columns.append(col)
    
    return related_columns

def compare_dataframes(df1, df2):
    """
    Compare two DataFrames and return a dictionary with comparison results.
    """
    comparison = {
        'columns_match': False,
        'column_count_match': False,
        'row_count_match': False,
        'missing_columns': [],
        'extra_columns': [],
        'column_differences': {},
        'missing_mismatch_analysis': {},  # Renamed to reflect focus on mismatches
        'numeric_type_mismatches': {}  # New field for numeric type mismatches
    }
    
    # Check if column counts match
    comparison['column_count_match'] = len(df1.columns) == len(df2.columns)
    
    # Check if row counts match
    comparison['row_count_match'] = len(df1) == len(df2)
    
    # Check if columns match
    df1_cols = set(df1.columns)
    df2_cols = set(df2.columns)
    
    comparison['columns_match'] = df1_cols == df2_cols
    
    # Find missing and extra columns
    comparison['missing_columns'] = list(df1_cols - df2_cols)
    comparison['extra_columns'] = list(df2_cols - df1_cols)
    
    # Check data types and missing value mismatches for common columns
    common_cols = df1_cols.intersection(df2_cols)
    for col in common_cols:
        # Check data types
        if df1[col].dtype != df2[col].dtype:
            comparison['column_differences'][col] = {
                'df1_type': str(df1[col].dtype),
                'df2_type': str(df2[col].dtype)
            }
            
        # Check for numeric type mismatches by comparing each non-null pair
        # Get non-null pairs of values
        non_null_mask = ~(df1[col].isna() | df2[col].isna())
        if non_null_mask.any():
            df1_values = df1.loc[non_null_mask, col]
            df2_values = df2.loc[non_null_mask, col]
            
            # Try converting each value to numeric
            df1_numeric = pd.to_numeric(df1_values, errors='coerce')
            df2_numeric = pd.to_numeric(df2_values, errors='coerce')
            
            # Find where one value is numeric but the other isn't
            numeric_mismatch_mask = (df1_numeric.notna() & df2_numeric.isna()) | (df1_numeric.isna() & df2_numeric.notna())
            
            if numeric_mismatch_mask.any():
                # Get examples of mismatches
                mismatch_examples = []
                mismatch_indices = numeric_mismatch_mask[numeric_mismatch_mask].index[:5]  # Get up to 5 examples
                
                for idx in mismatch_indices:
                    val1 = df1_values.loc[idx]
                    val2 = df2_values.loc[idx]
                    mismatch_examples.append({
                        'index': idx,
                        'df1_value': val1,
                        'df2_value': val2,
                        'df1_convertible': pd.to_numeric(pd.Series([val1]), errors='coerce').notna().iloc[0],
                        'df2_convertible': pd.to_numeric(pd.Series([val2]), errors='coerce').notna().iloc[0]
                    })
                
                comparison['numeric_type_mismatches'][col] = {
                    'mismatch_count': numeric_mismatch_mask.sum(),
                    'total_pairs': non_null_mask.sum(),
                    'mismatch_percentage': (numeric_mismatch_mask.sum() / non_null_mask.sum()) * 100,
                    'examples': mismatch_examples
                }
        
        # Find mismatches in missing values
        missing_df1 = df1[col].isna()
        missing_df2 = df2[col].isna()
        
        # Find cases where values mismatch (one has value, other is missing)
        mismatches = (missing_df1 & ~missing_df2) | (~missing_df1 & missing_df2)
        mismatch_count = mismatches.sum()
        mismatch_percentage = (mismatch_count / len(df1)) * 100
        
        if mismatch_count > 0:
            # Get examples of mismatches (up to 10)
            mismatch_examples = []
            mismatch_indices = df1[mismatches].index.tolist()[:10]
            
            for idx in mismatch_indices:
                example = {
                    'TWIN_ID': df1.loc[idx, 'TWIN_ID'] if 'TWIN_ID' in df1.columns else None,
                    'WAVE': df1.loc[idx, 'WAVE'] if 'WAVE' in df1.columns else None,
                    'original_missing': bool(missing_df1[idx]),
                    'imputed_missing': bool(missing_df2[idx]),
                    'original_value': None if missing_df1[idx] else df1.loc[idx, col],
                    'imputed_value': None if missing_df2[idx] else df2.loc[idx, col]
                }
                mismatch_examples.append(example)
            
            comparison['missing_mismatch_analysis'][col] = {
                'mismatch_percentage': round(mismatch_percentage, 2),
                'mismatch_count': mismatch_count,
                'examples': mismatch_examples
            }
    
    return comparison

def predict_expected_columns(qid, q_type, q_info):
    """
    Predict the expected columns for a given question based on its type and structure.
    
    Parameters:
    -----------
    qid : str
        Question ID
    q_type : str
        Question type (MC, Matrix, TE, etc.)
    q_info : dict
        Question information containing settings, rows, columns, etc.
    
    Returns:
    --------
    list
        List of expected column names
    """
    expected_columns = []
    
    if q_type == 'MC':
        selector = q_info['settings'].get('Selector', '')
        
        if selector in ['SAVR', 'SAHR']:
            # Single answer vertical/horizontal - usually just one column
            expected_columns.append(qid)
        else:
            # Multiple selection - a column for each choice
            choices = q_info['choices_id'] if q_info['choices_id'] else [str(i+1) for i in range(len(q_info['choices']))]
            for choice in choices:
                expected_columns.append(f"{qid}_{choice}")
    
    elif q_type == 'Matrix':
        # For matrix questions, check the structure
        sub_selector = q_info['settings'].get('SubSelector', '')
        selector = q_info['settings'].get('Selector', '')
        
        # Determine which ID list to use for rows
        row_ids = []
        if q_info['statements_id']:
            row_ids = q_info['statements_id']
        elif q_info['rows_id']:
            row_ids = q_info['rows_id']
        else:
            # If no explicit IDs, use numeric indices based on available lists
            statements_count = len(q_info['statements']) if q_info['statements'] else 0
            rows_count = len(q_info['rows']) if q_info['rows'] else 0
            count = max(statements_count, rows_count)
            if count > 0:
                row_ids = [str(i+1) for i in range(count)]
        
        if sub_selector == 'SingleAnswer':
            # For Likert or similar scales - one column per row
            for row_id in row_ids:
                expected_columns.append(f"{qid}_{row_id}")
        elif sub_selector == 'MultipleAnswer':
            # Multiple answer for each row - column for each row+column combination
            column_ids = q_info['columns_id'] if q_info['columns_id'] else [str(i+1) for i in range(len(q_info['columns']))]
            for row_id in row_ids:
                for col_id in column_ids:
                    expected_columns.append(f"{qid}_{row_id}_{col_id}")
    
    elif q_type == 'Slider':
        # Check if this is a multi-slider based on Statements or Rows
        has_statements = (q_info['statements'] and len(q_info['statements']) > 0) or (q_info['statements_id'] and len(q_info['statements_id']) > 0)
        has_rows = (q_info['rows'] and len(q_info['rows']) > 0) or (q_info['rows_id'] and len(q_info['rows_id']) > 0)
        
        if has_statements or has_rows:
            # Determine which ID list to use
            item_ids = []
            if q_info['statements_id']:
                item_ids = q_info['statements_id']
            elif q_info['rows_id']:
                item_ids = q_info['rows_id']
            else:
                # If no explicit IDs, use numeric indices based on available lists
                statements_count = len(q_info['statements']) if q_info['statements'] else 0
                rows_count = len(q_info['rows']) if q_info['rows'] else 0
                count = max(statements_count, rows_count)
                if count > 0:
                    item_ids = [str(i+1) for i in range(count)]
            
            # Create a column for each statement/row
            for item_id in item_ids:
                expected_columns.append(f"{qid}_{item_id}")
        else:
            # Single slider
            expected_columns.append(qid)
    
    elif q_type == 'TE':
        # Text entry
        selector = q_info['settings'].get('Selector', '')
        
        # Check if it has multiple statements or rows
        has_statements = (q_info['statements'] and len(q_info['statements']) > 0) or (q_info['statements_id'] and len(q_info['statements_id']) > 0)
        has_rows = (q_info['rows'] and len(q_info['rows']) > 0) or (q_info['rows_id'] and len(q_info['rows_id']) > 0)
        
        if selector in ['SL', 'ML']:
            if has_statements or has_rows:
                # Multiple text entries - determine which ID list to use
                item_ids = []
                if q_info['statements_id']:
                    item_ids = q_info['statements_id']
                elif q_info['rows_id']:
                    item_ids = q_info['rows_id']
                else:
                    # If no explicit IDs, use numeric indices
                    statements_count = len(q_info['statements']) if q_info['statements'] else 0
                    rows_count = len(q_info['rows']) if q_info['rows'] else 0
                    count = max(statements_count, rows_count)
                    if count > 0:
                        item_ids = [str(i+1) for i in range(count)]
                
                for item_id in item_ids:
                    expected_columns.append(f"{qid}_{item_id}")
            else:
                # Single text field
                expected_columns.append(f"{qid}_TEXT")
        elif selector == 'FORM':
            # Form with multiple entries
            row_ids = []
            if q_info['rows_id']:
                row_ids = q_info['rows_id']
            elif q_info['statements_id']:
                row_ids = q_info['statements_id']
            else:
                # If no explicit IDs, use numeric indices
                rows_count = len(q_info['rows']) if q_info['rows'] else 0
                statements_count = len(q_info['statements']) if q_info['statements'] else 0
                count = max(rows_count, statements_count)
                if count > 0:
                    row_ids = [str(i+1) for i in range(count)]
            
            for row_id in row_ids:
                expected_columns.append(f"{qid}_{row_id}")
    
    else:
        # For other question types, check if there are statements or rows
        has_statements = (q_info['statements'] and len(q_info['statements']) > 0) or (q_info['statements_id'] and len(q_info['statements_id']) > 0)
        has_rows = (q_info['rows'] and len(q_info['rows']) > 0) or (q_info['rows_id'] and len(q_info['rows_id']) > 0)
        
        if has_statements or has_rows:
            # Multiple entries - determine which ID list to use
            item_ids = []
            if q_info['statements_id']:
                item_ids = q_info['statements_id']
            elif q_info['rows_id']:
                item_ids = q_info['rows_id']
            else:
                # If no explicit IDs, use numeric indices
                statements_count = len(q_info['statements']) if q_info['statements'] else 0
                rows_count = len(q_info['rows']) if q_info['rows'] else 0
                count = max(statements_count, rows_count)
                if count > 0:
                    item_ids = [str(i+1) for i in range(count)]
            
            for item_id in item_ids:
                expected_columns.append(f"{qid}_{item_id}")
        else:
            # Single entry
            expected_columns.append(qid)
    
    # If we couldn't predict any columns, default to the base QID
    if not expected_columns:
        expected_columns.append(qid)
    
    return expected_columns

def process_files(config: dict):
    """
    Process all files in answer_blocks_llm_imputed and compare with corresponding files in answer_blocks.
    Always processes both numeric and text formats.
    """
    # Get configuration
    output_dir = config['pipeline']['output_dir']
    csv_dir = os.path.join(output_dir, 'csv_comparison')
    
    print("Processing both numeric and text formats for all datasets")
    
    # Create output directory for CSV files
    os.makedirs(csv_dir, exist_ok=True)
    
    # Find a sample answer block to analyze question types
    sample_file = None
    for root, _, files in os.walk(os.path.join(output_dir, 'answer_blocks')):
        for file in files:
            if file.endswith('A.json'):
                sample_file = os.path.join(root, file)
                break
        if sample_file:
            break
    
    # Process datasets with both formats
    def process_dataset(wave, source_dir, output_csv_base):
        """Helper function to process a dataset with both formats"""
        format_config = {
            'pipeline': {
                'output_dir': output_dir,
                'source_dir': source_dir
            },
            'mega_persona_extraction': {
                'wave': wave,
                'output_csv': output_csv_base
            }
        }
        
        # Process all formats
        result = process_mega_persona(format_config)
        
        if result is not None:
            # Copy to csv_comparison directory
            numeric_df = result['numeric']
            text_df = result['text']
            
            # Use original name for numeric files
            numeric_csv_path = os.path.join(csv_dir, output_csv_base)
            numeric_df.to_csv(numeric_csv_path, index=False)
            print(f"Copied numeric {wave} CSV to {numeric_csv_path}")
            
            # Use _label suffix for text files
            text_csv_path = os.path.join(csv_dir, output_csv_base.replace('.csv', '_label.csv'))
            text_df.to_csv(text_csv_path, index=False)
            print(f"Copied text {wave} CSV to {text_csv_path}")
            
            # For question type analysis, use the numeric version
            return numeric_df
        
        return None
    
    # Process all datasets
    df_wave1_3 = process_dataset('wave1_3', 'answer_blocks', 'responses_wave1_3.csv')
    print(f"Created wave1_3 CSVs with {len(df_wave1_3) if df_wave1_3 is not None else 0} rows")
    
    # After creating the DataFrame, identify question types and related columns
    question_types = {}
    if df_wave1_3 is not None and sample_file:
        # Find all unique question IDs (base form without suffixes)
        unique_qids = set()
        for col in df_wave1_3.columns:
            match = re.match(r'(QID\d+)', col)
            if match:
                unique_qids.add(match.group(1))
        
        # Get question type info for each base QID
        for qid in unique_qids:
            q_type, q_info = identify_question_type(sample_file, qid)
            if q_type:
                # Get all columns that actually exist in the data
                actual_columns = get_all_related_columns(qid, df_wave1_3)
                
                # Predict what columns we would expect based on question structure
                expected_columns = predict_expected_columns(qid, q_type, q_info)
                
                # Compare actual vs expected columns
                if set(actual_columns) != set(expected_columns) and expected_columns:
                    missing = set(expected_columns) - set(actual_columns)
                    extra = set(actual_columns) - set(expected_columns)
                    if missing:
                        print(f"Warning: Question {qid} ({q_type}) is missing predicted columns: {missing}")
                    if extra:
                        print(f"Note: Question {qid} ({q_type}) has additional columns not predicted: {extra}")
                
                # Store all information
                question_types[qid] = {
                    'type': q_type,
                    'info': q_info,
                    'actual_columns': actual_columns,
                    'expected_columns': expected_columns,
                    # Use a combination of actual and expected columns to be safe
                    'related_columns': list(set(actual_columns).union(set(expected_columns)))
                }
                print(f"Question {qid} is type {q_type} with {len(question_types[qid]['related_columns'])} related columns")
                
                # Detailed debug info for complex questions
                if q_type in ['Matrix'] and len(question_types[qid]['related_columns']) > 1:
                    print(f"  Structure: {q_info['settings'].get('Selector', '')} / {q_info['settings'].get('SubSelector', '')}")
                    if q_info['statements']:
                        print(f"  Statements: {q_info['statements'][:3]}... (total: {len(q_info['statements'])})")
                    if q_info['statements_id']:
                        print(f"  StatementsID: {q_info['statements_id'][:3]}... (total: {len(q_info['statements_id'])})")
                    if q_info['rows']:
                        print(f"  Rows: {q_info['rows'][:3]}... (total: {len(q_info['rows'])})")
                    if q_info['rows_id']:
                        print(f"  RowsID: {q_info['rows_id'][:3]}... (total: {len(q_info['rows_id'])})")
    
    # Process wave4 data
    df_wave4 = process_dataset('wave4', 'answer_blocks', 'responses_wave4.csv')
    print(f"Created wave4 CSVs with {len(df_wave4) if df_wave4 is not None else 0} rows")
    
    # Process imputed data
    df_imputed = process_dataset('wave1_3', 'answer_blocks_llm_imputed', 'responses_llm_imputed.csv')
    
    # If we have identified question types, ensure the imputed data has all related columns
    if df_imputed is not None and question_types:
        for qid, info in question_types.items():
            expected_columns = info['related_columns']
            actual_columns = get_all_related_columns(qid, df_imputed)
            
            missing_columns = set(expected_columns) - set(actual_columns)
            if missing_columns:
                print(f"Warning: Question {qid} ({info['type']}) is missing expected columns in imputed data: {missing_columns}")
                
                # For complex question types, ensure all columns exist
                if info['type'] in ['MC', 'Matrix', 'TE']:
                    for col in missing_columns:
                        # Add the column with NaN values
                        df_imputed[col] = None
                        print(f"Added missing column {col} to imputed data")
        
        # Rewrite the imputed CSV files with all necessary columns
        numeric_output_path = os.path.join(csv_dir, 'responses_llm_imputed_numeric.csv')
        df_imputed.to_csv(numeric_output_path, index=False)
        print(f"Saved updated numeric imputed CSV with all necessary columns to {numeric_output_path}")
    
    print(f"Created imputed CSVs with {len(df_imputed) if df_imputed is not None else 0} rows")

def print_comparison_results(comparison):
    """Print the results of a DataFrame comparison."""
    print(f"  Columns match: {comparison['columns_match']}")
    print(f"  Column count match: {comparison['column_count_match']}")
    print(f"  Row count match: {comparison['row_count_match']}")
    
    if comparison['missing_columns']:
        print(f"  Missing columns in imputed: {comparison['missing_columns']}")
    
    if comparison['extra_columns']:
        print(f"  Extra columns in imputed: {comparison['extra_columns']}")
    
    if comparison['column_differences']:
        print("  Column type differences:")
        for col, diff in comparison['column_differences'].items():
            print(f"    {col}: {diff['df1_type']} vs {diff['df2_type']}")
    
    # Print missing value mismatch analysis
    print("\n  Missing Value Mismatch Analysis:")
    for col, stats in comparison['missing_mismatch_analysis'].items():
        print(f"    {col}:")
        print(f"      {stats['mismatch_count']} mismatches found ({stats['mismatch_percentage']}% of data)")
        print("      Examples of mismatches:")
        for ex in stats['examples']:
            status = "Original has value but imputed is missing" if ex['original_value'] is not None else "Original is missing but imputed has value"
            print(f"        TWIN_ID: {ex['TWIN_ID']}, WAVE: {ex['WAVE']}")
            print(f"        Status: {status}")
            print(f"        Original value: {ex['original_value']}")
            print(f"        Imputed value: {ex['imputed_value']}")
            print(f"        {'=' * 40}")

def main():
    parser = argparse.ArgumentParser(description='Convert and compare response files')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Process files
    process_files(config)

if __name__ == "__main__":
    main() 