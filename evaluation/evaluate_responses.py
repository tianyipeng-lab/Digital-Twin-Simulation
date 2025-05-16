import argparse
import json
import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import textwrap
import warnings
from scipy import stats

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_invalid_reason(series1: pd.Series, series2: pd.Series) -> Tuple[bool, str]:
    """Determine why correlation cannot be computed between two series."""
    # Check if either series is empty
    if len(series1) == 0 or len(series2) == 0:
        return False, "No data available"
    
    # Try converting to numeric, counting failures
    numeric1 = pd.to_numeric(series1, errors='coerce')
    numeric2 = pd.to_numeric(series2, errors='coerce')
    
    # Check if conversion failed completely
    if numeric1.isna().all() or numeric2.isna().all():
        return False, "Non-numeric data"
    
    # Get valid pairs (non-NaN in both series)
    valid_mask = ~(numeric1.isna() | numeric2.isna())
    valid1 = numeric1[valid_mask]
    valid2 = numeric2[valid_mask]
    
    if len(valid1) == 0:
        return False, "No valid numeric pairs"
    
    # Check if either series is constant
    if len(valid1.unique()) == 1 or len(valid2.unique()) == 1:
        return True, "Constant values"
    
    # Check if either series has zero standard deviation
    if np.std(valid1) == 0 or np.std(valid2) == 0:
        return True, "Zero standard deviation"
    
    return False, "Unknown error"

def _is_series_binary_like(series: pd.Series) -> bool:
    """Check if a series looks like binary data (e.g., two unique integer values after NA drop)."""
    if series.empty:
        return False
    numeric_series = pd.to_numeric(series, errors='coerce')
    # Drop NA for uniqueness and type checking
    valid_numeric = numeric_series.dropna()
    if valid_numeric.empty: # All NA after conversion or originally empty
        return False
    unique_vals = valid_numeric.unique()
    # Check for 2 unique values that are effectively integers
    # (e.g. 1.0 and 2.0 are common for survey data)
    if len(unique_vals) == 2 and all(v == np.floor(v) for v in unique_vals if not pd.isna(v)): # Check if values are whole numbers
        return True
    return False

def get_definitive_column_type(column_name: str,
                               groundtruth_df: pd.DataFrame,
                               wave4_df: Optional[pd.DataFrame]) -> str:
    """Determines the definitive type of a column based on ground truth and wave4 data."""
    gt_col_exists = column_name in groundtruth_df.columns
    w4_col_exists = wave4_df is not None and column_name in wave4_df.columns

    # Check ground truth first for binary type
    if gt_col_exists:
        if _is_series_binary_like(groundtruth_df[column_name]):
            return 'binary'

    # If not binary in ground truth, check wave4
    if w4_col_exists:
        if _is_series_binary_like(wave4_df[column_name]): # type: ignore
            return 'binary'

    # If not binary in either, check for numeric data presence
    if gt_col_exists:
        numeric_gt = pd.to_numeric(groundtruth_df[column_name], errors='coerce')
        if not numeric_gt.isna().all(): # If any valid numeric data exists
            return 'numeric'
    
    if w4_col_exists:
        numeric_w4 = pd.to_numeric(wave4_df[column_name], errors='coerce') # type: ignore
        if not numeric_w4.isna().all(): # If any valid numeric data exists
            return 'numeric'
            
    return 'unknown' # Default if no clear type or no data

def safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Safely compute correlation coefficient, returning 0 for any invalid cases."""
    try:
        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        # Return 0 if not enough data or constant values
        if len(x_clean) < 2 or np.std(x_clean) == 0 or np.std(y_clean) == 0:
            return 0.0
            
        return np.corrcoef(x_clean, y_clean)[0,1]
    except:
        return 0.0

def calculate_metrics(gt_series: pd.Series, pred_series: pd.Series, forced_type: Optional[str] = None) -> Dict[str, Any]:
    """Calculate metrics (accuracy, correlation, R², beta) for a pair of series."""
    metrics = {
        'type': 'unknown', # Will be overwritten by forced_type or deduced type
        'samples': 0,
        'accuracy': np.nan,
        'correlation': np.nan,
        'r2': np.nan,
        'beta': np.nan,
        'gt_non_na_count': 0, # Added
        'pred_non_na_count': 0, # Added
        'note': None
    }

    try:
        # Store raw non-NA counts
        metrics['gt_non_na_count'] = int(gt_series.notna().sum())
        metrics['pred_non_na_count'] = int(pred_series.notna().sum())

        # Convert to numeric, coercing errors - needed for all paths
        numeric_gt = pd.to_numeric(gt_series, errors='coerce')
        numeric_pred = pd.to_numeric(pred_series, errors='coerce')

        # Get valid pairs (non-NaN in both series)
        valid_mask = ~(numeric_gt.isna() | numeric_pred.isna())
        valid_gt = numeric_gt[valid_mask]
        valid_pred = numeric_pred[valid_mask]
        
        metrics['samples'] = len(valid_gt)

        if metrics['samples'] == 0:
            metrics['note'] = "No valid numeric pairs"
            if forced_type and forced_type in ['binary', 'numeric']:
                 metrics['type'] = forced_type # Reflect intended type even if no samples
            else:
                 metrics['type'] = 'unknown'
            return metrics

        type_to_process = metrics['type'] # Start with default or previous value

        if forced_type and forced_type in ['binary', 'numeric']:
            type_to_process = forced_type
        else: # Fallback to dynamic type detection if forced_type is not helpful
            unique_vals_combined = pd.concat([valid_gt, valid_pred]).unique()
            # Check if data is binary (only 1s and 2s or potentially other two unique values)
            is_binary_candidate = (len(unique_vals_combined) == 2 and 
                                   all(v == np.floor(v) for v in unique_vals_combined if not pd.isna(v))) # Use np.floor for float check

            if is_binary_candidate:
                type_to_process = 'binary'
            elif metrics['samples'] >= 2: # Potentially numeric
                 type_to_process = 'numeric'
            # else, type_to_process remains 'unknown' or what it was from forced_type if invalid

        metrics['type'] = type_to_process # Final determined type for this calculation

        if type_to_process == 'binary':
            metrics['accuracy'] = (valid_gt == valid_pred).mean()
            # Correlation/R2/Beta are less meaningful for binary, remain NaN.
        
        elif type_to_process == 'numeric':
            if metrics['samples'] >= 2:
                 # Check for constant values or zero standard deviation before calculating numeric metrics
                if len(valid_gt.unique()) == 1 or len(valid_pred.unique()) == 1:
                     metrics['note'] = "Constant values"
                     # Metrics will remain NaN or become 0.0 later
                elif np.std(valid_gt) == 0 or np.std(valid_pred) == 0:
                     metrics['note'] = "Zero standard deviation"
                     # Metrics will remain NaN or become 0.0 later
                else:
                    # If not binary and enough samples, calculate numeric metrics
                    try:
                        # Use safe_correlation
                        metrics['correlation'] = safe_correlation(valid_gt.values, valid_pred.values)
                        if not pd.isna(metrics['correlation']) and metrics['correlation'] != 0.0:
                            metrics['r2'] = metrics['correlation']**2
                            # Calculate beta (slope)
                            beta, _ = np.polyfit(valid_gt.values, valid_pred.values, 1)
                            metrics['beta'] = beta
                        else: # if safe_correlation returned 0 or NaN (due to its internal checks)
                            metrics['r2'] = 0.0 if not pd.isna(metrics['correlation']) else np.nan 
                            metrics['beta'] = np.nan 
                    except np.linalg.LinAlgError:
                         metrics['note'] = "LinAlgError during polyfit"
                    except Exception as e:
                         metrics['note'] = f"Error in numeric calc: {e}"
            else: # samples < 2
                metrics['note'] = "Not enough samples for numeric metrics"
        else: # type_to_process is 'unknown' or some other non-handled type
            if not metrics['note']: # Don't overwrite existing notes like "No valid numeric pairs"
                metrics['note'] = "Type unknown or insufficient data for classification"
            
    except Exception as e:
        metrics['note'] = f"Error during metric calculation: {e}" # Broader catch for initial conversion issues

    # Clean up NaN values to None or 0.0 where appropriate for JSON serialization/downstream use
    # Current script keeps NaNs, which is fine.
    for key in ['accuracy', 'correlation', 'r2', 'beta']:
        if pd.isna(metrics[key]):
             pass 
             
    return metrics

def evaluate_response_accuracy(generated_df: pd.DataFrame, 
                               groundtruth_df: pd.DataFrame, 
                               name: str = "",
                               column_types_map: Optional[Dict[str, str]] = None
                               ) -> Dict[str, Any]:
    """Evaluate accuracy of generated responses against ground truth."""
    results = {
        "name": name,
        "column_metrics": {}, # Store per-column results here
        "missing_columns": [],
        "overall_stats": {
            "total_columns": 0,
            "binary_columns": 0,
            "numeric_columns": 0,
            "other_columns": 0, # Columns that weren't binary/numeric or had errors
            "missing_columns": 0,
            "average_binary_accuracy": np.nan,
            "average_correlation": np.nan,
            "average_r2": np.nan,
            "average_beta": np.nan,
            "average_non_na_ratio_vs_gt": np.nan # Added
        }
    }
    
    # Verify TWIN_ID exists in both dataframes
    if 'TWIN_ID' not in generated_df.columns or 'TWIN_ID' not in groundtruth_df.columns:
        raise ValueError("TWIN_ID column missing from one or both dataframes")
        
    # Merge dataframes on TWIN_ID to ensure proper alignment
    merged_df = pd.merge(groundtruth_df, generated_df, on='TWIN_ID', suffixes=('_gt', '_gen'), how='inner')
    
    if merged_df.empty:
        print(f"Warning: No common TWIN_IDs found between ground truth and generated data for '{name}'.")
        # Set stats to indicate no data processed
        results["overall_stats"].update({
            "total_columns": len([col for col in groundtruth_df.columns if col.startswith('QID')]),
            "missing_columns": len([col for col in groundtruth_df.columns if col.startswith('QID')]),
        })
        return results

    # Get question columns (starting with QID) present in ground truth
    question_cols = [col for col in groundtruth_df.columns if col.startswith('QID')]
    results["overall_stats"]["total_columns"] = len(question_cols)
    
    # Store metrics from calculation
    all_metrics = []
    all_current_model_non_na_ratios_vs_gt = [] # Added
    
    for col in question_cols:
        gt_col_name = f"{col}_gt"
        gen_col_name = f"{col}_gen"
        
        if gen_col_name not in merged_df.columns:
            results["missing_columns"].append(col)
            continue
            
        # Get values from merged dataframe
        gt_series_for_calc = merged_df[gt_col_name]
        gen_series_for_calc = merged_df[gen_col_name]
        
        # Determine forced type for this column
        definitive_type_for_col = None
        if column_types_map:
            definitive_type_for_col = column_types_map.get(col)

        if name == "ML":
            # Convert GT to numeric to check its nature
            gt_numeric_for_check = pd.to_numeric(gt_series_for_calc, errors='coerce')
            gt_numeric_clean = gt_numeric_for_check.dropna()

            # Check if all valid ground truth values are effectively integers
            if not gt_numeric_clean.empty and (gt_numeric_clean == np.floor(gt_numeric_clean)).all():
                # If so, convert ML predictions to numeric and round them
                ml_predictions_numeric = pd.to_numeric(gen_series_for_calc, errors='coerce')
                # .round() will produce floats (e.g., 1.0, 2.0), which is fine for calculate_metrics
                gen_series_for_calc = ml_predictions_numeric.round()
        
        # Calculate all metrics for this column
        col_metrics = calculate_metrics(gt_series_for_calc, gen_series_for_calc, forced_type=definitive_type_for_col)
        results["column_metrics"][col] = col_metrics
        all_metrics.append(col_metrics) # Collect for overall stats

        # Collect non-NA ratio for overall stats for the current model
        gt_non_na_count_for_col = col_metrics.get('gt_non_na_count', 0)
        pred_non_na_count_for_col = col_metrics.get('pred_non_na_count', 0)
        if gt_non_na_count_for_col > 0:
            non_na_ratio = pred_non_na_count_for_col / gt_non_na_count_for_col
            all_current_model_non_na_ratios_vs_gt.append(non_na_ratio)
            
    # Compute overall statistics from collected metrics
    binary_metrics = [m for m in all_metrics if m['type'] == 'binary' and not pd.isna(m['accuracy'])]
    numeric_metrics = [m for m in all_metrics if m['type'] == 'numeric']
    
    # Filter numeric metrics further for valid correlation, r2, beta
    valid_numeric_corr = [m['correlation'] for m in numeric_metrics if not pd.isna(m['correlation'])]
    valid_numeric_r2 = [m['r2'] for m in numeric_metrics if not pd.isna(m['r2'])]
    valid_numeric_beta = [m['beta'] for m in numeric_metrics if not pd.isna(m['beta'])]

    results["overall_stats"].update({
        "binary_columns": len(binary_metrics),
        "numeric_columns": len(numeric_metrics),
        "missing_columns": len(results["missing_columns"]),
        "other_columns": len(all_metrics) - len(binary_metrics) - len(numeric_metrics),
        "average_binary_accuracy": np.mean([m['accuracy'] for m in binary_metrics]) if binary_metrics else np.nan,
        "average_correlation": np.mean(valid_numeric_corr) if valid_numeric_corr else np.nan,
        "average_r2": np.mean(valid_numeric_r2) if valid_numeric_r2 else np.nan,
        "average_beta": np.mean(valid_numeric_beta) if valid_numeric_beta else np.nan
    })
    
    if all_current_model_non_na_ratios_vs_gt: # Added block
        results["overall_stats"]["average_non_na_ratio_vs_gt"] = np.mean(all_current_model_non_na_ratios_vs_gt)
    else:
        results["overall_stats"]["average_non_na_ratio_vs_gt"] = np.nan

    # Add notes for columns where metrics calculation failed or had issues
    results["notes"] = {col: m['note'] for col, m in results["column_metrics"].items() if m.get('note')}

    return results

def load_config(config_file: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_evaluation_results(results_list: List[Dict[str, Any]], output_dir: str, mapping_path: str) -> None:
    """Save evaluation results as CSV files with question mapping integration."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load question mapping if available
    question_map = None
    if os.path.exists(mapping_path):
        try:
            question_map = pd.read_csv(mapping_path)
            # Ensure we have the expected QuestionID column
            if 'QuestionID' not in question_map.columns:
                if 'column_id' in question_map.columns:
                    question_map['QuestionID'] = question_map['column_id']
                elif 'column' in question_map.columns:
                    question_map['QuestionID'] = question_map['column']
                else:
                    print(f"Warning: Mapping file does not have required column (QuestionID, column_id, or column)")
                    print(f"Available columns: {list(question_map.columns)}")
        except Exception as e:
            print(f"Warning: Could not load question mapping: {e}")
    
    # Prepare data for numeric correlations CSV
    numeric_data = []
    all_numeric_cols = set()
    for r in results_list:
        all_numeric_cols.update(col for col, m in r.get('column_metrics', {}).items() if m.get('type') == 'numeric')
        
    for col in sorted(list(all_numeric_cols)):
        row = {'QuestionID': col}
        for result in results_list:
            metrics = result.get('column_metrics', {}).get(col)
            if metrics and metrics.get('type') == 'numeric':
                row[f"{result['name']}_correlation"] = metrics.get('correlation', np.nan)
                row[f"{result['name']}_r2"] = metrics.get('r2', np.nan)
                row[f"{result['name']}_beta"] = metrics.get('beta', np.nan)
                row[f"{result['name']}_samples"] = metrics.get('samples', 0)
                # Add Non-NA ratio for numeric
                gt_nona = metrics.get('gt_non_na_count')
                pred_nona = metrics.get('pred_non_na_count')
                if gt_nona is not None and pred_nona is not None and gt_nona > 0:
                    row[f"{result['name']}_non_na_ratio_vs_gt"] = pred_nona / gt_nona
                else:
                    row[f"{result['name']}_non_na_ratio_vs_gt"] = np.nan
            else:
                 # Fill with NaN if column not present or not numeric for this result
                 row[f"{result['name']}_correlation"] = np.nan
                 row[f"{result['name']}_r2"] = np.nan
                 row[f"{result['name']}_beta"] = np.nan
                 row[f"{result['name']}_samples"] = 0
                 row[f"{result['name']}_non_na_ratio_vs_gt"] = np.nan # Added
        numeric_data.append(row)
    
    # Save numeric correlations CSV with mapping
    if numeric_data:
        numeric_df = pd.DataFrame(numeric_data)
        if question_map is not None:
            # Ensure merge key exists and handle potential missing columns
            map_cols = ['QuestionID', 'description']
            available_map_cols = [c for c in map_cols if c in question_map.columns]
            if 'QuestionID' in available_map_cols:
                 numeric_df = pd.merge(numeric_df, question_map[available_map_cols], 
                                     on='QuestionID', how='left')
            else:
                print("Warning: 'QuestionID' not in mapping columns for numeric CSV merge.")

        numeric_csv = os.path.join(output_dir, 'numeric_metrics.csv') # Renamed file
        numeric_df.to_csv(numeric_csv, index=False, float_format='%.4f') # Format float precision
        print(f"Saved numeric metrics to {numeric_csv}")
    
    # Prepare data for binary accuracy CSV
    binary_data = []
    all_binary_cols = set()
    for r in results_list:
        all_binary_cols.update(col for col, m in r.get('column_metrics', {}).items() if m.get('type') == 'binary')

    for col in sorted(list(all_binary_cols)):
        row = {'QuestionID': col}
        for result in results_list:
            metrics = result.get('column_metrics', {}).get(col)
            if metrics and metrics.get('type') == 'binary':
                row[f"{result['name']}_accuracy"] = metrics.get('accuracy', np.nan)
                row[f"{result['name']}_samples"] = metrics.get('samples', 0)
                # Add Non-NA ratio for binary
                gt_nona = metrics.get('gt_non_na_count')
                pred_nona = metrics.get('pred_non_na_count')
                if gt_nona is not None and pred_nona is not None and gt_nona > 0:
                    row[f"{result['name']}_non_na_ratio_vs_gt"] = pred_nona / gt_nona
                else:
                    row[f"{result['name']}_non_na_ratio_vs_gt"] = np.nan
            else:
                # Fill with NaN if column not present or not binary for this result
                row[f"{result['name']}_accuracy"] = np.nan
                row[f"{result['name']}_samples"] = 0
                row[f"{result['name']}_non_na_ratio_vs_gt"] = np.nan # Added
        binary_data.append(row)
    
    # Save binary accuracy CSV with mapping
    if binary_data:
        binary_df = pd.DataFrame(binary_data)
        if question_map is not None:
            map_cols = ['QuestionID', 'description']
            available_map_cols = [c for c in map_cols if c in question_map.columns]
            if 'QuestionID' in available_map_cols:
                 binary_df = pd.merge(binary_df, question_map[available_map_cols], 
                                   on='QuestionID', how='left')
            else:
                 print("Warning: 'QuestionID' not in mapping columns for binary CSV merge.")
            
        # Organize columns in a logical order
        column_order = ['QuestionID']
        if 'description' in binary_df.columns:
             column_order.append('description')
             
        for result in results_list:
            name = result['name']
            column_order.extend([
                f"{name}_accuracy",
                f"{name}_samples"
            ])
            if f"{name}_non_na_ratio_vs_gt" in binary_df.columns: # Check if column exists
                column_order.append(f"{name}_non_na_ratio_vs_gt")
        
        # Reorder columns, keeping any extra columns at the end
        existing_cols = [col for col in column_order if col in binary_df.columns]
        other_cols = [col for col in binary_df.columns if col not in existing_cols]
        binary_df = binary_df[existing_cols + other_cols]
        
        binary_csv = os.path.join(output_dir, 'binary_accuracy.csv')
        binary_df.to_csv(binary_csv, index=False, float_format='%.4f') # Format float precision
        print(f"Saved binary accuracy to {binary_csv}")
    
    # Prepare summary statistics
    summary_data = []
    for result in results_list:
        stats = result.get('overall_stats', {})
        row = {
            'Model': result.get('name', 'Unknown'),
            'Total Questions': stats.get('total_columns', 0),
            'Binary Questions': stats.get('binary_columns', 0),
            'Numeric Questions': stats.get('numeric_columns', 0),
            'Other Questions': stats.get('other_columns', 0),
            'Missing Questions': stats.get('missing_columns', 0),
            'Average Binary Accuracy': stats.get('average_binary_accuracy', np.nan),
            'Average Correlation': stats.get('average_correlation', np.nan),
            'Average R²': stats.get('average_r2', np.nan), # Added R²
            'Average Beta': stats.get('average_beta', np.nan),   # Added Beta
            'Average Non-NA Ratio vs GT': stats.get('average_non_na_ratio_vs_gt', np.nan) # Added
        }
        summary_data.append(row)
    
    # Save summary statistics CSV
    summary_df = pd.DataFrame(summary_data)
    summary_csv = os.path.join(output_dir, 'performance_summary.csv')
    # Define column order for summary
    summary_cols_order = ['Model', 'Total Questions', 'Binary Questions', 'Numeric Questions', 
                          'Other Questions', 'Missing Questions', 'Average Binary Accuracy', 
                          'Average Correlation', 'Average R²', 'Average Beta',
                          'Average Non-NA Ratio vs GT'] # Added
    summary_df = summary_df[summary_cols_order] # Reorder
    summary_df.to_csv(summary_csv, index=False, float_format='%.4f') # Format float precision
    print(f"Saved performance summary to {summary_csv}")

def _collect_questions_from_block(block: dict) -> List[dict]:
    """Recursively collect question dictionaries from a Qualtrics style JSON block."""
    questions = []
    if not isinstance(block, dict):
        return questions

    if 'Questions' in block:
        questions.extend(block['Questions'])
    elif 'Elements' in block:
        for element in block['Elements']:
            questions.extend(_collect_questions_from_block(element))
    else:
        # Sometimes the block itself is a question
        if 'QuestionID' in block:
            questions.append(block)
    return questions


def _get_columns_for_question(question: Dict[str, Any]) -> List[str]:
    """Determine the dataframe column names that correspond to a Qualtrics question.

    This mirrors the logic used in `extract_mega_persona_csv.py` so that we can
    match DataFrame columns back to survey questions.
    """
    qid = question.get('QuestionID')
    qtype = question.get('QuestionType', '')
    settings = question.get('Settings', {})
    columns: List[str] = []

    if not qid:
        return columns

    if qtype == 'Matrix':
        # One column per row for single-answer matrices
        rows_id = question.get('RowsID', [])
        rows = question.get('Rows', [])
        # Use RowsID if available otherwise 1-based index
        for idx, _ in enumerate(rows):
            row_identifier = str(idx + 1)
            if idx < len(rows_id):
                row_identifier = str(rows_id[idx])
            columns.append(f"{qid}_{row_identifier}")

    elif qtype == 'MC':
        selector = settings.get('Selector', '')
        options = question.get('Options', [])
        if selector in ['SAVR', 'SAHR']:
            # Single answer → single column
            columns.append(qid)
        else:
            # Multi-answer → one column per option
            options_id = question.get('OptionsID', [])
            for idx, _ in enumerate(options):
                opt_identifier = str(idx + 1)
                if idx < len(options_id):
                    opt_identifier = str(options_id[idx])
                columns.append(f"{qid}_{opt_identifier}")

    elif qtype in ['Slider', 'TE']:
        # Sliders map directly; for TE numeric responses we suffixed _TEXT when extracted
        if qtype == 'TE' and settings.get('Selector') in ['SL', 'ML']:
            columns.append(f"{qid}_TEXT")
        else:
            columns.append(qid)
    else:
        columns.append(qid)

    return columns


def _create_model_vs_model_performance_scatter_page(pdf: PdfPages, results_list: List[Dict[str, Any]], plot_type: str):
    """Create a PDF page with scatter plots comparing model performances for a specific metric type."""
    llm_results = next((r for r in results_list if r['name'] == 'LLM'), None)
    wave4_results = next((r for r in results_list if r['name'] == 'Wave4'), None)
    ml_results = next((r for r in results_list if r['name'] == 'ML'), None)

    if not llm_results or not wave4_results:
        print(f"Warning: LLM or Wave4 results not found. Skipping model performance scatter plot page for {plot_type}.")
        return

    # --- Data Preparation ---
    points1 = []  # LLM vs Wave4
    points2 = []  # LLM vs ML
    metric_key = 'accuracy' if plot_type == 'binary' else 'correlation'
    x_label_base = f"LLM {metric_key.capitalize()}"
    y_label_w4 = f"Wave4 {metric_key.capitalize()}"
    y_label_ml = f"ML {metric_key.capitalize()}"
    page_title = f"Model Performance Comparison: {plot_type.capitalize()} {metric_key.capitalize()}"
    is_accuracy_plot = (plot_type == 'binary')

    if llm_results.get('column_metrics'):
        for qid, llm_metric_data in llm_results['column_metrics'].items():
            if llm_metric_data.get('type') != plot_type: # Only consider relevant type
                continue

            w4_metric_data = wave4_results.get('column_metrics', {}).get(qid)
            ml_metric_data = ml_results.get('column_metrics', {}).get(qid) if ml_results else None

            llm_val = llm_metric_data.get(metric_key)

            # LLM vs Wave4
            if w4_metric_data and w4_metric_data.get('type') == plot_type:
                w4_val = w4_metric_data.get(metric_key)
                if pd.notna(llm_val) and pd.notna(w4_val):
                    points1.append((llm_val, w4_val))
            
            # LLM vs ML
            if ml_metric_data and ml_metric_data.get('type') == plot_type:
                ml_val = ml_metric_data.get(metric_key)
                if pd.notna(llm_val) and pd.notna(ml_val):
                    points2.append((llm_val, ml_val))

    # --- Plotting ---
    num_plots = 1
    if ml_results and points2: # Only add second plot if ML results exist AND there's data for it
        num_plots = 2

    if not points1 and not points2: # No data for any plot for this type
        print(f"Warning: No sufficient data for {plot_type} model performance scatter plots. Skipping page.")
        return
    
    # If points1 is empty but points2 is not, we might still want to plot points2 if num_plots was set to 2.
    # However, the primary plot is LLM vs W4. If that's empty, maybe we skip.
    # For now, if points1 is empty, we will still proceed if points2 has data and num_plots is 2.
    # This means an empty first plot might be shown if only LLMvsML has data.
    # A better check: at least one set of points must have data.
    if not points1 and (num_plots == 1 or not points2):
        print(f"Warning: No data for primary (LLM vs Wave4) {plot_type} scatter plot and no ML data to show. Skipping page.")
        return

    fig, axs = plt.subplots(1, num_plots, figsize=(7 * num_plots, 6), constrained_layout=True)
    fig.suptitle(page_title, fontsize=16, y=1.05)

    if num_plots == 1:
        current_axs = [axs] # Make it iterable for consistency if only one plot
    else:
        current_axs = axs

    # Helper for plotting (can be kept outside or nested if preferred, but external is fine)
    # plot_scatter is defined later in the context or assumed to be available.
    # For this edit, I will redefine a local version for clarity of what it needs.
    def _local_plot_scatter(ax, plot_points, x_lbl, y_lbl, plot_title, is_accuracy_metric):
        if not plot_points:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(plot_title)
            ax.set_xlabel(x_lbl)
            ax.set_ylabel(y_lbl)
            limit_min, limit_max = (0, 1) if is_accuracy_metric else (-1, 1)
            ax.set_xlim(limit_min, limit_max)
            ax.set_ylim(limit_min, limit_max)
            return

        x_vals = [p[0] for p in plot_points]
        y_vals = [p[1] for p in plot_points]
        
        sns.regplot(x=x_vals, y=y_vals, ax=ax, scatter_kws={'s': 20, 'alpha': 0.6}, line_kws={'color':'red'})
        
        limit_min, limit_max = (0, 1) if is_accuracy_metric else (-1, 1)
        ax.plot([limit_min, limit_max], [limit_min, limit_max], 'k--', alpha=0.5, lw=1) # 1:1 line
        ax.set_xlim(limit_min, limit_max)
        ax.set_ylim(limit_min, limit_max)
        ax.set_xlabel(x_lbl)
        ax.set_ylabel(y_lbl)
        ax.set_title(plot_title)

        if len(x_vals) >= 2:
            try:
                r_val, _ = stats.pearsonr(x_vals, y_vals)
                ax.text(0.05, 0.95, f"R = {r_val:.2f}", transform=ax.transAxes, 
                        va='top', ha='left', fontsize=9, 
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
            except ValueError:
                ax.text(0.05, 0.95, "R: N/A", transform=ax.transAxes, 
                        va='top', ha='left', fontsize=9, 
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    # Plot 1: LLM vs Wave4
    _local_plot_scatter(current_axs[0], points1, x_label_base, y_label_w4, f"LLM vs Wave4 ({plot_type.capitalize()})", is_accuracy_plot)

    if num_plots == 2:
        # Plot 2: LLM vs ML
        _local_plot_scatter(current_axs[1], points2, x_label_base, y_label_ml, f"LLM vs ML ({plot_type.capitalize()})", is_accuracy_plot)
    
    pdf.savefig(fig)
    plt.close(fig)


def generate_pdf_report(template_json_path: str,
                         llm_df: pd.DataFrame,
                         wave4_df: pd.DataFrame,
                         wave1_3_df: pd.DataFrame,
                         mapping_df: Optional[pd.DataFrame],
                         output_pdf_path: str,
                         config: dict,
                         results_list: List[Dict[str, Any]]) -> None:
    """Generate a PDF report with per-question distribution and regression plots."""
    
    def conf_matrix(x: np.ndarray, y: np.ndarray, num_classes: int) -> np.ndarray:
        """Generate confusion matrix for discrete values."""
        mat = np.zeros((num_classes, num_classes), dtype=int)
        for a, b in zip(x.astype(int), y.astype(int)):
            mat[a - 1, b - 1] += 1
        return mat
    
    with PdfPages(output_pdf_path) as pdf:
        # Create cover page
        fig = plt.figure(figsize=(14, 10))
        plt.axis('off')
        
        # Title
        plt.text(0.5, 0.95, "Digital Twin Simulation Report", 
                ha='center', va='top', fontsize=20, fontweight='bold')
        
        # Configuration details
        config_text = [
            "Configuration Details:",
            f"PID Range: {config['pipeline']['pid_range']['start']} - {config['pipeline']['pid_range']['end']}",
            f"Output Directory: {config['pipeline']['output_dir']}",
            "\nLLM Simulation:",
            f"Default Model: {config['llm_simulation']['model']['default']}",
            f"Fallback Model: {config['llm_simulation']['model']['fallback']}",
            f"Temperature: {config['llm_simulation']['model']['temperature']}",
            "\nPrompt:",
            f"{config['llm_simulation']['prompts']['system']}"
        ]
        
        plt.text(0.1, 0.8, '\n'.join(config_text), 
                ha='left', va='top', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=1'))
        
        pdf.savefig(fig)
        plt.close()
        
        # Create summary page using pre-calculated metrics from results_list
        fig = plt.figure(figsize=(14, 10))
        plt.axis('off')
        
        plt.text(0.5, 0.95, "Performance Summary", 
                ha='center', va='top', fontsize=16, fontweight='bold')
        
        # Dynamically build column headers and extract model stats
        model_names_in_results = [res['name'] for res in results_list] # e.g., ["LLM", "Wave4", "ML"]
        
        # Base columns
        col_headers = ["Metric", "Number of Questions"]
        # Add performance columns for each model found
        for model_name in model_names_in_results:
            col_headers.append(f"{model_name} Performance")
        col_headers.append("Note")

        summary_table_data = []
        
        # Define metrics and their corresponding keys in overall_stats
        metric_definitions = [
            ("Binary Accuracy", "average_binary_accuracy", "binary_columns", "Higher is better (range: 0-1)"),
            ("Correlation (ρ)", "average_correlation", "numeric_columns", "Higher is better (range: -1 to 1)"),
            ("R-squared (R²)", "average_r2", "numeric_columns", "Higher is better (range: 0-1)"),
            ("Beta (β)", "average_beta", "numeric_columns", "Closer to 1 is better (slope alignment)"),
            ("Avg Non-NA % vs GT", "average_non_na_ratio_vs_gt", "total_columns", "Avg. (Pred Non-NA / GT Non-NA) * 100%") # Added
        ]

        # Get stats for all models, store in a dict by name for easy access
        all_model_stats = {res['name']: res.get('overall_stats', {}) for res in results_list}
        
        # Use the first model's stats for 'Number of Questions' as representative
        representative_model_name = model_names_in_results[0] if model_names_in_results else None
        representative_stats = all_model_stats.get(representative_model_name, {})

        for display_name, metric_key, count_key, note_text in metric_definitions:
            row = {"Metric": display_name}
            
            num_questions = representative_stats.get(count_key, 'N/A')
            row["Number of Questions"] = num_questions
            
            for model_name in model_names_in_results:
                model_stats = all_model_stats.get(model_name, {})
                val = model_stats.get(metric_key, np.nan)
                row[f"{model_name} Performance"] = f"{val:.3f}" if not pd.isna(val) else 'N/A'
            
            row["Note"] = note_text
            summary_table_data.append(row)

        summary_df = pd.DataFrame(summary_table_data, columns=col_headers)
        
        # Create table
        table = plt.table(cellText=summary_df.values,
                         colLabels=summary_df.columns,
                         cellLoc='left',
                         loc='center',
                         bbox=[0.05, 0.3, 0.9, 0.5]) # Adjusted bbox for potentially more columns
        
        table.auto_set_font_size(False)
        table.set_fontsize(8) 
        table.scale(1.0, 1.8)

        # Dynamic column width strategy
        final_col_widths = []
        num_perf_cols = len(model_names_in_results)

        if summary_df.empty: # Handle empty summary_df case
            pass # No table to format
        elif len(summary_df.columns) > 0:
            final_col_widths.append(0.20)  # Metric
            if len(summary_df.columns) > 1:
                final_col_widths.append(0.15)  # Num Qs
            
            note_width_target = 0.25
            
            # Calculate widths for performance columns
            perf_widths = []
            if num_perf_cols > 0 :
                if num_perf_cols == 1: perf_widths = [0.40]
                elif num_perf_cols == 2: perf_widths = [0.175, 0.175] 
                elif num_perf_cols == 3: perf_widths = [0.13, 0.13, 0.13] 
                else: # fallback for more 
                    perf_total_width_available = 1.0 - final_col_widths[0] - final_col_widths[1] - note_width_target
                    if perf_total_width_available < 0: perf_total_width_available = 0.1 * num_perf_cols # ensure some minimal width
                    perf_widths = [perf_total_width_available / num_perf_cols] * num_perf_cols
            
            final_col_widths.extend(perf_widths)
            
            if len(summary_df.columns) > (2 + num_perf_cols): # If Note column exists
                 final_col_widths.append(note_width_target)

            # Normalize final_col_widths to sum to 1.0
            current_total = sum(final_col_widths)
            if not np.isclose(current_total, 1.0) and current_total > 0 and len(final_col_widths) == len(summary_df.columns):
                factor = 1.0 / current_total
                final_col_widths = [w * factor for w in final_col_widths]
            
            if len(final_col_widths) == len(summary_df.columns):
                for i, width in enumerate(final_col_widths):
                    for row_idx in range(len(summary_df) + 1): # Include header
                        table[(row_idx, i)].set_width(width)
                        # Center align Number of Questions and Performance columns
                        if 0 < i <= 1 + num_perf_cols: 
                             table[(row_idx, i)].set_text_props(ha='center')
            else:
                 print("Warning: Column width calculation mismatch for summary table in PDF. Using auto-widths.")
                 # If mismatch, rely on auto-sizing by not setting widths explicitly for all cells.

        # Add note about overall question counts
        count_note = f"Total questions analyzed: {representative_stats.get('total_columns', 'N/A')} " + \
                     f"(Missing: {representative_stats.get('missing_columns', 'N/A')}, Other: {representative_stats.get('other_columns', 'N/A')})"
        plt.text(0.1, 0.25, count_note, \
                 ha='left', va='top', fontsize=9)

        # Add note about metrics explanation
        note_text = (
            "Notes:\n"
            "- Binary Accuracy: Proportion of exact matches for binary responses.\n"
            "- Correlation (ρ): Pearson correlation for numeric responses.\n"
            "- R-squared (R²): Explained variance (correlation squared).\n"
            "- Beta (β): Regression slope, indicating scale alignment.\n"
            "- N/A indicates metric could not be computed (e.g., no valid data).\n"
            "- Avg Non-NA % vs GT: Average percentage of non-missing responses in the model compared to non-missing in ground truth." # Added
        )
        plt.text(0.1, 0.1, note_text,
                ha='left', va='top', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
        
        pdf.savefig(fig)
        plt.close()

        # Retrieve pre-calculated metrics from results_list
        metrics_by_model = {res['name']: res['column_metrics'] for res in results_list}

        # Add detailed metric comparison pages (using pre-calculated metrics)
        def create_metric_comparison_page(metric_key, metric_display_name, column_metrics_by_model, 
                                            mapping_df, note, sort_descending=True):
            """Create a page showing detailed comparison of a metric across columns."""
            # Prepare data for the table
            table_plot_data = []
            all_cols_metric = set() # Columns that have the primary metric

            # Determine a short name for the primary metric for column headers
            short_metric_name = metric_display_name
            if "Accuracy" in metric_display_name: short_metric_name = "Acc"
            elif "Correlation" in metric_display_name: short_metric_name = "Corr"
            elif "R-squared" in metric_display_name: short_metric_name = "R²" # Keep as is
            elif "Beta" in metric_display_name: short_metric_name = "Beta" # Keep as is
            
            # Populate all_cols_metric based on primary metric availability
            for model_metrics in column_metrics_by_model.values():
                all_cols_metric.update(col for col, m in model_metrics.items() if metric_key in m and not pd.isna(m[metric_key]))
            
            map_key_col = 'target_id' if mapping_df is not None and 'target_id' in mapping_df.columns else None

            for col in sorted(list(all_cols_metric)):
                row = {'Column': col}
                # Get metric values for each model
                for model_name, model_col_metrics_dict in column_metrics_by_model.items():
                     col_specific_metrics = model_col_metrics_dict.get(col, {})
                     
                     metric_val = col_specific_metrics.get(metric_key, np.nan)
                     row[f"{model_name} {short_metric_name}"] = metric_val # e.g. LLM Acc, Wave4 Acc
                     
                table_plot_data.append(row)

            if not table_plot_data:
                # Create a new figure specifically for the "No data" message page
                fig_no_data = plt.figure(figsize=(14, 10)) 
                plt.axis('off') 
                
                # Title for this "No data" page
                no_data_page_title = f"{metric_display_name} by Column" 
                plt.text(0.5, 0.95, no_data_page_title, 
                         ha='center', va='top', fontsize=16, fontweight='bold')
                
                plt.text(0.5, 0.5, f"No data available for {metric_display_name}", 
                         ha='center', va='center', fontsize=12)
                pdf.savefig(fig_no_data) 
                plt.close(fig_no_data) # Close the figure for the "No data" page
                return 

            # Convert to DataFrame for display (format numbers after sorting)
            df_plot = pd.DataFrame(table_plot_data)
            # Dynamically create column order for df_plot based on model_names_in_results
            # This ensures consistent column ordering in the PDF table.
            # model_names_in_results is defined in the outer scope of generate_pdf_report
            
            ordered_df_cols = ['Column']
            for model_name_from_outer_scope in model_names_in_results: # Ensures order from results_list
                # Check if columns actually exist in df_plot before adding to ordered list
                primary_metric_col_for_model = f"{model_name_from_outer_scope} {short_metric_name}"
                if primary_metric_col_for_model in df_plot.columns:
                    ordered_df_cols.append(primary_metric_col_for_model)
            
            # Include any other columns that might have been added (e.g., from mapping)
            # For now, this logic assumes only 'Column' and model-specific metrics.
            # If mapping adds columns, they would need to be handled or placed.
            df_plot = df_plot[ordered_df_cols]

            for model_name_col_key in column_metrics_by_model.keys(): # Iterate using keys from input dict
                # Format primary metric
                primary_metric_col_name_in_df = f"{model_name_col_key} {short_metric_name}"
                if primary_metric_col_name_in_df in df_plot:
                    df_plot[primary_metric_col_name_in_df] = df_plot[primary_metric_col_name_in_df].apply(
                        lambda x: f"{x:.3f}" if pd.notna(x) else 'N/A'
                    )
                
            # --- Create table using matplotlib.table --- 
            rows_per_page = 28 
            total_rows = len(df_plot)
            # If table_plot_data was not empty, total_rows > 0, so num_pages >= 1.
            num_pages = int(np.ceil(total_rows / rows_per_page))
            
            for page_num in range(num_pages): # Around L807 in original
                # Create a new figure for EACH page of the table
                fig = plt.figure(figsize=(14, 10)) 
                plt.axis('off') 
                # Removed conditional figure recreation from original L809-L811.
                
                # Title for the current page
                page_title = f"{metric_display_name} by Column" 
                if num_pages > 1:
                    page_title += f" (Page {page_num + 1})"
                plt.text(0.5, 0.95, page_title, ha='center', va='top', fontsize=16, fontweight='bold') 

                start_row = page_num * rows_per_page
                end_row = min(start_row + rows_per_page, total_rows)
                page_df = df_plot.iloc[start_row:end_row]
                
                if page_df.empty and total_rows > 0: # Should not happen if total_rows > 0 implies non-empty df_plot
                    # This case implies an issue with pagination logic if df_plot was not empty
                    plt.text(0.5, 0.5, "Error generating table page.", ha='center', va='center', fontsize=12)
                elif not page_df.empty:
                    table = plt.table(cellText=page_df.values,
                                    colLabels=page_df.columns,
                                    cellLoc='left',
                                    loc='center',
                                    bbox=[0.05, 0.1, 0.9, 0.8]) 
                    
                    table.auto_set_font_size(False)
                    table.set_fontsize(8)
                    table.scale(1, 1.4)

                    col_widths_dynamic = { 'Column': 0.15} 
                    # Calculate total width available for metric columns
                    available_width_for_metrics = 1.0 - col_widths_dynamic['Column']
                    # Distribute width among all actual metric columns present in df_plot
                    total_metric_data_columns = len(df_plot.columns) - 1  # -1 for 'Column'
                    metric_col_width = available_width_for_metrics / max(1, total_metric_data_columns)
                    
                    for i, col_name_in_df in enumerate(page_df.columns): # page_df has the ordered columns
                        width = col_widths_dynamic.get(col_name_in_df, metric_col_width)
                        for row_idx in range(len(page_df) + 1): 
                             table[(row_idx, i)].set_width(width)
                             # Right align model metric columns
                             if col_name_in_df not in ['Column']:
                                  table[(row_idx, i)].set_text_props(ha='right')
                
                # Add note about metric
                plt.text(0.05, 0.05, f"Note: {note}",
                        ha='left', va='top', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
                
                pdf.savefig(fig) # Save current page 
                plt.close(fig) # Always close the figure for the current page

        # --- Generate pages for each metric --- 
        binary_cols_exist = any(m.get('type') == 'binary' for model_metrics_set in metrics_by_model.values() for m in model_metrics_set.values())
        numeric_cols_exist = any(m.get('type') == 'numeric' for model_metrics_set in metrics_by_model.values() for m in model_metrics_set.values())

        # 1. Binary Accuracy Page
        if binary_cols_exist:
            create_metric_comparison_page(
                metric_key='accuracy',
                metric_display_name="Binary Accuracy",
                column_metrics_by_model=metrics_by_model,
                mapping_df=mapping_df,
                note="Higher values indicate better performance (range: 0-1)",
                sort_descending=True
            )
            # Add binary scatter plot page immediately after binary tables
            _create_model_vs_model_performance_scatter_page(pdf, results_list, 'binary')

        # 2. Numeric Metrics Pages
        if numeric_cols_exist:
            # Correlation page
            create_metric_comparison_page(
                metric_key='correlation',
                metric_display_name="Correlation (ρ)",
                column_metrics_by_model=metrics_by_model,
                mapping_df=mapping_df,
                note="Higher absolute values indicate stronger correlation (range: -1 to 1). Sorted by value descending.",
                sort_descending=True
            )
            # Add numeric scatter plot page immediately after numeric tables
            _create_model_vs_model_performance_scatter_page(pdf, results_list, 'numeric')

        # Verify TWIN_ID exists in all dataframes (already done in evaluate_response_accuracy)
        # Merge dataframes on TWIN_ID (already done in evaluate_response_accuracy)
        merged_llm = pd.merge(wave1_3_df, llm_df, on='TWIN_ID', suffixes=('_gt', '_llm'), how='inner')
        merged_w4 = pd.merge(wave1_3_df, wave4_df, on='TWIN_ID', suffixes=('_gt', '_w4'), how='inner')
        
        # Combine all comparison data into one dict keyed by model name for easier access
        comparison_data = {
            'LLM': merged_llm,
            'Wave4': merged_w4
        }
        
        # Load template JSON
        with open(template_json_path, 'r', encoding='utf-8') as f:
            template = json.load(f)

        # Collect all questions recursively
        questions: List[dict] = []
        if isinstance(template, dict):
            questions.extend(_collect_questions_from_block(template))
        elif isinstance(template, list):
            for block in template:
                questions.extend(_collect_questions_from_block(block))

        sns.set(style="whitegrid")

        # Get column order mapping
        column_order_map = {}
        if mapping_df is not None and 'target_id' in mapping_df.columns and 'wave4_order' in mapping_df.columns:
            # Handle potential non-unique target_id, prioritize first occurrence
            mapping_df_unique = mapping_df.drop_duplicates(subset=['target_id'], keep='first')
            for _, row in mapping_df_unique.iterrows():
                if pd.notna(row['target_id']) and pd.notna(row['wave4_order']):
                    column_order_map[row['target_id']] = row['wave4_order']

        # Determine columns to process based on available metrics
        all_processed_cols = set()
        for model_metrics in metrics_by_model.values():
            all_processed_cols.update(model_metrics.keys())
           
        # Sort columns using the order map
        ordered_columns = []
        unordered_columns = []
        for col in all_processed_cols:
            if col in column_order_map:
                ordered_columns.append((col, column_order_map[col]))
            else:
                unordered_columns.append(col)
        
        # Sort the ordered columns by the wave4_order value
        ordered_columns.sort(key=lambda x: x[1])
        # Combine sorted and unsorted columns
        columns_to_process = [col for col, _ in ordered_columns] + sorted(unordered_columns)

        # Filter the specific GridSpec warning before generating the per-column plots
        # as constrained_layout can sometimes issue this harmlessly with plt.axes.
        warnings.filterwarnings(
            "ignore",
            message='There are no gridspecs with layoutgrids. Possibly did not call parent GridSpec with the "figure" keyword',
            category=UserWarning
        )

        for col in columns_to_process:
            # Skip TWIN_ID if it somehow ends up here
            if col == 'TWIN_ID': continue
            
            # Get metrics for this column for all models
            llm_metrics = metrics_by_model.get('LLM', {}).get(col)
            w4_metrics = metrics_by_model.get('Wave4', {}).get(col)
            
            # Get the data series needed for plotting distributions
            gt_series = pd.to_numeric(merged_llm[f"{col}_gt"], errors='coerce') if f"{col}_gt" in merged_llm else pd.Series(dtype=float)
            llm_series = pd.to_numeric(merged_llm[f"{col}_llm"], errors='coerce') if f"{col}_llm" in merged_llm else pd.Series(dtype=float)
            w4_series = pd.to_numeric(merged_w4[f"{col}_w4"], errors='coerce') if f"{col}_w4" in merged_w4 else pd.Series(dtype=float)

            if gt_series.dropna().empty:
                print(f"Skipping plot for {col}: No ground truth data after merge.")
                continue

            # Get question description
            description = ""
            if mapping_df is not None:
                map_key_col = 'target_id' # Assume this is the key
                if map_key_col in mapping_df.columns:
                    desc_row = mapping_df[mapping_df[map_key_col] == col]
                    if not desc_row.empty:
                        description = desc_row.iloc[0].get('description', '')
                
                if not description and '_' in col:
                    base_qid = col.split('_')[0]
                    desc_row = mapping_df[mapping_df[map_key_col] == base_qid]
                    if not desc_row.empty:
                        description = desc_row.iloc[0].get('description', '')
                else:
                    pass

            # Use constrained_layout=True for better automatic spacing
            fig = plt.figure(figsize=(14, 10), constrained_layout=True)
            
            # Create description box
            desc_ax = plt.axes([0.05, 0.85, 0.9, 0.1]) # Position top
            desc_ax.axis('off')
            title_text = f"Analysis for: {col}"
            if description:
                wrapped_text = textwrap.fill(f"{title_text}\n{str(description)}", width=120)
            else:
                wrapped_text = f"{title_text}\n(No description available)"
            desc_ax.text(0.5, 0.5, wrapped_text,
                       ha='center', va='center',
                       fontsize=10,
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

            # Check if discrete data based on unique values in ground truth (or combined?)
            # Using combined seems safer to determine plot type
            all_values_plot = pd.concat([gt_series.dropna(), w4_series.dropna(), llm_series.dropna()])
            unique_vals_plot = sorted(all_values_plot.unique())
            is_discrete_plot = len(unique_vals_plot) > 0 and len(unique_vals_plot) < 10 and all(v == int(v) for v in unique_vals_plot)

            if is_discrete_plot:
                # --- Discrete Plotting --- 
                # Barplot for distributions (left side)
                ax_dist = plt.axes([0.1, 0.4, 0.35, 0.35]) # y pos, height
                
                gt_cat = gt_series.dropna()
                w4_cat = w4_series.dropna()
                llm_cat = llm_series.dropna()
                
                sources = ['Wave1-3', 'Wave4', 'LLM']
                color_map = {'Wave1-3': 'black', 'Wave4': 'green', 'LLM': 'blue'}
                counts_plot = []
                for val in unique_vals_plot:
                    for src, data in zip(sources, [gt_cat, w4_cat, llm_cat]):
                        total = len(data)
                        percentage = (data == val).sum() / total * 100 if total > 0 else 0
                        counts_plot.append({'Value': val, 'Source': src, 'Percentage': percentage})
                count_df_plot = pd.DataFrame(counts_plot)
                
                sns.barplot(
                    data=count_df_plot, x='Value', y='Percentage', hue='Source',
                    hue_order=sources, palette=color_map, ax=ax_dist, dodge=True
                )
                ax_dist.set_title("Response Distribution")
                ax_dist.set_xlabel("Value")
                ax_dist.set_ylabel("Percentage (%)")
                ax_dist.set_xticks(range(len(unique_vals_plot)))
                ax_dist.set_xticklabels([str(int(v)) for v in unique_vals_plot]) # Ensure int labels
                ax_dist.legend(title='Source', loc='upper right')

                # Confusion matrix / Regression panels (right side)
                ax_w4_panel = plt.axes([0.55, 0.48, 0.35, 0.28]) # y pos, height (raised top panel slightly)
                ax_llm_panel = plt.axes([0.55, 0.1, 0.35, 0.28]) # y pos, height (lowered bottom panel)

                def add_discrete_panel(ax, gt_data, resp_data, metrics, title, num_classes, unique_vals):
                    """Add panel for discrete data: confusion matrix, regression line, and metrics."""
                    ax.clear() # Clear axis before plotting
                    
                    # Create confusion matrix
                    value_map = {val: i for i, val in enumerate(unique_vals)} # 0-based index
                    gt_mapped = gt_data.map(value_map).dropna()
                    resp_mapped = resp_data.map(value_map).dropna()
                    
                    # Align indices before creating matrix
                    common_idx = gt_mapped.index.intersection(resp_mapped.index)
                    gt_aligned = gt_mapped.loc[common_idx].astype(int)
                    resp_aligned = resp_mapped.loc[common_idx].astype(int)
                    
                    mat = np.zeros((num_classes, num_classes), dtype=int)
                    if len(gt_aligned) > 0:
                        np.add.at(mat, (resp_aligned, gt_aligned), 1) # (row=resp, col=gt)

                    # Plot confusion matrix heatmap (inverted: gt=x, resp=y)
                    im = ax.imshow(mat.T, aspect='equal', cmap='Greys', origin='lower',
                                 vmin=0, vmax=mat.max() if mat.max() > 0 else 1, alpha=0.8,
                                 extent=[-0.5, num_classes-0.5, -0.5, num_classes-0.5])
                    
                    ax.grid(False)
                    # Add 45-degree reference line (y=x)
                    ax.plot([-0.5, num_classes-0.5], [-0.5, num_classes-0.5], 
                           color='#888888', ls='--', lw=1, zorder=3)
                    
                    # Add regression line using original (unmapped) values
                    paired_data = pd.DataFrame({'gt': gt_data, 'resp': resp_data}).dropna()
                    if len(paired_data) >= 2 and paired_data['gt'].nunique() > 1 and paired_data['resp'].nunique() > 1:
                        x_orig = paired_data['gt'].values
                        y_orig = paired_data['resp'].values
                        try:
                            beta_plot, alpha_plot = np.polyfit(x_orig, y_orig, 1)
                            # Use min/max of actual data for line endpoints
                            x_line_orig = np.array([min(x_orig), max(x_orig)]) 
                            y_line_orig = alpha_plot + beta_plot * x_line_orig
                            # Map line coordinates to plot coordinates (0-based index)
                            x_plot = np.interp(x_line_orig, unique_vals, range(num_classes)) - 0.5
                            y_plot = np.interp(y_line_orig, unique_vals, range(num_classes)) - 0.5
                            ax.plot(x_plot, y_plot, color='#1f78b4', lw=2, zorder=4)
                        except np.linalg.LinAlgError:
                             pass # Cannot plot regression line

                    # Add pre-calculated statistics
                    if metrics:
                         # Use accuracy for binary, correlation/r2/beta for numeric-like discrete
                         acc = metrics.get('accuracy', np.nan)
                         corr = metrics.get('correlation', np.nan)
                         r2 = metrics.get('r2', np.nan)
                         beta = metrics.get('beta', np.nan)
                         stats_text_list = []
                         if not pd.isna(acc):
                             stats_text_list.append(f"Acc={acc:.2f}")
                         if not pd.isna(corr):
                             stats_text_list.append(f"ρ={corr:.2f}")
                         if not pd.isna(r2):
                              stats_text_list.append(f"R²={r2:.2f}")
                         if not pd.isna(beta):
                             stats_text_list.append(f"β={beta:.2f}")
                         stats_text = "\n".join(stats_text_list)
                         
                         bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                         ax.text(0.05, 0.95, stats_text,
                               transform=ax.transAxes, va='top', ha='left',
                               color='#1f78b4', fontsize=9, bbox=bbox_props, zorder=5)
                   
                    # Set ticks and labels using original values
                    tick_labels = [f"{v:.0f}" for v in unique_vals]
                    ax.set_xticks(range(num_classes))
                    ax.set_xticklabels(tick_labels)
                    ax.set_yticks(range(num_classes))
                    ax.set_yticklabels(tick_labels)
                    ax.set_xlabel("Ground-truth (Wave1-3)")
                    ax.set_ylabel("Response")
                    ax.set_xlim(-0.5, num_classes-0.5)
                    ax.set_ylim(-0.5, num_classes-0.5)
                    ax.set_title(title)
                    return im

                # Create panels
                num_classes_plot = len(unique_vals_plot)
                im1 = add_discrete_panel(ax_w4_panel, gt_series, w4_series, w4_metrics, "Wave4 vs Ground-truth", num_classes_plot, unique_vals_plot)
                im2 = add_discrete_panel(ax_llm_panel, gt_series, llm_series, llm_metrics, "LLM vs Ground-truth", num_classes_plot, unique_vals_plot)
                
                # Add a single colorbar for both heatmaps
                fig.colorbar(im1, ax=[ax_w4_panel, ax_llm_panel], label='Count', shrink=0.7, aspect=30)
                
            else:
                # --- Continuous Plotting --- 
                # Distribution plots (left side)
                ax_gt_dist = plt.axes([0.1, 0.65, 0.4, 0.12]) # y, height 
                ax_w4_dist = plt.axes([0.1, 0.50, 0.4, 0.12])
                ax_llm_dist = plt.axes([0.1, 0.35, 0.4, 0.12])
                
                if not gt_series.dropna().empty:
                    sns.histplot(gt_series.dropna(), color='black', ax=ax_gt_dist, stat='percent', kde=True, bins=15, element='step')
                ax_gt_dist.set_title("Wave1-3 (Ground Truth) Distribution")
                ax_gt_dist.set_xlabel("")
                ax_gt_dist.set_ylabel("(%)")
                ax_gt_dist.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # Hide x-axis labels/ticks

                if not w4_series.dropna().empty:
                    sns.histplot(w4_series.dropna(), color='green', ax=ax_w4_dist, stat='percent', kde=True, bins=15, element='step')
                ax_w4_dist.set_title("Wave4 Distribution")
                ax_w4_dist.set_xlabel("")
                ax_w4_dist.set_ylabel("(%)")
                ax_w4_dist.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

                if not llm_series.dropna().empty:
                    sns.histplot(llm_series.dropna(), color='blue', ax=ax_llm_dist, stat='percent', kde=True, bins=15, element='step')
                ax_llm_dist.set_title("LLM Distribution")
                ax_llm_dist.set_xlabel("Value")
                ax_llm_dist.set_ylabel("(%)")
                
                # Share X axis limits for distributions
                min_val = all_values_plot.min()
                max_val = all_values_plot.max()
                ax_gt_dist.set_xlim(min_val, max_val)
                ax_w4_dist.set_xlim(min_val, max_val)
                ax_llm_dist.set_xlim(min_val, max_val)

                # Regression plots (right side)
                ax_reg_w4 = plt.axes([0.55, 0.48, 0.4, 0.28]) # y, height (raised top panel)
                ax_reg_llm = plt.axes([0.55, 0.1, 0.4, 0.28]) # y, height (lowered bottom panel slightly)

                def add_continuous_panel(ax, gt_data, resp_data, metrics, title, point_color, line_color):
                    """Add regression plot panel for continuous data."""
                    ax.clear()
                    common_mask = ~(gt_data.isna() | resp_data.isna())
                    if common_mask.any():
                        x_plot = gt_data[common_mask]
                        y_plot = resp_data[common_mask]
                        # Basic scatter + regression line
                        sns.regplot(x=x_plot, y=y_plot, ax=ax,
                                  scatter_kws={'alpha': 0.2, 's': 15, 'color': point_color}, 
                                  line_kws={'color': line_color, 'lw': 2})
                        
                        # Add pre-calculated stats from metrics dict
                        if metrics:
                            corr = metrics.get('correlation', np.nan)
                            r2 = metrics.get('r2', np.nan)
                            beta = metrics.get('beta', np.nan)
                            stats_text_list = []
                            if not pd.isna(corr):
                                stats_text_list.append(f"ρ={corr:.2f}")
                            if not pd.isna(r2):
                                stats_text_list.append(f"R²={r2:.2f}")
                            if not pd.isna(beta):
                                stats_text_list.append(f"β={beta:.2f}")
                            stats_text = "\n".join(stats_text_list)
                            
                            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                                      va='top', ha='left', fontsize=9,
                                      color=line_color, bbox=bbox_props, zorder=5)
                   
                    ax.set_xlabel('Ground-truth (Wave1-3)')
                    ax.set_ylabel('Response')
                    ax.set_title(title)
                    # Define linestyle only once using ls='--'\n                    ax.plot(lims, lims, \'k\', alpha=0.75, zorder=0, lw=1, ls=\'--\') \n                    ax.set_xlim(lims)\n                    ax.set_ylim(lims)\n

                # Add Wave4 vs GT panel
                add_continuous_panel(ax_reg_w4, gt_series, w4_series, w4_metrics, 
                                     "Wave4 vs Ground-truth", 'green', 'darkgreen')
                
                # Add LLM vs GT panel
                add_continuous_panel(ax_reg_llm, gt_series, llm_series, llm_metrics, 
                                     "LLM vs Ground-truth", 'blue', 'darkblue')

            # Add column ID and Wave4 order info at the bottom
            info_text = f"Column ID: {col} | Wave4 Order: {column_order_map.get(col, 'N/A')}"
            plt.figtext(0.5, 0.02, info_text,
                      ha='center', va='bottom',
                      fontsize=8,
                      bbox=dict(facecolor='white', alpha=0.7, edgecolor='lightgray', boxstyle='round,pad=0.3'))

            pdf.savefig(fig)
            plt.close(fig)

        # Add the model performance comparison scatter plot page -- This call is now removed from here
        # _create_model_vs_model_performance_scatter_page(pdf, results_list) -- OLD CALL REMOVED

    print(f"Saved PDF report to {output_pdf_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate response accuracy against ground truth")
    parser.add_argument("--config", required=True, help="Path to configuration YAML file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = config['pipeline']['output_dir']
    csv_dir = os.path.join(output_dir, 'csv_comparison')
    eval_dir = os.path.join(output_dir, 'accuracy_evaluation')
    mapping_path = os.path.join(output_dir, 'mega_persona_csv', 'question_description_map.csv')
    
    # Load the data
    try:
        llm_df = pd.read_csv(os.path.join(csv_dir, 'responses_llm_imputed.csv'))
        wave4_df = pd.read_csv(os.path.join(csv_dir, 'responses_wave4.csv'))
        groundtruth_df = pd.read_csv(os.path.join(csv_dir, 'responses_wave1_3.csv'))
        
        # Load ML data if available
        ml_df = None
        ml_df_path = os.path.join(csv_dir, 'response_ml_predicted_best.csv')
        if os.path.exists(ml_df_path):
            try:
                ml_df = pd.read_csv(ml_df_path)
                print(f"Loaded ML predictions from {ml_df_path}")
            except Exception as e:
                print(f"Warning: Could not load ML prediction file {ml_df_path}: {e}")
                ml_df = None # Ensure ml_df is None if loading fails
        else:
            print(f"Info: ML prediction file not found at {ml_df_path}. Proceeding without ML comparison for some parts.")

    except FileNotFoundError as e:
        print(f"Error loading input CSV file: {e}")
        print("Please ensure the csv_comparison step has been run successfully.")
        return # Exit if files are missing
    
    # Pre-determine column types based on groundtruth and wave4
    column_definitive_types: Dict[str, str] = {}
    if 'groundtruth_df' in locals() and groundtruth_df is not None:
        # wave4_df might be None if its file was not found or failed to load, 
        # get_definitive_column_type handles this.
        wave4_for_typing = locals().get('wave4_df') 
        
        # Use QID columns from groundtruth as the primary set for typing
        question_cols_for_typing = [q_col for q_col in groundtruth_df.columns if q_col.startswith('QID')]
        for col_name in question_cols_for_typing:
            column_definitive_types[col_name] = get_definitive_column_type(col_name, groundtruth_df, wave4_for_typing)

    # Evaluate comparisons
    results_list = []
    if 'llm_df' in locals() and llm_df is not None and 'groundtruth_df' in locals() and groundtruth_df is not None:
        results_list.append(evaluate_response_accuracy(llm_df, groundtruth_df, "LLM", column_definitive_types))
    if 'wave4_df' in locals() and wave4_df is not None and 'groundtruth_df' in locals() and groundtruth_df is not None:
        results_list.append(evaluate_response_accuracy(wave4_df, groundtruth_df, "Wave4", column_definitive_types))
    
    if ml_df is not None and 'groundtruth_df' in locals() and groundtruth_df is not None:
        results_list.append(evaluate_response_accuracy(ml_df, groundtruth_df, "ML", column_definitive_types))
    
    # Print results
    # if results_list: # Only print if there's something to print
    #     print_evaluation_results(results_list)
    # else:
    #     print("No data available to print evaluation results.")
    
    # Save results as CSVs
    if results_list: # Only save if there's something to save
        save_evaluation_results(results_list, eval_dir, mapping_path)
    else:
        print("No data available to save evaluation results.")

    # Generate PDF report with distributions and regression plots
    # Ensure groundtruth_df is loaded before proceeding to PDF generation
    if 'groundtruth_df' not in locals() or groundtruth_df is None:
        print("Error: Ground truth data not loaded. Cannot generate PDF report.")
        return

    try:
        mapping_df = None
        if os.path.exists(mapping_path):
            try:
                mapping_df = pd.read_csv(mapping_path)
                print(f"Loaded mapping file with {len(mapping_df)} rows and columns: {list(mapping_df.columns)}")
            except Exception as e:
                print(f"Warning: Error loading mapping file: {e}")
        else:
            print(f"Warning: Mapping file not found at {mapping_path}")

        pdf_path = os.path.join(eval_dir, 'response_distributions_and_regressions.pdf')
        template_path = os.path.join('Data', 'wave_qsf_json', 'Digital_Twins_-_Wave_4_parsed.json')
        if not os.path.exists(template_path):
             print(f"Error: Template JSON file not found at {template_path}")
             raise FileNotFoundError(f"Template JSON file not found: {template_path}")
            
        generate_pdf_report(
            template_json_path=template_path,
            llm_df=llm_df,
            wave4_df=wave4_df,
            wave1_3_df=groundtruth_df,
            mapping_df=mapping_df,
            output_pdf_path=pdf_path,
            config=config,
            results_list=results_list
        )
    except FileNotFoundError as e: # Catch file not found specifically
        print(f"Error generating PDF report: {e}")
    except Exception as e:
        import traceback
        print(f"Warning: Failed to generate PDF report due to an unexpected error: {e}")
        print(traceback.format_exc()) # Print stack trace for debugging

if __name__ == "__main__":
    main() 