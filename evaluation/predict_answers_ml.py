#!/usr/bin/env python3
import pandas as pd
import numpy as np
import json
import os
import argparse
import yaml
from typing import Dict, List, Any, Tuple
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import xgboost as xgb
from .extract_mega_persona_csv import extract_answers_from_mega_persona_json

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

def extract_combined_data(mega_persona_dir: str, answer_blocks_dir: str, output_csv: str) -> pd.DataFrame:
    """
    Extract data from both mega_persona and answer_blocks json files and combine them into a single dataframe.
    
    Args:
        mega_persona_dir: Directory containing mega_persona JSON files
        answer_blocks_dir: Directory containing answer_blocks JSON files
        output_csv: Path to save the combined CSV
    
    Returns:
        Combined DataFrame with both feature and target columns
    """
    # Collect all person IDs
    all_pids = set()
    
    # Find all PIDs from mega_persona files
    for filename in os.listdir(mega_persona_dir):
        if "mega_persona.json" in filename:
            pid = filename.split('_')[1]  # Extract PID from filename
            all_pids.add(pid)
    
    print(f"Found {len(all_pids)} unique person IDs")
    
    # Extract data for each person
    all_data = []
    
    for pid in all_pids:
        person_data = {"TWIN_ID": pid}
        
        # Extract mega_persona features
        mega_persona_file = os.path.join(mega_persona_dir, f"pid_{pid}_mega_persona.json")
        if os.path.exists(mega_persona_file):
            mega_persona_answers = extract_answers_from_mega_persona_json(mega_persona_file)
            # Add prefix to distinguish from target features
            feature_data = {f"feature_{k}": v for k, v in mega_persona_answers.items() if k != "TWIN_ID" and k != "WAVE"}
            person_data.update(feature_data)
        
        # Extract answer_blocks targets (both wave1_3 and wave4)
        for wave in ["wave1_3", "wave4"]:
            answer_block_file = os.path.join(answer_blocks_dir, f"pid_{pid}_wave4_Q_{wave}_A.json")
            if os.path.exists(answer_block_file):
                answer_block_data = extract_answers_from_mega_persona_json(answer_block_file)
                # Add prefix and wave to distinguish from feature columns
                target_data = {f"target_{wave}_{k}": v for k, v in answer_block_data.items() if k != "TWIN_ID" and k != "WAVE"}
                person_data.update(target_data)
                person_data[f"has_{wave}"] = 1
            else:
                person_data[f"has_{wave}"] = 0
        
        # Only include if we have at least one set of targets
        if person_data.get("has_wave1_3", 0) + person_data.get("has_wave4", 0) > 0:
            all_data.append(person_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved combined data to {output_csv} with {len(df)} rows and {len(df.columns)} columns")
    
    return df

def train_and_evaluate_models(df: pd.DataFrame, output_prefix: str) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    """
    Train ML models to predict answer block responses from mega persona features.
    Uses 2-fold cross-validation, reports accuracy/correlation metrics,
    and collects predictions for all models.
    
    Args:
        df: DataFrame with combined feature and target columns (must include 'TWIN_ID')
        output_prefix: Prefix for output files for metrics
    
    Returns:
        A tuple containing:
        - all_models_predictions: Dict[str, pd.DataFrame] where keys are model names
          and values are DataFrames with 'TWIN_ID' and predictions for all targets.
        - results: Dict containing detailed metrics.
    """
    # Separate features and targets
    feature_cols = [col for col in df.columns if col.startswith("feature_")]
    target_cols_wave1_3 = [col for col in df.columns if col.startswith("target_wave1_3_")]
    
    print(f"Found {len(feature_cols)} feature columns and {len(target_cols_wave1_3)} target columns for wave1_3")
    
    # Keep only numeric feature columns
    numeric_features = []
    for col in feature_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().sum() < len(df[col]) * 0.5: # Keep if less than 50% NaN
                numeric_features.append(col)
        except:
            pass # Ignore columns that can't be made numeric
    
    print(f"Using {len(numeric_features)} numeric feature columns after filtering")
    
    # Define models to evaluate
    models = {
        "OLS": LinearRegression(),
        "LASSO": Lasso(alpha=0.1, max_iter=50000, tol=1e-4), # Increased max_iter further and set tol
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    }
    
    # Set up 2-fold cross-validation
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    
    results = {
        "binary_metrics": {},
        "numeric_metrics": {},
        "notes": {},
        "overall_stats": {
            "total_columns": len(target_cols_wave1_3),
            "binary_columns": 0,
            "numeric_columns": 0,
            "invalid_columns": 0,
            "skipped_few_samples_targets": 0
        }
    }
    
    # Temporary storage for predictions: Dict[model_name, Dict[clean_target, pd.Series]]
    # This replaces the initial loop that created all_models_predictions[model_name] DataFrames.
    temp_model_predictions_series: Dict[str, Dict[str, pd.Series]] = {model_name: {} for model_name in models.keys()}

    # For each target column from wave1_3
    for target_col in target_cols_wave1_3:
        clean_target = target_col.replace("target_wave1_3_", "")
        
        # Get ground truth values (non-null)
        mask = ~df[target_col].isna()
        if mask.sum() < 10:  # Skip if too few samples
            results["notes"][clean_target] = f"Skipped: Too few samples ({mask.sum()}) for target {target_col}"
            results["overall_stats"]["skipped_few_samples_targets"] += 1
            continue
            
        X = df.loc[mask, numeric_features].fillna(0) # Fill NaNs in features with 0
        y = df.loc[mask, target_col]
        
        # Try converting to numeric
        try:
            y = pd.to_numeric(y)
        except ValueError: # Handles non-convertible strings
            results["notes"][clean_target] = f"Skipped: Non-numeric data in target {target_col}"
            results["overall_stats"]["invalid_columns"] += 1
            continue
        
        unique_vals = set(y.unique())
        is_binary = unique_vals.issubset({1, 2, 1.0, 2.0}) # Allow float representations
        
        if is_binary:
            results["overall_stats"]["binary_columns"] += 1
            results["binary_metrics"].setdefault(clean_target, {})
        else:
            results["overall_stats"]["numeric_columns"] += 1
            results["numeric_metrics"].setdefault(clean_target, {})
        
        # For each model
        for model_name, model in models.items():
            if X.empty or len(X) < 2: # Ensure X is not empty and has enough samples for CV
                results["notes"][f"{clean_target}_{model_name}"] = f"Skipped model: Not enough samples in X ({len(X)}) or X is empty."
                continue
                
            try:
                y_pred_cv = np.zeros_like(y, dtype=float)
                
                for train_idx, test_idx in kf.split(X, y): # Pass y to kf.split for classification tasks if needed later
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train = y.iloc[train_idx]
                    
                    if X_train.empty: # Skip fold if X_train is empty
                        results["notes"][f"{clean_target}_{model_name}_fold_skip"] = "Skipped fold due to empty X_train"
                        y_pred_cv[test_idx] = np.nan # Mark predictions as NaN for this fold
                        continue

                    model.fit(X_train, y_train)
                    fold_preds = model.predict(X_test)
                    y_pred_cv[test_idx] = fold_preds
                
                # Create the complete prediction series for this target and model
                model_target_predictions = pd.Series(np.nan, index=df.index)
                if is_binary:
                    # Round binary predictions and ensure they are int
                    # y_pred_cv might contain NaNs if folds were skipped, handle them before astype(int)
                    # Create a temporary series from y_pred_cv to handle mask and rounding
                    temp_preds_for_mask = pd.Series(y_pred_cv, index=y.index) # y.index is the 'mask' index
                    rounded_preds = np.round(temp_preds_for_mask.dropna()) # Round valid predictions
                    # Assign back to model_target_predictions, which has the full df.index
                    # Only assign where original y (and thus y_pred_cv for those indices) was not NaN
                    # and where rounded_preds are not NaN
                    model_target_predictions.loc[mask & rounded_preds.notna().reindex(mask.index, fill_value=False)] = rounded_preds.astype(int)
                else:
                    model_target_predictions.loc[mask] = y_pred_cv
                
                # Store the fully formed series in the temporary structure
                temp_model_predictions_series[model_name][clean_target] = model_target_predictions

                # Evaluate
                if is_binary:
                    y_pred_binary_eval = np.round(y_pred_cv).astype(int)
                    accuracy = accuracy_score(y, y_pred_binary_eval)
                    results["binary_metrics"][clean_target][model_name] = {
                        "accuracy": float(accuracy), "samples": int(len(y))
                    }
                else:
                    valid_corr_indices = ~np.isnan(y_pred_cv) & ~np.isnan(y)
                    correlation = np.nan
                    if np.sum(valid_corr_indices) > 1:
                        # Check for constant series before correlation
                        if len(set(y[valid_corr_indices])) > 1 and len(set(y_pred_cv[valid_corr_indices])) > 1:
                            correlation = np.corrcoef(y[valid_corr_indices], y_pred_cv[valid_corr_indices])[0, 1]
                        else: # One or both series are constant
                            correlation = 0.0 if len(set(y[valid_corr_indices])) == len(set(y_pred_cv[valid_corr_indices])) else np.nan

                    results["numeric_metrics"][clean_target][model_name] = {
                        "correlation": float(correlation) if not np.isnan(correlation) else None,
                        "samples": int(len(y)),
                        "mean_difference": float((y_pred_cv[valid_corr_indices] - y[valid_corr_indices]).mean()) if np.sum(valid_corr_indices) > 0 else None,
                        "std_difference": float((y_pred_cv[valid_corr_indices] - y[valid_corr_indices]).std()) if np.sum(valid_corr_indices) > 1 else None
                    }
            except Exception as e:
                results["notes"][f"{clean_target}_{model_name}"] = f"Error during training/prediction: {str(e)}"
    
    # Construct final DataFrames for all_models_predictions from the collected series
    all_models_predictions: Dict[str, pd.DataFrame] = {} 

    for model_name, series_map in temp_model_predictions_series.items():
        # Start with TWIN_ID if available, otherwise an empty DataFrame with the correct index
        if 'TWIN_ID' in df.columns:
            # Ensure it's a DataFrame to start with, with the correct index from the original df
            model_df = pd.DataFrame({'TWIN_ID': df['TWIN_ID']}, index=df.index)
        else:
            model_df = pd.DataFrame(index=df.index)

        if series_map: # If there are actual predictions (series) for this model
            # Create a DataFrame from all collected series for this model
            # The series_map keys will become column names, and series are already indexed by df.index
            predictions_data_df = pd.DataFrame(series_map) 
            # Concatenate prediction columns to model_df (which has TWIN_ID or is an empty indexed DF)
            model_df = pd.concat([model_df, predictions_data_df], axis=1)
        
        all_models_predictions[model_name] = model_df

    # Calculate overall statistics for each model
    for model_name in models.keys():
        binary_accs = [metrics[model_name]["accuracy"] 
                     for metrics in results["binary_metrics"].values() 
                     if model_name in metrics and "accuracy" in metrics[model_name]]
        
        numeric_corrs = [metrics[model_name]["correlation"] 
                       for metrics in results["numeric_metrics"].values() 
                       if model_name in metrics and metrics[model_name].get("correlation") is not None]
        
        results["overall_stats"][f"{model_name}_avg_binary_accuracy"] = float(np.mean(binary_accs)) if binary_accs else 0.0
        results["overall_stats"][f"{model_name}_avg_correlation"] = float(np.mean(numeric_corrs)) if numeric_corrs else 0.0
    
    # Save detailed metrics (as before)
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    with open(f"{output_prefix}_results.json", 'w') as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2)
    
    # Save summary CSV files for metrics (as before)
    binary_data = {}
    for col, metrics_dict in results["binary_metrics"].items():
        binary_data[col] = {f"{model}_accuracy": m.get("accuracy") for model, m in metrics_dict.items()}
    
    numeric_data = {}
    for col, metrics_dict in results["numeric_metrics"].items():
        numeric_data[col] = {f"{model}_correlation": m.get("correlation") for model, m in metrics_dict.items()}
    
    if binary_data:
        binary_df = pd.DataFrame.from_dict(binary_data, orient='index')
        binary_df.to_csv(f"{output_prefix}_binary_metrics.csv")
        print(f"Saved binary metrics to: {output_prefix}_binary_metrics.csv")
    
    if numeric_data:
        numeric_df = pd.DataFrame.from_dict(numeric_data, orient='index')
        numeric_df.to_csv(f"{output_prefix}_numeric_metrics.csv")
        print(f"Saved numeric metrics to: {output_prefix}_numeric_metrics.csv")
    
    overall_df = pd.DataFrame.from_dict({k: [v] for k, v in results["overall_stats"].items()})
    overall_df.to_csv(f"{output_prefix}_overall_stats.csv", index=False)
    print(f"Saved overall statistics to: {output_prefix}_overall_stats.csv")

    print("\n=== ML Prediction Metrics Summary ===")
    print(f"Total target columns processed: {results['overall_stats']['total_columns']}")
    print(f"Binary columns: {results['overall_stats']['binary_columns']}")
    print(f"Numeric columns: {results['overall_stats']['numeric_columns']}")
    print(f"Invalid columns (non-numeric target): {results['overall_stats']['invalid_columns']}")
    print(f"Targets skipped (too few samples): {results['overall_stats']['skipped_few_samples_targets']}")
    
    for model_name in models.keys():
        print(f"\n{model_name} Performance:")
        avg_bin_acc = results['overall_stats'].get(f"{model_name}_avg_binary_accuracy", 0.0)
        avg_num_corr = results['overall_stats'].get(f"{model_name}_avg_correlation", 0.0)
        print(f"  Avg Binary Accuracy: {avg_bin_acc:.3f}")
        print(f"  Avg Numeric Correlation: {avg_num_corr:.3f}")

    return all_models_predictions, results

def save_ml_predictions_csv(predictions_df: pd.DataFrame, output_path: str):
    """Saves the ML predictions to a CSV file."""
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Ensure TWIN_ID is first column if it exists
        if 'TWIN_ID' in predictions_df.columns:
            cols = ['TWIN_ID'] + [col for col in predictions_df.columns if col != 'TWIN_ID']
            predictions_df = predictions_df[cols]
        predictions_df.to_csv(output_path, index=False)
        print(f"Successfully saved ML predictions to: {output_path}")
    except Exception as e:
        print(f"Error saving ML predictions CSV to {output_path}: {e}")

def determine_and_save_best_model_predictions(
    all_models_predictions: Dict[str, pd.DataFrame],
    results: Dict,
    output_dir_base: str, # This is the main output_dir from config
    df_for_twin_id: pd.DataFrame # Original df to get TWIN_ID column
):
    """
    Determines the best model for each question and saves a composite prediction CSV.
    """
    print("\nDetermining best model predictions per question...")
    
    if "TWIN_ID" not in df_for_twin_id.columns:
        print("Error: TWIN_ID not found in the provided DataFrame for best predictions.")
        return

    # Initialize DataFrame with only TWIN_ID
    best_predictions_df = pd.DataFrame({'TWIN_ID': df_for_twin_id['TWIN_ID']})
    
    # List to hold all new column Series
    columns_to_add = []
    
    processed_targets = set(results["binary_metrics"].keys()) | set(results["numeric_metrics"].keys())
    
    if not processed_targets:
        print("No processed targets found in results. Skipping best model CSV generation.")
        # If no targets, we still might want to save the TWIN_ID column, or handle as per requirements.
        # For now, it will save a CSV with only TWIN_ID if output_path is defined.
        # Consider if an empty CSV or no CSV is better in this case.
        output_path = os.path.join(output_dir_base, "csv_comparison", "response_ml_predicted_best.csv")
        save_ml_predictions_csv(best_predictions_df, output_path) # Save even if only TWIN_ID
        return

    for clean_target in processed_targets:
        best_model_name = None
        best_metric_value = -float('inf')
        is_target_binary = clean_target in results["binary_metrics"]
        
        # Determine best model for the current target
        if is_target_binary:
            if clean_target in results["binary_metrics"]:
                for model_name, metrics in results["binary_metrics"][clean_target].items():
                    accuracy = metrics.get("accuracy")
                    if accuracy is not None and accuracy > best_metric_value:
                        best_metric_value = accuracy
                        best_model_name = model_name
        else: # Numeric
            if clean_target in results["numeric_metrics"]:
                for model_name, metrics in results["numeric_metrics"][clean_target].items():
                    correlation = metrics.get("correlation")
                    if correlation is not None and correlation > best_metric_value:
                        best_metric_value = correlation
                        best_model_name = model_name
        
        # Prepare the series for the current target's predictions
        current_target_series = pd.Series(np.nan, index=df_for_twin_id.index, name=clean_target)

        if best_model_name and best_model_name in all_models_predictions:
            if clean_target in all_models_predictions[best_model_name].columns:
                current_target_series = all_models_predictions[best_model_name][clean_target].copy()
                current_target_series.name = clean_target # Ensure series is named correctly
            else:
                print(f"  Warning: Predictions for {clean_target} not found for best model {best_model_name}. Series remains NaNs.")
        else:
            # print(f"  Warning: No best model found for {clean_target} or predictions unavailable. Series remains NaNs.")
            pass # current_target_series is already initialized with NaNs and named

        columns_to_add.append(current_target_series)

    # Concatenate all collected Series (columns) to the best_predictions_df
    if columns_to_add:
        best_predictions_df = pd.concat([best_predictions_df] + columns_to_add, axis=1)

    output_path = os.path.join(output_dir_base, "csv_comparison", "response_ml_predicted_best.csv")
    save_ml_predictions_csv(best_predictions_df, output_path)

def check_output_files_exist(output_dir: str) -> bool:
    """
    Check if key output files from previous ML prediction run exist.
    
    Args:
        output_dir: Base output directory from config
        
    Returns:
        bool: True if all key output files exist, False otherwise
    """
    # Define paths to check
    csv_comparison_dir = os.path.join(output_dir, "csv_comparison")
    metrics_dir = os.path.join(output_dir, "accuracy_evaluation")
    
    # Key files to check
    files_to_check = [
        os.path.join(csv_comparison_dir, "response_ml_predicted_best.csv"),
        os.path.join(csv_comparison_dir, "response_ml_predicted_OLS.csv"),
        os.path.join(csv_comparison_dir, "response_ml_predicted_LASSO.csv"),
        os.path.join(csv_comparison_dir, "response_ml_predicted_XGBoost.csv"),
        os.path.join(metrics_dir, "ml_prediction_results.json"),
        os.path.join(metrics_dir, "ml_prediction_binary_metrics.csv"),
        os.path.join(metrics_dir, "ml_prediction_numeric_metrics.csv"),
        os.path.join(metrics_dir, "ml_prediction_overall_stats.csv")
    ]
    
    # Check if all files exist
    return all(os.path.exists(file) for file in files_to_check)

def main():
    parser = argparse.ArgumentParser(description='ML-based prediction of answers using mega persona features')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the YAML configuration file')
    args = parser.parse_args()

    # Load config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Extract output_dir and rerun flag from config
    output_dir = config.get('pipeline', {}).get('output_dir')
    if not output_dir:
        raise ValueError("pipeline.output_dir not found in the configuration file.")
    
    # Get rerun flag from config, default to True if not specified
    rerun = config.get('pipeline', {}).get('rerun', True)
    
    # Check if output files exist and skip if rerun=False
    if not rerun and check_output_files_exist(output_dir):
        print("ML prediction output files already exist and rerun=False in config. Skipping ML prediction step.")
        return

    # Derive paths
    mega_persona_dir = os.path.join(output_dir, "mega_persona")
    answer_blocks_dir = os.path.join(output_dir, "answer_blocks")
    
    output_csv_dir = os.path.join(output_dir, "mega_persona_csv")
    output_csv_path = os.path.join(output_csv_dir, "combined_features_targets.csv")
    
    metrics_output_prefix_dir = os.path.join(output_dir, "accuracy_evaluation")
    metrics_output_prefix_path = os.path.join(metrics_output_prefix_dir, "ml_prediction") # For metrics JSON/CSVs
    
    # Create output directories for metrics and combined data
    os.makedirs(output_csv_dir, exist_ok=True)
    os.makedirs(metrics_output_prefix_dir, exist_ok=True)
    
    # Extract and combine data
    # This df will also be used to get the full list of TWIN_IDs for the 'best' predictions df
    df_combined = extract_combined_data(mega_persona_dir, answer_blocks_dir, output_csv_path)
    
    if df_combined.empty:
        print("No data extracted. Exiting ML prediction script.")
        return

    # Train and evaluate models, and get all model predictions and results
    all_predictions, detailed_results = train_and_evaluate_models(df_combined, metrics_output_prefix_path)
    
    # Define directory for prediction CSVs
    ml_response_output_dir = os.path.join(output_dir, "csv_comparison")
    os.makedirs(ml_response_output_dir, exist_ok=True) # Ensure it exists

    # Save predictions for each model
    if all_predictions:
        for model_name, predictions_df in all_predictions.items():
            if not predictions_df.empty:
                file_name = f"response_ml_predicted_{model_name}.csv"
                output_path = os.path.join(ml_response_output_dir, file_name)
                save_ml_predictions_csv(predictions_df, output_path)
            else:
                print(f"No predictions DataFrame found for model {model_name} or it's empty.")
    else:
        print("No model predictions were generated.")

    # Determine and save best model predictions
    if all_predictions and detailed_results:
        determine_and_save_best_model_predictions(all_predictions, detailed_results, output_dir, df_combined)
    else:
        print("Skipping best model CSV generation due to missing predictions or results.")

if __name__ == "__main__":
    main() 