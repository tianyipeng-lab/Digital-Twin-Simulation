#!/usr/bin/env python3
"""
Simplified XGBoost-only prediction script for survey answers.
Uses nested cross-validation with hyperparameter tuning.
"""

import pandas as pd
import numpy as np
import json
import os
import argparse
import yaml
from typing import Dict, Tuple, Optional
from pathlib import Path

try:
    from evaluation.json2csv import AnswerExtractor, ExtractionMode
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from evaluation.json2csv import AnswerExtractor, ExtractionMode

# Import XGBoost and numpy encoder
import xgboost as xgb
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from scipy.stats import uniform, randint

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


def extract_training_data(mega_persona_dir: str, answer_blocks_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract training features and target labels from JSON files.
    
    Returns:
        Tuple of (training_df, label_df)
    """
    print("Extracting data from JSON files...")
    
    # Find all person IDs
    all_pids = set()
    for filename in os.listdir(mega_persona_dir):
        if "mega_persona.json" in filename:
            pid = filename.split('_')[1]
            all_pids.add(pid)
    
    print(f"Found {len(all_pids)} unique person IDs")
    
    # Extract data
    training_data = []
    label_data = []
    extractor = AnswerExtractor(ExtractionMode.NUMERIC)
    
    for pid in all_pids:
        # Extract features from mega_persona
        mega_persona_file = os.path.join(mega_persona_dir, f"pid_{pid}_mega_persona.json")
        if os.path.exists(mega_persona_file):
            features = extractor.extract_from_file(mega_persona_file)
            if features and 'TWIN_ID' not in features:
                features['TWIN_ID'] = pid
            training_data.append(features)
        
        # Extract labels from answer_blocks (wave1_3)
        answer_file = os.path.join(answer_blocks_dir, f"pid_{pid}_wave4_Q_wave1_3_A.json")
        if os.path.exists(answer_file):
            labels = extractor.extract_from_file(answer_file)
            if labels and 'TWIN_ID' not in labels:
                labels['TWIN_ID'] = pid
            label_data.append(labels)
    
    # Convert to DataFrames
    training_df = pd.DataFrame(training_data)
    label_df = pd.DataFrame(label_data)
    
    # Remove WAVE column if present
    for df in [training_df, label_df]:
        if 'WAVE' in df.columns:
            df.drop('WAVE', axis=1, inplace=True)
    
    print(f"Training data: {len(training_df)} rows, {len(training_df.columns)} columns")
    print(f"Label data: {len(label_df)} rows, {len(label_df.columns)} columns")
    
    return training_df, label_df


def train_xgboost_predictions(training_df: pd.DataFrame, label_df: pd.DataFrame,
                             output_dir: str, cv_folds: int = 3,
                             inner_cv_folds: int = 3, n_iter: int = 10,
                             use_nested_cv: bool = True) -> Dict:
    """
    Train XGBoost model and generate predictions for all targets.
    
    Args:
        training_df: Features DataFrame (with TWIN_ID)
        label_df: Targets DataFrame (with TWIN_ID)
        output_dir: Directory to save outputs
        cv_folds: Number of outer CV folds (default 3)
        inner_cv_folds: Number of inner CV folds for hyperparameter tuning (default 3)
        n_iter: Number of iterations for random search (default 10)
        use_nested_cv: Whether to use nested CV for hyperparameter tuning
        
    Returns:
        Dictionary containing predictions and results
    """
    # Merge data
    df = pd.merge(training_df, label_df, on='TWIN_ID', how='inner')
    print(f"Merged data: {len(df)} samples")
    
    # Identify feature and target columns
    feature_cols = [col for col in training_df.columns if col != 'TWIN_ID']
    target_cols = [col for col in label_df.columns if col != 'TWIN_ID']
    
    # Filter to numeric features only
    numeric_features = []
    for col in feature_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].notna().sum() > len(df) * 0.5:  # Keep if >50% non-null
                numeric_features.append(col)
        except:
            pass
    
    print(f"Using {len(numeric_features)} numeric features")
    print(f"Processing {len(target_cols)} target columns...")
    
    # Storage for results
    prediction_series = []  # Collect all predictions here
    results = {
        'metrics': {},
        'hyperparameters': {},
        'summary': {
            'total_targets': len(target_cols),
            'successful_targets': 0,
            'failed_targets': 0,
            'cv_folds': cv_folds,
            'nested_cv': use_nested_cv
        }
    }
    
    # Set up cross-validation
    outer_cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=inner_cv_folds, shuffle=True, random_state=42)
    
    # XGBoost hyperparameter search space
    xgb_param_dist = {
        'n_estimators': randint(50, 500),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_weight': randint(1, 10),
        'gamma': uniform(0, 0.5),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 2)
    }
    
    # Process each target
    for idx, target_col in enumerate(target_cols, 1):
        print(f"\nProcessing target {idx}/{len(target_cols)}: {target_col}")
        
        # Get non-null samples for this target
        mask = df[target_col].notna()
        if mask.sum() < 10:
            print(f"  Skipping: Too few samples ({mask.sum()})")
            results['summary']['failed_targets'] += 1
            continue
        
        X = df.loc[mask, numeric_features].fillna(0)
        y = df.loc[mask, target_col]
        
        # Check if target is numeric
        try:
            y = pd.to_numeric(y)
        except:
            print(f"  Skipping: Non-numeric target")
            results['summary']['failed_targets'] += 1
            continue
        
        # Check if binary
        unique_vals = set(y.unique())
        is_binary = unique_vals.issubset({1, 2, 1.0, 2.0})
        
        try:
            if is_binary:
                # Convert labels to 0,1 for XGBoost
                y_binary = (y == 2).astype(int)
                y_to_use = y_binary
            else:
                y_to_use = y
            
            if use_nested_cv:
                # Nested CV with hyperparameter tuning
                y_pred_cv = np.zeros(len(y_to_use))
                fold_hyperparams = []
                
                for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y_to_use)):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y_to_use.iloc[train_idx], y_to_use.iloc[test_idx]
                    
                    # Set up XGBoost model
                    xgb_model = xgb.XGBRegressor(
                        objective='reg:squarederror' if not is_binary else 'binary:logistic',
                        random_state=42,
                        n_jobs=-1,
                        early_stopping_rounds=10,
                        eval_metric='rmse' if not is_binary else 'logloss'
                    )
                    
                    # Random search
                    random_search = RandomizedSearchCV(
                        xgb_model,
                        param_distributions=xgb_param_dist,
                        n_iter=n_iter,
                        cv=inner_cv,
                        scoring='neg_mean_squared_error' if not is_binary else 'accuracy',
                        n_jobs=-1,
                        random_state=42,
                        verbose=0
                    )
                    
                    # Fit with early stopping
                    eval_set = [(X_train, y_train)]
                    fit_params = {'eval_set': eval_set, 'verbose': False}
                    random_search.fit(X_train, y_train, **fit_params)
                    
                    # Get best model and refit
                    best_model = random_search.best_estimator_
                    fold_hyperparams.append(random_search.best_params_)
                    
                    # Predict on test fold
                    y_pred_cv[test_idx] = best_model.predict(X_test)
                
                results['hyperparameters'][target_col] = fold_hyperparams
                
            else:
                # Simple CV without tuning
                xgb_model = xgb.XGBRegressor(
                    objective='reg:squarederror' if not is_binary else 'binary:logistic',
                    random_state=42,
                    n_jobs=-1,
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1
                )
                
                y_pred_cv = np.zeros(len(y_to_use))
                
                for train_idx, test_idx in outer_cv.split(X, y_to_use):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y_to_use.iloc[train_idx], y_to_use.iloc[test_idx]
                    
                    xgb_model.fit(X_train, y_train)
                    y_pred_cv[test_idx] = xgb_model.predict(X_test)
            
            # Store predictions
            pred_series = pd.Series(np.nan, index=df.index, name=target_col)
            if is_binary:
                # Convert back to 1,2 labels
                y_pred_binary = np.round(y_pred_cv).astype(int)
                y_pred_original = y_pred_binary + 1  # Convert 0,1 back to 1,2
                pred_series.loc[mask] = y_pred_original
                
                # Calculate accuracy
                accuracy = accuracy_score(y, y_pred_original)
                metrics = {'accuracy': accuracy}
            else:
                pred_series.loc[mask] = y_pred_cv
                
                # Calculate correlation
                correlation = np.corrcoef(y, y_pred_cv)[0, 1] if len(y) > 1 else None
                metrics = {'correlation': correlation}
            
            prediction_series.append(pred_series)
            results['metrics'][target_col] = metrics
            results['summary']['successful_targets'] += 1
            
            # Print metrics
            if is_binary:
                print(f"  Accuracy: {metrics['accuracy']:.3f}")
            else:
                if metrics['correlation'] is not None:
                    print(f"  Correlation: {metrics['correlation']:.3f}")
                    
        except Exception as e:
            print(f"  Error: {str(e)}")
            results['summary']['failed_targets'] += 1
    
    # Concatenate all predictions
    all_predictions = pd.concat([pd.DataFrame({'TWIN_ID': df['TWIN_ID']})] + prediction_series, axis=1)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions
    predictions_path = os.path.join(output_dir, 'xgboost_predictions.csv')
    all_predictions.to_csv(predictions_path, index=False)
    print(f"\nSaved predictions to {predictions_path}")
    
    # Save metrics and hyperparameters
    results_path = os.path.join(output_dir, 'xgboost_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2)
    print(f"Saved results to {results_path}")
    
    # Print summary
    print("\n=== XGBoost Training Summary ===")
    print(f"Total targets: {results['summary']['total_targets']}")
    print(f"Successful: {results['summary']['successful_targets']}")
    print(f"Failed: {results['summary']['failed_targets']}")
    
    # Calculate average metrics
    binary_accuracies = [m['accuracy'] for m in results['metrics'].values() 
                        if 'accuracy' in m]
    numeric_correlations = [m['correlation'] for m in results['metrics'].values() 
                           if m.get('correlation') is not None]
    
    if binary_accuracies:
        print(f"\nAverage binary accuracy: {np.mean(binary_accuracies):.3f}")
    if numeric_correlations:
        print(f"Average numeric correlation: {np.mean(numeric_correlations):.3f}")
    
    return {
        'predictions': all_predictions,
        'results': results
    }


def main():
    parser = argparse.ArgumentParser(
        description='XGBoost prediction for survey answers',
        epilog='''
Examples:
  # Quick test (2 folds, no tuning):
  python %(prog)s --config config.yaml --cv-folds 2 --no-tuning
  
  # Standard run with defaults (3 folds, 10 iterations):
  python %(prog)s --config config.yaml
  
  # Extensive search (5 folds, 30 iterations):
  python %(prog)s --config config.yaml --cv-folds 5 --n-iter 30
  
  # Custom inner CV (3 outer, 5 inner folds, 20 iterations):
  python %(prog)s --config config.yaml --inner-cv-folds 5 --n-iter 20
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration YAML file')
    parser.add_argument('--cv-folds', type=int, default=3,
                        help='Number of outer CV folds (default: 3)')
    parser.add_argument('--inner-cv-folds', type=int, default=3,
                        help='Number of inner CV folds for hyperparameter tuning (default: 3)')
    parser.add_argument('--n-iter', type=int, default=10,
                        help='Number of iterations for random search (default: 10)')
    parser.add_argument('--no-tuning', action='store_true',
                        help='Disable hyperparameter tuning')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get paths from config
    output_dir = config.get('pipeline', {}).get('output_dir', 'ml_output')
    mega_persona_dir = config.get('mega_persona_dir')
    answer_blocks_dir = config.get('answer_blocks_dir')
    
    if not mega_persona_dir or not answer_blocks_dir:
        raise ValueError("mega_persona_dir and answer_blocks_dir must be specified in config")
    
    # Extract data
    training_df, label_df = extract_training_data(mega_persona_dir, answer_blocks_dir)
    
    # Train XGBoost
    output_subdir = os.path.join(output_dir, 'xgboost_output')
    train_xgboost_predictions(
        training_df, 
        label_df,
        output_subdir,
        cv_folds=args.cv_folds,
        inner_cv_folds=args.inner_cv_folds,
        n_iter=args.n_iter,
        use_nested_cv=not args.no_tuning
    )


if __name__ == "__main__":
    main()