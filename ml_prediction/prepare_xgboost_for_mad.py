#!/usr/bin/env python3
"""
Prepare XGBoost predictions for MAD accuracy evaluation.

This script takes the output from predict_answer_xgboost.py and formats it
to match the expected structure for mad_accuracy_evaluation.py.

The main steps are:
1. Load XGBoost predictions CSV
2. Extract wave1_3 ground truth from answer_blocks JSON files
3. Format both using JSONToCSVConverter to match benchmark structure
4. Save formatted CSVs in the expected directory structure
5. Run MAD evaluation
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional

# Import the necessary modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'evaluation'))
from evaluation.json2csv import JSONToCSVConverter, AnswerExtractor, ExtractionMode

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_config_for_formatting(benchmark_csv: str, column_mapping: str) -> dict:
    """Create a minimal config for JSONToCSVConverter."""
    return {
        'benchmark_csv': benchmark_csv,
        'column_mapping': column_mapping,
        'waves': {}  # Empty waves section to pass validation
    }


def extract_wave1_3_answers(answer_blocks_dir: str, twin_ids: List[str]) -> pd.DataFrame:
    """Extract wave1_3 answers from JSON files."""
    logger.info(f"Extracting wave1_3 data for {len(twin_ids)} participants...")
    
    extractor = AnswerExtractor(ExtractionMode.NUMERIC)
    all_answers = []
    
    for twin_id in twin_ids:
        # Look for wave1_3 answer file
        answer_file = os.path.join(answer_blocks_dir, f"pid_{twin_id}_wave4_Q_wave1_3_A.json")
        
        if os.path.exists(answer_file):
            try:
                answers = extractor.extract_from_file(answer_file)
                if answers:
                    # Ensure TWIN_ID is set
                    answers['TWIN_ID'] = twin_id
                    all_answers.append(answers)
            except Exception as e:
                logger.warning(f"Error processing {answer_file}: {e}")
        else:
            logger.debug(f"No wave1_3 file found for TWIN_ID {twin_id}")
    
    if not all_answers:
        logger.error("No wave1_3 answers extracted")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_answers)
    logger.info(f"Extracted {len(df)} wave1_3 records with {len(df.columns)} columns")
    return df


def extract_wave4_answers(answer_blocks_dir: str, twin_ids: List[str]) -> pd.DataFrame:
    """Extract wave4 answers from JSON files."""
    logger.info(f"Extracting wave4 data for {len(twin_ids)} participants...")
    
    extractor = AnswerExtractor(ExtractionMode.NUMERIC)
    all_answers = []
    
    for twin_id in twin_ids:
        # Look for wave4 answer file
        answer_file = os.path.join(answer_blocks_dir, f"pid_{twin_id}_wave4_Q_wave4_A.json")
        
        if os.path.exists(answer_file):
            try:
                answers = extractor.extract_from_file(answer_file)
                if answers:
                    # Ensure TWIN_ID is set
                    answers['TWIN_ID'] = twin_id
                    all_answers.append(answers)
            except Exception as e:
                logger.warning(f"Error processing {answer_file}: {e}")
        else:
            logger.debug(f"No wave4 file found for TWIN_ID {twin_id}")
    
    if not all_answers:
        logger.error("No wave4 answers extracted")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_answers)
    logger.info(f"Extracted {len(df)} wave4 records with {len(df.columns)} columns")
    return df


def save_formatted_csv(df: pd.DataFrame, output_path: str, description: str = ""):
    """Save DataFrame with description row (MAD evaluation format)."""
    import csv
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    headers = list(df.columns)
    description_row = [""] * len(headers)  # Empty descriptions for now
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(description_row)
        for _, data_row in df.iterrows():
            # Convert NaN to empty string
            processed_row = ["" if pd.isna(val) or (isinstance(val, float) and np.isnan(val)) else val for val in data_row]
            writer.writerow(processed_row)
    
    logger.info(f"Saved {description} to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare XGBoost predictions for MAD accuracy evaluation'
    )
    
    # Input/output arguments
    parser.add_argument('--predictions', type=str, 
                        default='ml_prediction/output/xgboost_output/xgboost_predictions.csv',
                        help='Path to XGBoost predictions CSV')
    parser.add_argument('--answer-blocks-dir', type=str,
                        default='data/mega_persona_json/answer_blocks',
                        help='Directory containing answer_blocks JSON files')
    parser.add_argument('--output-dir', type=str,
                        default='ml_prediction/output/mad_formatted',
                        help='Output directory for formatted files')
    
    # Benchmark configuration
    parser.add_argument('--benchmark-csv', type=str,
                        default='data/wave_csv/wave_4_numbers_anonymized.csv',
                        help='Benchmark CSV for column mapping')
    parser.add_argument('--column-mapping', type=str,
                        default='evaluation/column_mapping.csv',
                        help='Column mapping file')
    
    # MAD evaluation arguments
    parser.add_argument('--run-mad', action='store_true',
                        help='Run MAD evaluation after formatting')
    parser.add_argument('--mad-output-dir', type=str,
                        default='ml_prediction/output/mad_evaluation',
                        help='Output directory for MAD evaluation results')
    
    args = parser.parse_args()
    
    # Step 1: Load XGBoost predictions
    logger.info(f"Loading XGBoost predictions from: {args.predictions}")
    if not os.path.exists(args.predictions):
        logger.error(f"Predictions file not found: {args.predictions}")
        return 1
    
    predictions_df = pd.read_csv(args.predictions)
    logger.info(f"Loaded predictions: {len(predictions_df)} rows, {len(predictions_df.columns)} columns")
    
    # Get TWIN_IDs
    if 'TWIN_ID' not in predictions_df.columns:
        logger.error("Predictions CSV must have TWIN_ID column")
        return 1
    
    twin_ids = predictions_df['TWIN_ID'].astype(str).tolist()
    
    # Step 2: Extract wave1_3 ground truth
    ground_truth_df = extract_wave1_3_answers(args.answer_blocks_dir, twin_ids)
    if ground_truth_df.empty:
        logger.error("Failed to extract wave1_3 ground truth")
        return 1
    
    # Step 3: Extract wave4 data
    wave4_df = extract_wave4_answers(args.answer_blocks_dir, twin_ids)
    if wave4_df.empty:
        logger.error("Failed to extract wave4 data")
        return 1
    
    # Step 4: Format using JSONToCSVConverter
    logger.info("Formatting data to match benchmark structure...")
    
    # Create converter with minimal config
    config = create_config_for_formatting(args.benchmark_csv, args.column_mapping)
    converter = JSONToCSVConverter(config)
    
    # Format predictions (XGBoost output)
    logger.info("Formatting XGBoost predictions...")
    predictions_formatted = converter.format_to_benchmark(
        predictions_df,
        args.benchmark_csv,
        args.column_mapping
    )
    
    # Format ground truth (wave1_3)
    logger.info("Formatting wave1_3 ground truth...")
    ground_truth_formatted = converter.format_to_benchmark(
        ground_truth_df,
        args.benchmark_csv,
        args.column_mapping
    )
    
    # Format wave4 data
    logger.info("Formatting wave4 data...")
    wave4_formatted = converter.format_to_benchmark(
        wave4_df,
        args.benchmark_csv,
        args.column_mapping
    )
    
    # Step 5: Create output directory structure
    csv_formatted_dir = os.path.join(args.output_dir, 'csv_formatted')
    os.makedirs(csv_formatted_dir, exist_ok=True)
    
    # Save formatted files with expected names for MAD evaluation
    pred_path = os.path.join(csv_formatted_dir, 'responses_llm_imputed_formatted.csv')
    truth_path = os.path.join(csv_formatted_dir, 'responses_wave1_3_formatted.csv')
    wave4_path = os.path.join(csv_formatted_dir, 'responses_wave4_formatted.csv')
    
    save_formatted_csv(predictions_formatted, pred_path, "XGBoost predictions (as LLM)")
    save_formatted_csv(ground_truth_formatted, truth_path, "wave1_3 ground truth")
    save_formatted_csv(wave4_formatted, wave4_path, "wave4 test-retest data")
    
    logger.info(f"\nFormatted files saved to: {csv_formatted_dir}")
    logger.info("Files created:")
    logger.info(f"  - {pred_path}")
    logger.info(f"  - {truth_path}")
    logger.info(f"  - {wave4_path}")
    
    # Step 6: Run MAD evaluation if requested
    if args.run_mad:
        logger.info("\nRunning MAD accuracy evaluation...")
        
        from evaluation.mad_accuracy_evaluation import compute_mad_summary
        
        output_excel = os.path.join(args.mad_output_dir, 'xgboost_mad_accuracy.xlsx')
        output_fig = os.path.join(args.mad_output_dir, 'xgboost_accuracy_dist.png')
        
        try:
            compute_mad_summary(
                csv_dir=csv_formatted_dir,
                output_excel_path=output_excel,
                output_fig_path=output_fig,
                fig_title="XGBoost vs Wave1_3 Task-Level Accuracy"
            )
            
            logger.info(f"\nMAD evaluation completed!")
            logger.info(f"Results saved to:")
            logger.info(f"  - Excel report: {output_excel}")
            logger.info(f"  - Accuracy plot: {output_fig}")
            
        except Exception as e:
            logger.error(f"MAD evaluation failed: {e}")
            return 1
    
    logger.info("\n✓ Preparation complete!")
    
    # Print next steps
    if not args.run_mad:
        logger.info("\nTo run MAD evaluation, use:")
        logger.info(f"poetry run python evaluation/mad_accuracy_evaluation.py --csv-dir {csv_formatted_dir} --output-dir {args.mad_output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())