# ML Prediction Pipeline

This folder contains scripts for training XGBoost models to predict survey responses and evaluating their accuracy using MAD (Mean Absolute Difference) metrics.

## Files

- `predict_answer_xgboost.py` - Train XGBoost models with nested cross-validation
- `prepare_xgboost_for_mad.py` - Format predictions for MAD evaluation
- `ml_prediction_config.yaml` - Configuration file for the pipeline

## Usage

### Quick Test (Small Sample)

For a quick test with reduced computational requirements:

```bash
# Test with 2 CV folds, 5 iterations
poetry run python ml_prediction/predict_answer_xgboost.py \
    --config ml_prediction/ml_prediction_config.yaml \
    --cv-folds 2 \
    --inner-cv-folds 2 \
    --n-iter 5

# Even quicker: no hyperparameter tuning
poetry run python ml_prediction/predict_answer_xgboost.py \
    --config ml_prediction/ml_prediction_config.yaml \
    --cv-folds 2 \
    --no-tuning
```

### Standard Run

For a standard analysis with reasonable defaults:

```bash
# Default: 3 CV folds, 10 iterations
poetry run python ml_prediction/predict_answer_xgboost.py \
    --config ml_prediction/ml_prediction_config.yaml

# Evaluate predictions with MAD
poetry run python ml_prediction/prepare_xgboost_for_mad.py --run-mad
```

### Full Analysis

For publication-quality results with extensive hyperparameter search:

```bash
# 1. Train XGBoost models
poetry run python ml_prediction/predict_answer_xgboost.py \
    --config ml_prediction/ml_prediction_config.yaml \
    --cv-folds 5 \
    --inner-cv-folds 3 \
    --n-iter 30

# 2. Evaluate predictions with MAD
poetry run python ml_prediction/prepare_xgboost_for_mad.py --run-mad
```

### Limiting Data Size for Testing

To test with a subset of personas, modify `ml_prediction_config.yaml`:

```yaml
ml_prediction:
  max_personas: 100  # Limit to 100 personas for testing
```

## Output

Results will be saved to:
- Predictions: `ml_prediction/output/xgboost_output/`
- MAD evaluation: `ml_prediction/output/mad_evaluation/`

## Command Line Options

### predict_answer_xgboost.py
- `--cv-folds`: Number of outer CV folds (default: 3)
- `--inner-cv-folds`: Number of inner CV folds for hyperparameter tuning (default: 3)
- `--n-iter`: Number of iterations for random search (default: 10)
- `--no-tuning`: Disable hyperparameter tuning for faster runs

### prepare_xgboost_for_mad.py
- `--predictions`: Path to XGBoost predictions CSV (default: auto-detected)
- `--run-mad`: Run MAD evaluation after formatting
- `--mad-output-dir`: Custom output directory for MAD results

## Dependencies

These scripts depend on files in the evaluation folder:
- `evaluation/json2csv.py` - For formatting data
- `evaluation/mad_accuracy_evaluation.py` - For MAD computation and core functions
- `evaluation/column_mapping.csv` - Column name mappings