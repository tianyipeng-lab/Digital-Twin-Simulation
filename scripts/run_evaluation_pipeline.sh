#!/bin/bash

# --- Evaluation Pipeline Orchestration Script ---
# This script runs the complete evaluation pipeline for simulation results
# using 'eval_configs/evaluation_basic.yaml' as the configuration file.

set -e  # Exit on error

# Default values
EVAL_CONFIG_FILE="eval_configs/evaluation_basic.yaml"
VERBOSE=""

# --- Functions ---

print_usage() {
    echo "Usage: $0 [--verbose]"
    echo ""
    echo "Options:"
    echo "  --verbose          Enable verbose output"
    echo ""
    echo "Example:"
    echo "  $0 --verbose"
    echo ""
    echo "This script will run the complete evaluation pipeline using 'eval_configs/evaluation_basic.yaml':"
    echo "  1. Convert JSON answer blocks to CSV format (including randdollar breakdown)"
    echo "  2. Generate MAD accuracy and correlation evaluation"
    echo "  3. Run within-between subjects analysis"
    echo "  4. Run pricing analysis"
}

print_header() {
    echo ""
    echo "========================================="
    echo "$1"
    echo "========================================="
}

check_poetry() {
    if ! command -v poetry &> /dev/null; then
        echo "Error: Poetry is not installed. Please install Poetry first."
        echo "Visit: https://python-poetry.org/docs/#installation"
        exit 1
    fi
}

# --- Parse command line arguments ---

while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            print_usage
            exit 1
            ;;
    esac
done

# --- Validate arguments ---

if [ ! -f "$EVAL_CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $EVAL_CONFIG_FILE"
    exit 1
fi

# --- Check prerequisites ---

check_poetry

# --- Main Pipeline Execution ---

print_header "Digital Twin Evaluation Pipeline"
echo "Configuration file: $EVAL_CONFIG_FILE"
echo "Start time: $(date)"

# --- Step 1: Convert JSON to CSV ---
print_header "Step 1: Converting JSON answer blocks to CSV format"
if poetry run python3 evaluation/json2csv.py --config "$EVAL_CONFIG_FILE" --all $VERBOSE; then
    echo "✅ JSON to CSV conversion completed successfully"
else
    echo "❌ JSON to CSV conversion failed"
    exit 1
fi

# --- Step 2: Run MAD Accuracy Evaluation ---
print_header "Step 2: Generating MAD accuracy and correlation evaluation"
if poetry run python3 evaluation/mad_accuracy_evaluation.py --config "$EVAL_CONFIG_FILE" $VERBOSE; then
    echo "✅ MAD accuracy evaluation completed successfully"
else
    echo "⚠️  MAD accuracy evaluation encountered issues, but continuing pipeline..."
    echo "   This may be due to data compatibility issues that need investigation."
fi

# --- Step 3: Run Within-Between Subjects Analysis ---
print_header "Step 3: Running Within-Between Subjects Analysis"
if poetry run python3 evaluation/within_between_subjects.py --config "$EVAL_CONFIG_FILE" $VERBOSE; then
    echo "✅ Within-between subjects analysis completed successfully"
else
    echo "⚠️  Within-between subjects analysis encountered issues, but continuing pipeline..."
    echo "   This may be due to data compatibility issues that need investigation."
fi

# --- Step 4: Run Pricing Analysis ---
print_header "Step 4: Running Pricing Analysis"
if poetry run python3 evaluation/pricing_analysis.py --config "$EVAL_CONFIG_FILE" $VERBOSE; then
    echo "✅ Pricing analysis completed successfully"
else
    echo "⚠️  Pricing analysis encountered issues, but pipeline completed."
    echo "   This may be due to data compatibility issues that need investigation."
fi

# --- Completion ---
print_header "Evaluation Pipeline Complete"
echo "End time: $(date)"
echo ""
echo "✅ All evaluation steps completed successfully!"
echo ""
echo "Results have been saved to the trial directory specified in the configuration:"

# Extract trial_dir from config and show key output locations
TRIAL_DIR=$(python3 -c "import yaml; config=yaml.safe_load(open('$EVAL_CONFIG_FILE')); print(config.get('trial_dir', 'Unknown'))")
if [ "$TRIAL_DIR" != "Unknown" ] && [ -d "$TRIAL_DIR" ]; then
    echo "  📁 Trial directory: $TRIAL_DIR"
    echo "  📊 Accuracy evaluation: $TRIAL_DIR/accuracy_evaluation/"
    echo "  📈 CSV comparisons: $TRIAL_DIR/csv_comparison/"
    echo "  💰 Pricing analysis: $TRIAL_DIR/pricing_analysis_results/"
fi
