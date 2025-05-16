#!/bin/bash

# Exit on error
set -e

# Default value for max_personas
DEFAULT_MAX_PERSONAS=-1

# Check if max_personas argument is provided
if [ $# -eq 0 ]; then
    echo "No max_personas provided, using default value: $DEFAULT_MAX_PERSONAS"
    MAX_PERSONAS=$DEFAULT_MAX_PERSONAS
else
    # Support both '5' and 'max_personas=5' as input
    if [[ $1 == max_personas=* ]]; then
        MAX_PERSONAS="${1#max_personas=}"
    else
        MAX_PERSONAS="$1"
    fi
fi

# Update the max_personas in the config file
sed -i '' "s/max_personas: .*/max_personas: $MAX_PERSONAS  # Set to $MAX_PERSONAS for testing/" text_simulation/configs/openai_config.yaml

# Create necessary directories
mkdir -p text_simulation/data/mega_persona_text

# Run the pipeline steps
echo "Step 1: Converting personas..."
poetry run python text_simulation/batch_convert_personas.py \
    --persona_json_dir data/mega_persona_json/mega_persona \
    --output_text_dir text_simulation/data/mega_persona_text \
    --variant full

echo "Step 2: Converting question JSON to text..."
poetry run python text_simulation/convert_question_json_to_text.py

echo "Step 3: Creating text simulation input..."
poetry run python text_simulation/create_text_simulation_input.py

echo "Step 4: Running LLM simulation..."
poetry run python text_simulation/run_LLM_simulations.py --config text_simulation/configs/openai_config.yaml --max_personas $MAX_PERSONAS

echo "Pipeline completed!" 