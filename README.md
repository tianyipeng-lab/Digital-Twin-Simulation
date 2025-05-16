# Digital Twin Simulation

This repository contains code for simulating digital twins using Large Language Models (LLMs), for the purpose of reproducing the experiments in Twin-2K-500 (paper coming out soon). The project focuses on creating and simulating digital twins based on persona profiles and survey responses.

## Overview

The digital twin simulation system creates virtual representations of individuals based on their survey responses and simulates their behavior in response to new survey questions. The system uses LLMs to generate realistic responses that maintain consistency with the original persona profiles.

## Project Structure

```
.
├── text_simulation/           # Main simulation code
│   ├── configs/              # Configuration files
│   ├── data/                 # Data files
│   ├── text_personas/        # Persona profile data
│   ├── text_questions/       # Survey questions
│   ├── text_simulation_input/ # Combined input files
│   └── text_simulation_output/ # Simulation results
├── scripts/                  # Utility scripts
├── data/                     # Raw data
└── cache/                    # Cached data
```

## Key Components

1. **Persona Processing**
   - `convert_persona_to_text.py`: Converts persona data to text format
   - `batch_convert_personas.py`: Batch processes multiple personas

2. **Question Processing**
   - `convert_question_json_to_text.py`: Converts question data to text format

3. **Simulation**
   - `create_text_simulation_input.py`: Combines personas with questions
   - `run_LLM_simulations.py`: Runs the actual LLM simulations
   - `llm_helper.py`: Helper functions for LLM interactions
   - `postprocess_responses.py`: Processes and analyzes simulation results

## Requirements

- Python 3.11.7 or higher
- Poetry for dependency management

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd digital-twin-simulation
```

2. Install dependencies using Poetry:
```bash
poetry install
```

## Usage

1. Prepare the data:
```bash
python download_dataset.py
```

2. Set the API key for LLMs
```
OPENAI_API_KEY=XXXXX
```

3. Run the pipeline scripts
```bash
./scripts/run_pipeline.sh --max_personas=5 ## run 5 persona simulation
./scripts/run_pipeline.sh  ### run all 2058 persona simulation
```