# Twin-2K-500: Digital Twin Simulation with LLMs

This repository contains code for the experiments described in the paper ["Twin-2K-500: A dataset for building digital twins of over 2,000 people based on their answers to over 500 questions."](https://arxiv.org/abs/2505.17479). The project studies creating and simulating digital twins from persona profiles and survey responses using large language models. 

- **Dataset:** [LLM-Digital-Twin/Twin-2K-500 on HuggingFace](https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500)

## Citation

```
@article{toubia2025twin2k500,
  title     = {Twin-2K-500: A dataset for building digital twins of over 2,000 people based on their answers to over 500 questions},
  author    = {Toubia, Olivier and Gui, George Z. and Peng, Tianyi and Merlau, Daniel J. and Li, Ang and Chen, Haozhe},
  journal   = {arXiv preprint arXiv:2505.17479},
  year      = {2025}
}
```

## Overview

The digital twin simulation system creates virtual representations of individuals based on their survey responses and simulates their behavior in response to new survey questions. The system uses LLMs to generate realistic responses that maintain consistency with the original persona profiles.

## Project Structure

```
.
├── text_simulation/           # Main simulation code
│   ├── configs/              # Configuration files
│   ├── text_personas/        # Persona profile data
│   ├── text_questions/       # Survey questions
│   ├── text_simulation_input/ # Combined input files
│   └── text_simulation_output/ # Simulation results
├── evaluation/                # Evaluation folder  
├── notebooks/                 # Demo notebooks
│   ├── demo_simple_simulation.ipynb    # Quick start: simulate responses to new questions
│   └── demo_full_pipeline.ipynb        # Complete pipeline with evaluation (alternative to shell scripts)
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

### Quick Start: Simulate Responses to New Questions

For a quick introduction to digital twin simulation, try our interactive demo notebook:

```bash
jupyter notebook notebooks/demo_simple_simulation.ipynb
```

This notebook demonstrates:
- Loading persona summaries directly from Hugging Face dataset (no setup required!)
- Creating custom survey questions
- Simulating responses using GPT-4.1 mini
- Running batch simulations for multiple personas
- Automatic package installation and API key configuration
- Works seamlessly in both local environments and Google Colab

Perfect for researchers who want to quickly test new survey questions on digital twins without complex setup.

### Full Pipeline Demo (Interactive Alternative)

For those who prefer Jupyter notebooks over shell scripts, we provide a complete pipeline walkthrough:

```bash
jupyter notebook notebooks/demo_full_pipeline.ipynb
```

This notebook covers the entire workflow from data preparation to evaluation, making it an excellent alternative to the shell script approach described below.

### Full Pipeline (Command Line)

To run the complete digital twin simulation pipeline:

1.  **Prepare the Data**:
    First, download the necessary dataset by executing the following command:
    ```bash
    poetry run python download_dataset.py
    ```

2.  **Configure API Access**:
    Set the `OPENAI_API_KEY` environment variable to enable LLM interactions. Create a file named `.env` in the project's root directory and add your API key as follows:
    ```
    OPENAI_API_KEY=your_actual_api_key_here
    ```
    *Replace `your_actual_api_key_here` with your valid OpenAI API key.*

3.  **Run the Simulation Pipeline**:
    Execute the main simulation pipeline using the provided shell scripts. You can run a small test with a limited number of personas or simulate all available personas.

    *   For a small test run (e.g., 5 personas):
        ```bash
        ./scripts/run_pipeline.sh --max_personas=5
        ```
    *   To run the simulation for all 2058 personas:
        ```bash
        ./scripts/run_pipeline.sh
        ```

4.  **Evaluate the Results**:
    After running the simulations, evaluate the results using:
    ```bash
    ./scripts/run_evaluation_pipeline.sh
    ```
