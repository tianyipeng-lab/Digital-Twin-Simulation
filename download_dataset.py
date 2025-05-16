#!/usr/bin/env python3
"""
Download and process the Twin-2K-500 dataset from Hugging Face.
This script downloads the dataset and organizes it into appropriate directories
with proper file naming conventions.
"""

import os
import json
from datasets import load_dataset
from tqdm.auto import tqdm

def create_directories():
    """Create necessary directories for storing the dataset."""
    directories = [
        'data/mega_persona_json/mega_persona',
        'data/mega_persona_json/answer_blocks',
        'data/mega_persona_summary_text'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def download_wave_split_data():
    """
    Download and process the wave split data from the dataset.
    This includes mega persona data and answer blocks.
    """
    print("Downloading wave split data...")
    dataset = load_dataset("LLM-Digital-Twin/Twin-2K-500", 'wave_split', cache_dir='./cache')
    
    # Extract data from the dataset
    pid = dataset['data']["pid"]
    mega_persona = dataset['data']["wave1_3_persona_json"]
    wave4_Q_wave1_3_A_answer_blocks = dataset['data']["wave4_Q_wave1_3_A"]
    wave4_Q_wave4_A_answer_blocks = dataset['data']["wave4_Q_wave4_A"]
    
    # Save mega_persona data
    print("Saving mega persona data...")
    for idx, persona in zip(pid, mega_persona):
        with open(f'data/mega_persona_json/mega_persona/pid_{idx}_mega_persona.json', 'w') as f:
            f.write(persona)
    
    # Save answer blocks
    print("Saving answer blocks...")
    for idx, answer_block in zip(pid, wave4_Q_wave1_3_A_answer_blocks):
        with open(f'data/mega_persona_json/answer_blocks/pid_{idx}_wave4_Q_wave1_3_A.json', 'w') as f:
            f.write(answer_block)
    
    for idx, answer_block in zip(pid, wave4_Q_wave4_A_answer_blocks):
        with open(f'data/mega_persona_json/answer_blocks/pid_{idx}_wave4_Q_wave4_A.json', 'w') as f:
            f.write(answer_block)

def download_full_persona_data():
    """
    Download and process the full persona data from the dataset.
    This includes persona summaries.
    """
    print("Downloading full persona data...")
    dataset = load_dataset("LLM-Digital-Twin/Twin-2K-500", 'full_persona', cache_dir='./cache')
    
    # Extract data from the dataset
    pid = dataset['data']["pid"]
    persona_summary = dataset['data']["persona_summary"]
    
    # Save persona summaries
    print("Saving persona summaries...")
    for idx, summary in zip(pid, persona_summary):
        with open(f'data/mega_persona_summary_text/pid_{idx}_mega_persona.txt', 'w') as f:
            if summary is not None:
                f.write(summary)  # Write the summary directly without json.dump
            else:
                #print(f"No summary available for pid {idx}")
                pass
def main():
    """Main function to orchestrate the download and processing of the dataset."""
    try:
        # Create necessary directories
        create_directories()
        
        # Download and process wave split data
        download_wave_split_data()
        
        # Download and process full persona data
        download_full_persona_data()
        
        print("Dataset download and processing completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()