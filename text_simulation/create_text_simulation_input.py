import os
import argparse
import re
from tqdm import tqdm

COMBINED_PROMPT_HEADER = """## Persona Profile (This individual's past survey responses):
"""
COMBINED_PROMPT_SEPARATOR = """\n\n---\n## New Survey Question & Instructions (Please respond as the persona described above):
"""

def create_combined_prompts(persona_text_dir, question_prompts_dir, output_combined_prompts_dir):
    """
    Combines persona text files with their corresponding question prompt files to create final LLM prompts.
    Matches persona files with question files based on persona ID (pid).

    Args:
        persona_text_dir (str): Directory containing processed persona text files.
        question_prompts_dir (str): Directory containing LLM question prompt files.
        output_combined_prompts_dir (str): Directory to save the combined prompt files.
    """
    os.makedirs(output_combined_prompts_dir, exist_ok=True)

    # Get all persona files with their PIDs
    persona_files = {}
    for f in os.listdir(persona_text_dir):
        if f.endswith('.txt'):
            pid_match = re.search(r'(pid_\d+)', f)
            if pid_match:
                pid = pid_match.group(1)
                persona_files[pid] = os.path.join(persona_text_dir, f)
            else:
                # Use filename without extension as fallback
                persona_id = os.path.splitext(f)[0].replace("_persona", "")
                persona_files[persona_id] = os.path.join(persona_text_dir, f)

    # Get all question files with their PIDs
    question_files = {}
    for f in os.listdir(question_prompts_dir):
        if f.endswith('.txt'):
            pid_match = re.search(r'(pid_\d+)', f)
            if pid_match:
                pid = pid_match.group(1)
                question_files[pid] = os.path.join(question_prompts_dir, f)
            else:
                # Use filename without extension as fallback
                question_id = os.path.splitext(f)[0]
                question_files[question_id] = os.path.join(question_prompts_dir, f)

    if not persona_files:
        print(f"No persona text files found in {persona_text_dir}")
        return
    if not question_files:
        print(f"No question prompt files found in {question_prompts_dir}")
        return

    # Find matching PIDs
    matching_pids = set(persona_files.keys()) & set(question_files.keys())
    
    if not matching_pids:
        print("No matching persona and question files found.")
        print(f"Persona PIDs: {list(persona_files.keys())}")
        print(f"Question PIDs: {list(question_files.keys())}")
        return

    print(f"Found {len(matching_pids)} matching personas and question files.")
    pbar = tqdm(total=len(matching_pids), desc="Creating combined prompts")

    for pid in matching_pids:
        persona_path = persona_files[pid]
        question_path = question_files[pid]

        try:
            # Read persona content
            with open(persona_path, 'r', encoding='utf-8') as pf:
                persona_content = pf.read()
                
            # Read question content
            with open(question_path, 'r', encoding='utf-8') as qf:
                question_content = qf.read()
            
            # Combine the contents
            combined_content = f"{COMBINED_PROMPT_HEADER}{persona_content}{COMBINED_PROMPT_SEPARATOR}{question_content}"
            
            # Create output file
            output_filename = f"{pid}_prompt.txt"
            output_filepath = os.path.join(output_combined_prompts_dir, output_filename)
            
            with open(output_filepath, 'w', encoding='utf-8') as out_f:
                out_f.write(combined_content)
            
        except Exception as e:
            print(f"Error processing or writing combined prompt for {pid}: {e}")
        finally:
            pbar.update(1)
    
    pbar.close()
    print(f"Combined prompt creation complete. Files saved in {output_combined_prompts_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine persona texts with question prompts for LLM input.")
    parser.add_argument("--persona_text_dir", default="text_simulation/text_personas", help="Directory containing persona text files (output of batch_convert_personas.py).")
    parser.add_argument("--question_prompts_dir", default="text_simulation/text_questions", help="Directory containing question LLM prompt files (output of convert_question_json_to_text.py).")
    parser.add_argument("--output_combined_prompts_dir", default="text_simulation/text_simulation_input", help="Directory to save the final combined LLM prompts.")
    
    args = parser.parse_args()
    
    create_combined_prompts(args.persona_text_dir, args.question_prompts_dir, args.output_combined_prompts_dir) 