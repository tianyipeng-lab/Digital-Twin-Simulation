import os
import json
import argparse
from tqdm import tqdm
import re # Import regex for parsing
import copy
from typing import Union, List, Dict, Any

def is_valid_number(value: Any) -> bool:
    """Check if a value is a valid number (integer or float)."""
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False

def is_in_range(value: Union[int, float], min_val: Union[int, float], max_val: Union[int, float]) -> bool:
    """Check if a numeric value is within the specified range."""
    if not is_valid_number(value):
        return False
    num_value = float(value)
    return min_val <= num_value <= max_val

def validate_matrix_response(response: Dict, question: Dict) -> bool:
    """
    Validate Matrix question responses.
    
    Args:
        response: The response to validate
        question: The question dictionary containing validation rules
    
    Returns:
        bool: True if response is valid, False otherwise
    """
    if not isinstance(response, dict):
        return False
    
    # Check required fields
    if "SelectedByPosition" not in response or "SelectedText" not in response:
        return False
    
    # Check that both lists have the same length
    if len(response["SelectedByPosition"]) != len(response["SelectedText"]):
        return False
    
    # Check that all positions are valid numbers
    if not all(isinstance(pos, (int, str)) for pos in response["SelectedByPosition"]):
        return False
    
    # Check that all texts are strings
    if not all(isinstance(text, str) for text in response["SelectedText"]):
        return False
    
    return True

def validate_single_choice_response(response: Dict, question: Dict) -> bool:
    """
    Validate Single Choice question responses.
    
    Args:
        response: The response to validate
        question: The question dictionary containing validation rules
    
    Returns:
        bool: True if response is valid, False otherwise
    """
    if not isinstance(response, dict):
        return False
    
    # Check required fields
    if "SelectedByPosition" not in response or "SelectedText" not in response:
        return False
    
    # Check that position is a valid number (can be string or int)
    if not isinstance(response["SelectedByPosition"], (int, str)):
        return False
    
    # If position is a string, try to convert to int
    if isinstance(response["SelectedByPosition"], str):
        try:
            int(response["SelectedByPosition"])
        except ValueError:
            return False
    
    # Check that text is a string
    if not isinstance(response["SelectedText"], str):
        return False
    
    return True

def validate_slider_response(response: Dict, question: Dict) -> bool:
    """
    Validate Slider question responses.
    
    Args:
        response: The response to validate
        question: The question dictionary containing validation rules
    
    Returns:
        bool: True if response is valid, False otherwise
    """
    if not isinstance(response, dict):
        return False
    
    # Check required field
    if "Values" not in response:
        return False
    
    # Check that Values is a list
    if not isinstance(response["Values"], list):
        return False
    
    # Check that all values are valid numbers
    if not all(is_valid_number(val) for val in response["Values"]):
        return False
    
    # Check numeric constraints if they exist
    if "NumericConstraints" in question:
        constraints = question["NumericConstraints"]
        if "MinValue" in constraints and "MaxValue" in constraints:
            if not all(is_in_range(val, constraints["MinValue"], constraints["MaxValue"]) for val in response["Values"]):
                return False
    
    return True

def validate_text_entry_response(response: Dict, question: Dict) -> bool:
    """
    Validate Text Entry question responses.
    
    Args:
        response: The response to validate
        question: The question dictionary containing validation rules
    
    Returns:
        bool: True if response is valid, False otherwise
    """
    if not isinstance(response, dict):
        return False
    
    # Check required field
    if "Text" not in response:
        return False
    
    # Check that Text is a string
    if not isinstance(response["Text"], str):
        return False
    
    return True

def validate_response(response: Any, question: Dict) -> bool:
    """
    Validate response based on question type.
    
    Args:
        response: The response to validate
        question: The question dictionary containing validation rules
    
    Returns:
        bool: True if response is valid, False otherwise
    """
    if not isinstance(response, dict):
        return False
        
    # Handle both "Question Type" and "QuestionType" formats
    question_type = response.get("QuestionType") or response.get("Question Type")
    answers = response.get("Answers")
    
    if not question_type or not answers:
        return False
    
    validation_functions = {
        "Matrix": validate_matrix_response,
        "Single Choice": validate_single_choice_response,
        "Slider": validate_slider_response,
        "Text Entry": validate_text_entry_response
    }
    
    validation_func = validation_functions.get(question_type)
    if not validation_func:
        return False
    
    return validation_func(answers, question)

def update_question_json_with_response(qid, answer_block_json_path, simulation_response_data, output_dir):
    """
    Loads an original answer block JSON, updates its "Answers" field based on 
    parsed answers from the simulation response_text, and saves it.

    Args:
        answer_block_json_path (str): Path to the original answer block JSON file.
        simulation_response_data (dict): Parsed JSON data from the simulation output file,
                                         containing at least 'response_text'.
        output_dir (str): Directory to save the updated question JSON.
    """
    try:
        with open(answer_block_json_path, 'r', encoding='utf-8') as f:
            answer_block_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Answer block JSON not found: {answer_block_json_path}")
        return False
    except json.JSONDecodeError:
        print(f"Error: Could not decode answer block JSON: {answer_block_json_path}")
        return False

    response_text = simulation_response_data.get("response_text")
    if not response_text:
        print(f"Warning: No response_text found in simulation data for {answer_block_json_path}. Skipping.")
        return False

    # Save raw response text for debugging
    raw_response_text_dir = os.path.join(output_dir, "llm_response_text")
    if not os.path.exists(raw_response_text_dir):
        os.makedirs(raw_response_text_dir)
    with open(os.path.join(raw_response_text_dir, f"{qid}_response_text.txt"), "w", encoding="utf-8") as f:
        f.write(response_text)

    try:
        if "```json" in response_text:
            response_text_json = json.loads(response_text.split("```json")[1].split("```")[0])
        else:
            response_text_json = json.loads(response_text)
    except (json.JSONDecodeError, IndexError) as e:
        print(f"Error parsing response JSON for {qid}: {e}")
        return False

    question_items = []
    count = 0
    failed_responses = 0
    validation_failures = 0

    for block in answer_block_data:
        for question in block['Questions']:
            if question['QuestionType'] != 'DB':
                count += 1
                try:
                    retrieved_response = response_text_json[f"Q{count}"]
                    if retrieved_response is None:
                        failed_responses += 1
                        continue

                    # Validate response based on question type
                    if not validate_response(retrieved_response, question):
                        print(f"Warning: Invalid response for {qid} Q{count} ({question['QuestionType']}): {retrieved_response['Answers']}")
                        validation_failures += 1
                        continue

                    question["Original_Answers"] = copy.deepcopy(question["Answers"])
                    question['Answers'] = retrieved_response["Answers"]
                    if "Reasoning" in retrieved_response:
                        question['LLM_Reasoning'] = retrieved_response["Reasoning"]
                    question_items.append(question)
                except KeyError as e:
                    print(f"Error accessing response for {qid} Q{count}: {e}")
                    failed_responses += 1
                    continue

    if failed_responses > 0 or validation_failures > 0:
        print(f"Warning: {failed_responses} failed responses and {validation_failures} validation failures for {qid}")

    # Export the updated answer block json
    output_path = os.path.join(output_dir, os.path.basename(answer_block_json_path))
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(answer_block_data, f, indent=2)

    if failed_responses > 0 or validation_failures > 0:
        return False
    return True

def postprocess_simulation_outputs_with_pid(persona_id, simulation_output_dir, question_json_base_dir, output_updated_questions_dir):
    """
    Finds existing simulation outputs, matches them with original answer block JSONs,
    and updates the answer block JSONs with the generated responses.
    """
    answer_block_suffix = "_wave4_Q_wave4_A.json"
    answer_block_filename = f"{persona_id}{answer_block_suffix}"
    answer_block_path = os.path.join(question_json_base_dir, answer_block_filename)

    if not os.path.exists(answer_block_path):
        print(f"Warning: Answer block file not found at {answer_block_path} for persona {persona_id}. Skipping.")
        return False

    # 4. Construct response file path (we already confirmed it exists)
    response_filename = f"{persona_id}_response.json"
    response_file_path = os.path.join(simulation_output_dir, persona_id, response_filename)

    # 5. Load the simulation response data
    with open(response_file_path, 'r', encoding='utf-8') as f:
        simulation_response_data = json.load(f)

    # 6. Update the answer block JSON
    success = update_question_json_with_response(
        persona_id,
        answer_block_path,
        simulation_response_data,
        output_updated_questions_dir
    )
    return success

def postprocess_simulation_outputs(simulation_output_dir, question_json_base_dir, output_updated_questions_dir):
    """
    Finds existing simulation outputs, matches them with original answer block JSONs,
    and updates the answer block JSONs with the generated responses.
    """
    if not output_updated_questions_dir:
        print("Error: Output directory must be specified (--output_updated_questions_dir).")
        return

    os.makedirs(output_updated_questions_dir, exist_ok=True)
    print(f"Updated answer block JSONs will be saved to: {output_updated_questions_dir}")

    # 1. Get valid PIDs from the simulation output directory
    valid_persona_ids = []
    try:
        for item in os.listdir(simulation_output_dir):
            item_path = os.path.join(simulation_output_dir, item)
            # Check if it's a directory and looks like a PID
            if os.path.isdir(item_path) and item.startswith("pid_"):
                 # Further check if the expected response file exists within it
                 expected_response_filename = f"{item}_response.json"
                 expected_response_path = os.path.join(item_path, expected_response_filename)
                 if os.path.exists(expected_response_path):
                      valid_persona_ids.append(item)
                 else:
                      print(f"Warning: Directory {item} exists but missing expected response file {expected_response_filename}. Skipping.")
                      
    except FileNotFoundError:
        print(f"Error: Simulation output directory not found: {simulation_output_dir}")
        return

    if not valid_persona_ids:
        print(f"No valid simulation outputs (subdirectories like pid_XXX containing pid_XXX_response.json) found in {simulation_output_dir}")
        return

    print(f"Found {len(valid_persona_ids)} valid simulation outputs to process.")

    successful_updates = 0
    failed_or_skipped = 0

    # 2. Iterate through valid PIDs
    for persona_id in tqdm(valid_persona_ids, desc="Postprocessing responses"):
        try:
            success = postprocess_simulation_outputs_with_pid(persona_id, simulation_output_dir, question_json_base_dir, output_updated_questions_dir)
            if success:
                 successful_updates += 1
            else:
                 # update_question_json_with_response prints specific warnings
                 failed_or_skipped += 1

        except Exception as e:
            print(f"Error processing persona {persona_id}: {e}")
            failed_or_skipped += 1
            continue

    print(f"Postprocessing complete. Successful updates: {successful_updates}, Failed/Skipped: {failed_or_skipped}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Postprocess Gemini simulation responses and update original question JSONs.")
    parser.add_argument("--simulation_output_dir", default="text_simulation_output", 
                        help="Directory containing the simulation output JSON files (e.g., text_simulation_output).")
    parser.add_argument("--question_json_dir", default="../data/mega_persona_json/answer_blocks", 
                        help="Base directory containing the original answer block JSON files (e.g., data/mega_persona_json/answer_blocks).")
    parser.add_argument("--output_updated_questions_dir", default="./text_simulation_output/answer_blocks_llm_imputed", 
                        help="Directory to save the updated question JSON files.")
    
    args = parser.parse_args()
    
    postprocess_simulation_outputs(
        args.simulation_output_dir,
        args.question_json_dir,
        args.output_updated_questions_dir
    ) 