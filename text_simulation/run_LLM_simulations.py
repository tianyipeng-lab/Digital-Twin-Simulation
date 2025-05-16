import os
import json
import argparse
import re
import yaml
from tqdm import tqdm
import asyncio
from dotenv import load_dotenv
from postprocess_responses import postprocess_simulation_outputs_with_pid
from llm_helper import LLMConfig, process_prompts_batch
from datetime import datetime
    

load_dotenv()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    return config_data

def get_output_path(base_output_dir, persona_id, question_id):
    persona_output_folder = os.path.join(base_output_dir, persona_id)
    os.makedirs(persona_output_folder, exist_ok=True)
    output_filename = f"{question_id}_response.json"
    return os.path.join(persona_output_folder, output_filename)

# This is the callback function that will be passed to llm_helper
# It must be a regular synchronous function because it performs file I/O and calls another sync function.
# llm_helper will run it in a thread using asyncio.to_thread.
def save_and_verify_callback(prompt_id: str, llm_response_data: dict, original_prompt_text: str, **kwargs) -> bool:
    """
    Saves the LLM response and then verifies it.
    Expected kwargs: base_output_dir, question_json_base_dir, output_updated_questions_dir_for_verify
    """
    base_output_dir = kwargs.get("base_output_dir")
    question_json_base_dir = kwargs.get("question_json_base_dir")
    output_updated_questions_dir_for_verify = kwargs.get("output_updated_questions_dir_for_verify")

    if not all([base_output_dir, question_json_base_dir, output_updated_questions_dir_for_verify]):
        print(f"Error for {prompt_id}: Missing critical path arguments in verification_callback_args.")
        return False # Critical configuration error

    persona_id = prompt_id # Assuming prompt_id is the persona_id
    question_id = persona_id # Assuming question_id is also persona_id for this context

    output_path = get_output_path(base_output_dir, persona_id, question_id)

    # Save LLM response first
    output_json_data = {
        "persona_id": persona_id,
        "question_id": question_id,
        "prompt_text": original_prompt_text,
        "response_text": llm_response_data.get("response_text", ""), # Handle case where response_text might be missing
        "usage_details": llm_response_data.get("usage_details", {}),
        "llm_call_error": llm_response_data.get("error") # Include any error from the LLM call itself
    }
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_json_data, f, indent=2)
    except Exception as e:
        print(f"Error writing output file {output_path} for {prompt_id}: {e}")
        return False # If we can't save, verification is moot and it's a failure of this step

    # If the LLM call itself had an error, no point in verifying (already saved error in json)
    if "error" in llm_response_data and llm_response_data["error"]:
        # print(f"Skipping verification for {prompt_id} due to LLM call error: {llm_response_data['error']}")
        return False # Treat as overall failure for this attempt if LLM errored

    # Now, verify the output that was just saved
    try:
        is_verified = postprocess_simulation_outputs_with_pid(
            persona_id,
            base_output_dir, # This is the directory where individual persona folders are, used by postprocess
            question_json_base_dir,
            output_updated_questions_dir_for_verify
        )
        if not is_verified:
            # print(f"Verification explicitly failed for {prompt_id} by postprocess_simulation_outputs_with_pid.")
            pass # llm_helper will retry based on this False return
        return is_verified
    except Exception as e:
        print(f"Error during verification call for persona {prompt_id}: {e}")
        return False


async def run_simulations(prompts_root_dir, base_output_dir, llm_config_params, provider, num_workers, max_retries_for_sequence, force_regenerate, max_personas=None):
    question_json_base_dir_for_verify = "./data/mega_persona_json/answer_blocks" 
    output_updated_questions_dir_for_verify = os.path.join(base_output_dir, "answer_blocks_llm_imputed")

    # Prepare arguments for the verification callback
    verification_args = {
        "base_output_dir": base_output_dir,
        "question_json_base_dir": question_json_base_dir_for_verify,
        "output_updated_questions_dir_for_verify": output_updated_questions_dir_for_verify
    }

    llm_config = LLMConfig(
        model_name=llm_config_params['model_name'],
        temperature=llm_config_params.get('temperature', 0.7),
        max_tokens=llm_config_params.get('max_tokens'),
        system_instruction=llm_config_params.get('system_instruction'),
        max_retries=max_retries_for_sequence, # Max retries for the [LLM call + save + verification] sequence
        max_concurrent_requests=num_workers,
        verification_callback=save_and_verify_callback,
        verification_callback_args=verification_args
    )

    all_prompt_files_info = []
    try:
        prompt_files_fs = sorted([f for f in os.listdir(prompts_root_dir) if f.endswith('_prompt.txt')])
    except FileNotFoundError:
        print(f"Error: Prompts root directory not found: {prompts_root_dir}")
        return

    if max_personas is not None and max_personas > 0:
        prompt_files_fs = prompt_files_fs[:max_personas]
        print(f"Limiting processing to {max_personas} prompt files")

    for prompt_filename in prompt_files_fs:
        persona_match = re.search(r'(pid_\d+)', prompt_filename)
        if persona_match:
            persona_id = persona_match.group(1)
            full_prompt_path = os.path.join(prompts_root_dir, prompt_filename)
            all_prompt_files_info.append({'persona_id': persona_id, 'file_path': full_prompt_path})

    if not all_prompt_files_info:
        print(f"No prompt files (ending with '_prompt.txt') found in {prompts_root_dir}")
        return

    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    if not os.path.exists(output_updated_questions_dir_for_verify):
        os.makedirs(output_updated_questions_dir_for_verify)

    prompts_to_process_for_llm = []
    skipped_due_to_existing_verified_count = 0

    for info in all_prompt_files_info:
        p_id, f_path = info['persona_id'], info['file_path']
        # Check if output *already* exists and is *already* verified if not force_regenerate
        # The save_and_verify_callback will handle saving and verifying for *new* calls.
        output_json_path = get_output_path(base_output_dir, p_id, p_id) # p_id is also question_id here
        if os.path.exists(output_json_path) and not force_regenerate:
            # We need to check if this existing output is actually valid according to verification logic
            # This call to verify_and_process_output is for *pre-existing* files.
            if postprocess_simulation_outputs_with_pid(p_id, base_output_dir, question_json_base_dir_for_verify, output_updated_questions_dir_for_verify):
                skipped_due_to_existing_verified_count += 1
                continue
        
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                prompt_content = f.read()
            # (prompt_id_for_helper, prompt_text_for_helper)
            prompts_to_process_for_llm.append((p_id, prompt_content)) 
        except Exception as e:
            print(f"Error reading prompt file {f_path} for {p_id}: {e}")

    if skipped_due_to_existing_verified_count > 0:
        print(f"Skipped {skipped_due_to_existing_verified_count} files as their output already exists and is verified.")
    
    if not prompts_to_process_for_llm:
        print("No new files require LLM processing or re-processing.")
        if skipped_due_to_existing_verified_count == 0: 
             print("No prompt files found or processed in this run.")
        return

    print(f"Found {len(prompts_to_process_for_llm)} prompts to process with LLM and verify (up to {llm_config.max_concurrent_requests} concurrent requests).")

    # process_prompts_batch will handle LLM calls, saving (via callback), verification (via callback), and retries for the sequence.
    final_results = await process_prompts_batch(
        prompts_to_process_for_llm, 
        llm_config, 
        provider,
        desc=f"{provider.capitalize()} LLM calls & Verification"
    )

    successful_final_count = 0
    failed_final_count = 0
    
    for prompt_id, result_data in final_results.items():
        if "error" in result_data and result_data["error"]:
            failed_final_count += 1
            print(f"FINAL FAILURE for {prompt_id}: {result_data['error']}")
            # The error (and potentially last successful LLM response if verification was the failure point) 
            # should have been saved by the callback or the helper's error handling.
            # We could write a more specific error log here if needed.
            error_file_path = get_output_path(base_output_dir, prompt_id, prompt_id).replace(".json", "_final_error.txt")
            with open(error_file_path, 'w') as ef:
                 ef.write(f"Final processing failed for {prompt_id}. Details: {json.dumps(result_data, indent=2)}")
        else:
            successful_final_count += 1
            # Output and verification were handled by the callback, so here we just count success.
            # print(f"FINAL SUCCESS for {prompt_id}") # Optional: for verbose logging

    print(f"\nProcessing run complete.")
    print(f"  Successfully processed and verified (new or reprocessed): {successful_final_count}")
    print(f"  Failed permanently after all retries: {failed_final_count}")
    if skipped_due_to_existing_verified_count > 0:
        print(f"  Skipped (already existing and verified): {skipped_due_to_existing_verified_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM simulations with integrated verification and retries.")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--max_personas", type=int, help="Maximum number of personas to process")
    args = parser.parse_args()

    config_values = load_config(args.config)
    
    prompts_root_dir = config_values.get('input_folder_dir', './text_simulation_input_with_context')
    base_output_dir = config_values.get('output_folder_dir', './text_simulation_output_with_context')
    prompts_root_dir = os.path.join("./text_simulation", prompts_root_dir)
    base_output_dir = os.path.join("./text_simulation", base_output_dir)
    provider = config_values.get('provider', 'gemini')
    num_workers = config_values.get('num_workers', 5)
    max_retries_for_sequence = config_values.get('max_retries', 3) # Renamed for clarity: retries for the whole sequence
    force_regenerate = config_values.get('force_regenerate', False)
    max_personas = config_values.get('max_personas', None)
    if args.max_personas:
        max_personas = args.max_personas
        if max_personas == -1:
            max_personas = None
    
    llm_specific_config_dict = config_values.get('llm_config', {})
    if 'model_name' not in llm_specific_config_dict:
        llm_specific_config_dict['model_name'] = config_values.get('model_name')
        if not llm_specific_config_dict['model_name']:
             raise ValueError("LLM model_name must be specified in the config file.")

    if not prompts_root_dir or not base_output_dir:
        raise ValueError("prompts_root_dir and base_output_dir must be specified.")

    print(f"Starting LLM simulations from: {args.config} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Provider: {provider}, Model: {llm_specific_config_dict['model_name']}")
    print(f"Input prompts directory: {prompts_root_dir}")
    print(f"Base output directory: {base_output_dir}")
    print(f"Number of concurrent requests (num_workers in config): {num_workers}")
    print(f"Force regenerate: {force_regenerate}")
    if max_personas:
        print(f"Max personas to process: {max_personas}")

    asyncio.run(run_simulations(
        prompts_root_dir=prompts_root_dir,
        base_output_dir=base_output_dir,
        llm_config_params=llm_specific_config_dict,
        provider=provider,
        num_workers=num_workers,
        max_retries_for_sequence=max_retries_for_sequence,
        force_regenerate=force_regenerate,
        max_personas=max_personas
    )) 