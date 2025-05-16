import json
import os
import argparse
from tqdm import tqdm
from convert_persona_to_text import convert_persona_to_text

def batch_convert_personas(persona_json_dir: str, output_text_dir: str, variant: str = "full") -> None:
    """
    Batch convert persona JSON files to text files.
    
    Args:
        persona_json_dir: Directory containing persona JSON files
        output_text_dir: Directory to save output text files
        variant: Type of conversion ("full", "demographic", "summary", "summary+text", "combine")
    """
    if not os.path.exists(persona_json_dir):
        print(f"Persona JSON directory not found: {persona_json_dir}")
        return
    
    os.makedirs(output_text_dir, exist_ok=True)
    json_files = [f for f in os.listdir(persona_json_dir) if f.endswith('.json') and f.startswith('pid_')]
    
    if not json_files:
        print(f"No persona JSON files found in {persona_json_dir}")
        return
    
    print(f"Found {len(json_files)} persona JSON files to convert.")
    successful = 0
    failed = 0
    
    for json_file in tqdm(json_files, desc="Converting personas to text"):
        input_path = os.path.join(persona_json_dir, json_file)
        output_path = os.path.join(output_text_dir, json_file.replace('.json', '.txt'))
        
        if convert_persona_to_text(input_path, output_path, variant):
            successful += 1
        else:
            failed += 1
            print(f"Failed to convert {json_file}")
    
    print(f"\nConversion complete. Successful: {successful}, Failed: {failed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert persona JSON files to text files.")
    parser.add_argument('--persona_json_dir', required=True, help='Directory containing persona JSON files')
    parser.add_argument('--output_text_dir', required=True, help='Directory to save output text files')
    parser.add_argument('--variant', choices=["full", "demographic", "summary", "summary+text", "combine"],
                      default="full", help="Variant of persona to generate")
    args = parser.parse_args()
    batch_convert_personas(args.persona_json_dir, args.output_text_dir, args.variant) 