import json
import os
import re
import argparse
from pathlib import Path

def strip_html(text: any) -> str:
    """Strip HTML tags from text and normalize whitespace. Handles non-string inputs.
    """
    if text is None:
        return "" # Return empty string if input is None
    if not isinstance(text, str):
        text = str(text) # Convert to string if not already a string
    
    text = re.sub(r'<[^>]*>', ' ', text)
    text = text.replace('&nbsp;', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def _format_question_text_Matrix(question: dict, with_answers: bool = False) -> str:
    """Formats a single question with Matrix type into a readable string."""
    columns = question.get("Columns", [])
    answers = question.get("Answers", {})
    output_options = []
    output_options.append("Question Type: Matrix\n")
    if columns:
        output_options.append("Options:\n")
        for i, column_text in enumerate(columns, 1):
            output_options.append(f"  {i} = {strip_html(column_text)}\n")
        
        rows = question.get("Rows", [])
        selected_texts = answers.get("SelectedText", [])
        selected_positions = answers.get("SelectedByPosition", [])

        for i, row_text_content in enumerate(rows):
            row_display_id = str(i + 1)
            output_options.append(f"{row_display_id}. {strip_html(row_text_content)}\n")
            
            answer_detail = "[No Answer Provided]"
            if i < len(selected_positions) and i < len(selected_texts):
                answer_detail = f"{selected_positions[i]} - {strip_html(selected_texts[i])}"
            if with_answers:
                output_options.append(f"Answer: {answer_detail}\n")
            else:
                output_options.append(f"Answer: [Masked]\n")
        output_options.append("\n")

    return ''.join(output_options)


def _format_question_text_MC(question: dict, with_answers: bool = False) -> str:
    """Formats a single question with MC type into a readable string."""
    options = question.get("Options", [])
    answers = question.get("Answers", {})
    output_options = []
    selector = question.get("Settings", {}).get("Selector")
    if selector == "MAVR" or selector == "MAHR":
        output_options.append("Question Type: Multiple Choice\n")
    elif selector == "SAVR" or selector == "SAHR":
        output_options.append("Question Type: Single Choice\n")
    if options:
        output_options.append("Options:\n")
        for i, option_text in enumerate(options, 1):
            output_options.append(f"  {i} - {strip_html(option_text)}\n")
        
        selected_positions = answers.get("SelectedByPosition", [])
        selected_texts = answers.get("SelectedText", [])
        if with_answers:
            if selected_positions is not None:
                if (selector == "SAVR" or selector == "SAHR"):
                    selected_positions = [selected_positions]
                    selected_texts = [selected_texts]
                for i, selected_position in enumerate(selected_positions):
                    if i < len(selected_texts):
                        output_options.append(f"Answer: {selected_position} - {strip_html(selected_texts[i])}\n")
            else:
                output_options.append("Answer: [No Answer Provided]\n")
        else:
            output_options.append("Answer: [Masked]\n")

    output_options.append("\n")
    return ''.join(output_options)

def _format_question_text_TE(question: dict, with_answers: bool = False) -> str:
    """Formats a single question text into a readable string."""
    settings = question.get("Settings", {})
    answers = question.get("Answers", {})
    output_options = []

    selector = settings.get("Selector")
    if selector == "FORM":
        output_options.append("Question Type: Text Entry (Form)\n")
    else:
        output_options.append("Question Type: Text Entry\n")

    if selector == "FORM":
        form_rows = question.get("Rows", [])
        form_answers_text = answers.get("Text", []) # This is a list of dicts
        
        # Create a lookup for answers for easier access
        answer_lookup = {}
        for ans_item in form_answers_text:
            if isinstance(ans_item, dict):
                answer_lookup.update(ans_item)

        for i, row_label in enumerate(form_rows):
            clean_row_label = strip_html(row_label)
            answer_value = strip_html(str(answer_lookup.get(row_label, "[No Answer Provided]")))
            if with_answers:
                output_options.append(f"{clean_row_label}: {answer_value}\n")
            else:
                output_options.append(f"{clean_row_label}: [Masked]\n")

    elif selector == "SL" or selector == "ML": # Single Line or Multi Line
        text_answer = answers.get("Text")
        if with_answers:
            if text_answer is not None:
                output_options.append(f"Answer: {strip_html(str(text_answer))}\n")
            else:
                output_options.append("Answer: [No Answer Provided]\n")
        else:
            output_options.append("Answer: [Masked]\n")
    output_options.append("\n")
    return ''.join(output_options)

def _format_question_text_Slider(question: dict, with_answers: bool = False) -> str:
    """Formats a single question text into a readable string."""
    answers = question.get("Answers", {})
    output_options = []
    values = answers.get("Values")
    output_options.append("Question Type: Slider\n")
    if values:
        statements = question.get("Statements")
        for i, value_item in enumerate(values):
            statement_text_content = statements[i]
            stmt_display_id = str(i + 1)
            if statement_text_content == "":
                statement_text_content = "[No Statement Needed]"
            output_options.append(f"{stmt_display_id}. {strip_html(statement_text_content)}\n")
            if with_answers:
                output_options.append(f"Answer: {strip_html(str(value_item))}\n")
            else:
                output_options.append(f"Answer: [Masked]\n")
    else:
        output_options.append("Answer: [No Answer Provided]\n")
    output_options.append("\n")
    return ''.join(output_options)
    
def _format_question_text_DB(question: dict, with_answers: bool = False) -> str:
    """Formats a single question text into a readable string."""
    return "[Descriptive Information]\n\n"

def format_question_text(question: dict, with_answers: bool = False) -> str:
    """Formats a single question text into a readable string."""
    question_text = strip_html(question.get('QuestionText', ''))
    if question_text is None:
        question_text = ""

    question_type = question.get("QuestionType")

    if question_type == "Matrix":
        output_options = _format_question_text_Matrix(question, with_answers)
    elif question_type == "MC": # Multiple Choice
        output_options = _format_question_text_MC(question, with_answers)
    elif question_type == "TE": # Text Entry
        output_options = _format_question_text_TE(question, with_answers)
    elif question_type == "Slider":
        output_options = _format_question_text_Slider(question, with_answers)
    elif question_type == "DB":
        output_options = _format_question_text_DB(question, with_answers)
    else:
        raise ValueError(f"Unhandled question type: {question_type}")

    return question_text + "\n" + output_options

def _recursively_extract_text_from_elements(elements_list: list, all_text_lines: list, demographic_only: bool = False):
    """Recursively processes a list of elements (Blocks or Branches)."""
    for element in elements_list:
        element_type = element.get("ElementType")
        if element_type == "Block":
            # For demographic persona, only process the first block
            if demographic_only and all_text_lines:
                continue
            for question in element.get("Questions", []):
                all_text_lines.append(format_question_text(question, with_answers=True))
        elif element_type == "Branch":
            # For persona file, include branch content directly without special markers
            _recursively_extract_text_from_elements(element.get("Elements", []), all_text_lines, demographic_only)
        # Other element types could be handled here if necessary

def read_persona_summary(persona_id: str) -> str:
    """Read the persona summary text for a given persona ID."""
    summary_path = Path("../data/mega_persona_summary_text") / f"pid_{persona_id}_mega_persona.txt"
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: No summary file found for persona {persona_id}")
        return ""

def extract_persona_id(input_path: str) -> str:
    """Extract persona ID from the input file path."""
    # Try to extract from filename
    filename = os.path.basename(input_path)
    match = re.search(r'pid_(\d+)_', filename)
    if match:
        return match.group(1)
    return ""

def convert_persona_to_text(input_path: str, output_path: str, variant: str = "full") -> bool:
    """
    Convert a persona JSON file to a text file.
    
    Args:
        input_path: Path to the input JSON file
        output_path: Path to save the output text file
        variant: Type of conversion ("full", "demographic", "summary", "summary+text", "combine")
    
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            # First read the content as a string
            content = f.read()
            # Parse the JSON string (which may be wrapped in quotes)
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1]  # Remove surrounding quotes
                content = content.replace('\\"', '"')  # Unescape quotes
            # Now parse the actual JSON content
            data_elements = json.loads(content)
            if not isinstance(data_elements, list):
                # If it's a dict like the wave response, try to get 'Elements' key
                if isinstance(data_elements, dict) and "Elements" in data_elements:
                    data_elements = data_elements.get("Elements", [])
                else:
                    print("Error: Input JSON is not in the expected format (list of elements or dict with 'Elements' key).")
                    return False

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {input_path}: {e}")
        return False
    
    all_text_output_lines = []
    
    if variant == "demographic":
        _recursively_extract_text_from_elements(data_elements, all_text_output_lines, demographic_only=True)
    elif variant == "summary":
        persona_id = extract_persona_id(input_path)
        if persona_id:
            summary_text = read_persona_summary(persona_id)
            all_text_output_lines.append(summary_text)
    elif variant == "summary+text":
        persona_id = extract_persona_id(input_path)
        if persona_id:
            summary_text = read_persona_summary(persona_id)
            all_text_output_lines.append(summary_text)
            all_text_output_lines.append("\n--- Detailed Responses ---\n")
            _recursively_extract_text_from_elements(data_elements, all_text_output_lines)
    elif variant == "combine":
        persona_id = extract_persona_id(input_path)
        if persona_id:
            summary_text = read_persona_summary(persona_id)
            all_text_output_lines.append(summary_text)
            all_text_output_lines.append("\n--- Detailed Responses ---\n")
            _recursively_extract_text_from_elements(data_elements, all_text_output_lines)
    else:  # full
        _recursively_extract_text_from_elements(data_elements, all_text_output_lines)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(''.join(all_text_output_lines))
        return True
    except Exception as e:
        print(f"Error writing output file: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a persona JSON file to a single clean text file with questions and answers.")
    parser.add_argument("--input", required=True, help="Input persona JSON file path")
    parser.add_argument("--output", required=True, help="Output text file path")
    parser.add_argument("--variant", choices=["full", "demographic", "summary", "summary+text", "combine"], 
                      default="full", help="Variant of persona to generate")
    
    args = parser.parse_args()
    convert_persona_to_text(args.input, args.output, args.variant) 