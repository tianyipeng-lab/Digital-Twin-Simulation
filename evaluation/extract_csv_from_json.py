#!/usr/bin/env python3
import pandas as pd
import json
import os
from typing import Dict, List, Any, Optional
from collections import defaultdict

def extract_question_info_from_qsf(qsf_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Extract question information from QSF file to help with CSV column mapping.
    
    Args:
        qsf_path: Path to the QSF file
        
    Returns:
        Dictionary mapping QuestionIDs to their metadata
    """
    with open(qsf_path, 'r', encoding='utf-8') as f:
        qsf_data = json.load(f)
    
    question_info = {}
    
    def process_element(element: Dict):
        if not isinstance(element, dict):
            return
            
        if element.get('Type') == 'Question':
            qid = element.get('QuestionID')
            if qid:
                question_info[qid] = {
                    'type': element.get('Selector', ''),
                    'sub_type': element.get('SubSelector', ''),
                    'choices': element.get('Choices', {}),
                    'rows': element.get('Rows', {}),
                    'columns': element.get('Columns', {})
                }
        
        # Process nested elements
        for key, value in element.items():
            if isinstance(value, (dict, list)):
                if isinstance(value, dict):
                    process_element(value)
                else:
                    for item in value:
                        process_element(item)
    
    # Process all elements in QSF
    for element in qsf_data.get('SurveyElements', []):
        process_element(element)
    
    return question_info

def extract_answers_from_json(json_path: str) -> Dict[str, Any]:
    """
    Extract answers from a JSON template file.
    
    Args:
        json_path: Path to the JSON template file
        
    Returns:
        Dictionary mapping QuestionIDs to their answers
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        template_data = json.load(f)
    
    answers = {}
    
    def process_element(element: Dict):
        if not isinstance(element, dict):
            return
            
        if 'QuestionID' in element and 'Answers' in element:
            qid = element['QuestionID']
            
            if element['QuestionType'] == 'Matrix':
                # Handle matrix questions
                by_row = {}
                selected_positions = element['Answers'].get('SelectedByPosition', [])
                for i, pos in enumerate(selected_positions):
                    row_id = element.get('RowsID', [])[i] if 'RowsID' in element else str(i + 1)
                    by_row[f"{qid}_{row_id}"] = pos
                answers.update(by_row)
                
            elif element['QuestionType'] == 'Slider':
                # Handle slider questions
                values = element['Answers'].get('Values', [])
                for i, value in enumerate(values):
                    stmt_id = element.get('StatementsID', [])[i] if 'StatementsID' in element else str(i + 1)
                    answers[f"{qid}_{stmt_id}"] = value
                    
            elif element['QuestionType'] == 'MC':
                # Handle multiple choice questions
                if element['Settings']['Selector'] in ['SAVR', 'SAHR']:
                    answers[qid] = element['Answers'].get('SelectedByPosition', '')
                else:  # MAVR, MAHR
                    selected = element['Answers'].get('SelectedByPosition', [])
                    for i, pos in enumerate(selected):
                        option_id = element.get('OptionsID', [])[i] if 'OptionsID' in element else str(i + 1)
                        answers[f"{qid}_{option_id}"] = 1
                        
            elif element['QuestionType'] == 'TE':
                # Handle text entry questions
                if element['Settings']['Selector'] in ['SL', 'ML']:
                    answers[f"{qid}_TEXT"] = element['Answers'].get('Text', '')
                elif element['Settings']['Selector'] == 'FORM':
                    texts = element['Answers'].get('Text', [])
                    for i, text_dict in enumerate(texts):
                        for _, value in text_dict.items():
                            row_id = element.get('RowsID', [])[i] if 'RowsID' in element else str(i + 1)
                            answers[f"{qid}_{row_id}"] = value
        
        # Process nested elements
        if 'Elements' in element:
            for sub_element in element['Elements']:
                process_element(sub_element)
        if 'Questions' in element:
            for question in element['Questions']:
                process_element(question)
    
    # Process all elements in template
    for element in template_data.get('Elements', []):
        process_element(element)
    
    return answers

def json_templates_to_csv(json_dir: str, qsf_path: str, output_csv: str):
    """
    Convert multiple JSON template files to a single CSV file.
    
    Args:
        json_dir: Directory containing JSON template files
        qsf_path: Path to the QSF file
        output_csv: Path to save the output CSV file
    """
    # Extract question info from QSF
    question_info = extract_question_info_from_qsf(qsf_path)
    
    # Collect all answers from JSON templates
    all_responses = []
    question_ids = set()
    
    for filename in os.listdir(json_dir):
        if filename.endswith('_response.json'):
            pid = filename.split('_')[0]
            json_path = os.path.join(json_dir, filename)
            
            # Extract answers from this template
            answers = extract_answers_from_json(json_path)
            answers['TWIN_ID'] = pid  # Add participant ID
            
            all_responses.append(answers)
            question_ids.update(answers.keys())
    
    # Convert to DataFrame
    df = pd.DataFrame(all_responses)
    
    # Ensure TWIN_ID is the first column
    cols = ['TWIN_ID'] + [col for col in df.columns if col != 'TWIN_ID']
    df = df[cols]
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Created CSV file with {len(df)} rows and {len(df.columns)} columns")
    return df

# Test case
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert JSON templates back to CSV format")
    parser.add_argument("--json_dir", required=True, help="Directory containing JSON template files")
    parser.add_argument("--qsf_path", required=True, help="Path to the QSF file")
    parser.add_argument("--output_csv", required=True, help="Path to save the output CSV file")
    
    args = parser.parse_args()
    
    # Run the conversion
    df = json_templates_to_csv(args.json_dir, args.qsf_path, args.output_csv) 