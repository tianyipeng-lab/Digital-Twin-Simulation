#!/usr/bin/env python3
import pandas as pd
import json
import os
import yaml
import argparse
from typing import Dict, List, Any, Optional
from collections import defaultdict
import glob

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_answers_from_mega_persona_json(json_path: str, include_text_labels: bool = False) -> Dict[str, Any]:
    """
    Extract answers from a mega persona JSON template file, prioritizing numeric values.
    
    Args:
        json_path: Path to the JSON template file
        include_text_labels: Whether to include text labels in addition to numeric values
        
    Returns:
        Dictionary mapping QuestionIDs to their answers
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        blocks = json.load(f)
    
    answers = {}
    
    # Extract PID from filename
    pid = os.path.basename(json_path).split('_')[1]
    answers['TWIN_ID'] = pid
    
    # Extract wave info from filename, if available
    filename_parts = os.path.basename(json_path).split('_')
    if "mega_persona.json" in os.path.basename(json_path): # Check if it's a mega_persona file
        answers['WAVE'] = 'mega_persona' # Or some other appropriate default
    elif len(filename_parts) > 4 and "A.json" in filename_parts[-1]:
        answers['WAVE'] = filename_parts[4].replace('A.json', '')
    else:
        answers['WAVE'] = 'unknown' # Default if format is unexpected
    
    # Handle both list and dict formats
    blocks_list = blocks if isinstance(blocks, list) else [blocks]
    
    for block in blocks_list:
        if not isinstance(block, dict):
            continue
            
        # Handle both direct questions and nested questions in Elements
        questions = []
        if 'Questions' in block:
            questions.extend(block['Questions'])
        elif 'Elements' in block:
            for element in block['Elements']:
                if isinstance(element, dict) and 'Questions' in element:
                    questions.extend(element['Questions'])
        else:
            # Handle case where block is directly a question
            questions.append(block)
            
        for question in questions:
            try:
                # Extract question info
                qid = question['QuestionID']
                question_type = question.get('QuestionType', '')
                settings = question.get('Settings', {})
                answer_data = question.get('Answers', {})
                
                if not answer_data:
                    continue
                    
                if question_type == 'MC':
                    # Handle multiple choice questions
                    selector = settings.get('Selector', '')
                    
                    if selector in ['SAVR', 'SAHR']:
                        # For single answer MC, prioritize numeric value over text
                        selected_position = answer_data.get('SelectedByPosition', '')
                        
                        if isinstance(selected_position, (int, str)) and selected_position:
                            # Try to use the position as a numeric value
                            try:
                                pos = int(selected_position)
                                answers[qid] = pos  # Store as numeric
                            except ValueError:
                                # If it can't be converted to int, use text as fallback
                                selected_text = answer_data.get('SelectedText', '')
                                answers[qid] = selected_text
                        else:
                            # Use text if no position is available
                            selected_text = answer_data.get('SelectedText', '')
                            answers[qid] = selected_text
                    else:
                        # For multiple selection, use Selected array (numeric) rather than SelectedText
                        selected = answer_data.get('Selected', [])
                        
                        # Store individual selections with 1/0 for presence
                        for choice_id in selected:
                            answers[f"{qid}_{choice_id}"] = 1  # Present
                        
                        # Also store combined selection as a string if requested
                        if include_text_labels:
                            selected_text = answer_data.get('SelectedText', [])
                            if selected_text:
                                answers[f"{qid}_COMBINED_TEXT"] = "|".join(selected_text)
                
                elif question_type == 'Matrix':
                    # Handle matrix questions with numeric emphasis
                    if settings.get('SubSelector') == 'SingleAnswer':
                        # For Likert-type scales, use numeric position values
                        selected = answer_data.get('SelectedByPosition', [])
                        
                        # Check for Statements/StatementsID first, then fall back to Rows/RowsID
                        if 'Statements' in question or 'StatementsID' in question:
                            statement_ids = question.get('StatementsID', [])
                            statements = question.get('Statements', [])
                            if not statement_ids and statements:
                                statement_ids = [str(i+1) for i in range(len(statements))]
                            
                            for i, statement_id in enumerate(statement_ids):
                                if selected and i < len(selected):
                                    # Try to convert to numeric
                                    value = selected[i]
                                    try:
                                        pos = int(value)
                                        answers[f"{qid}_{statement_id}"] = pos  # Store as numeric
                                    except (ValueError, TypeError):
                                        # If not numeric, fall back to text
                                        selected_text = answer_data.get('SelectedText', [])
                                        if selected_text and i < len(selected_text):
                                            answers[f"{qid}_{statement_id}"] = selected_text[i]
                                        else:
                                            answers[f"{qid}_{statement_id}"] = None
                                else:
                                    answers[f"{qid}_{statement_id}"] = None
                        else:
                            # Use Rows/RowsID if no Statements are present
                            row_ids = question.get('RowsID', [])
                            rows = question.get('Rows', [])
                            if not row_ids and rows:
                                row_ids = [str(i+1) for i in range(len(rows))]
                            
                            for i, row_id in enumerate(row_ids):
                                if selected and i < len(selected):
                                    # Try to convert to numeric
                                    value = selected[i]
                                    try:
                                        pos = int(value)
                                        answers[f"{qid}_{row_id}"] = pos  # Store as numeric
                                    except (ValueError, TypeError):
                                        # If not numeric, fall back to text
                                        selected_text = answer_data.get('SelectedText', [])
                                        if selected_text and i < len(selected_text):
                                            answers[f"{qid}_{row_id}"] = selected_text[i]
                                        else:
                                            answers[f"{qid}_{row_id}"] = None
                                else:
                                    answers[f"{qid}_{row_id}"] = None
                    
                elif question_type == 'Slider':
                    # Handle slider questions - numeric values are already prioritized
                    values = answer_data.get('Values', [])
                    
                    # Check if this is a multi-slider question with Statements/Rows
                    has_multiple = (len(values) > 1 and 
                                   ('Statements' in question or 'StatementsID' in question or 
                                    'Rows' in question or 'RowsID' in question))
                    
                    if has_multiple:
                        # Look for StatementsID first, then RowsID
                        item_ids = question.get('StatementsID', []) or question.get('RowsID', [])
                        item_text = question.get('Statements', []) or question.get('Rows', [])
                        
                        if not item_ids and item_text:
                            # If neither is defined, use Statements or Rows with numeric indices
                            item_ids = [str(i+1) for i in range(len(item_text))]
                        
                        # Store each value in a separate column
                        for i, value in enumerate(values):
                            if i < len(item_ids):
                                item_id = item_ids[i]
                                answers[f"{qid}_{item_id}"] = float(value) if value is not None else None
                                
                                # Add text version if requested
                                if include_text_labels and i < len(item_text):
                                    answers[f"{qid}_{item_id}_TEXT"] = item_text[i]
                    else:
                        # Single slider - store as a single value
                        if values:  # Check if we have any values
                            answers[qid] = float(values[0]) if values[0] is not None else None
                        
                elif question_type == 'TE':
                    # For text entry, keep as text (no numeric version available)
                    selector = settings.get('Selector', '')
                    
                    if selector in ['SL', 'ML']:
                        # Check if this is a multi-text question with Statements/Rows
                        if ('Statements' in question or 'StatementsID' in question or 
                            'Rows' in question or 'RowsID' in question):
                            
                            # Look for StatementsID first, then RowsID
                            item_ids = question.get('StatementsID', []) or question.get('RowsID', [])
                            item_text = question.get('Statements', []) or question.get('Rows', [])
                            
                            if not item_ids and item_text:
                                # If neither is defined, use Statements or Rows with numeric indices
                                item_ids = [str(i+1) for i in range(len(item_text))]
                            
                            # Handle multiple text fields - one for each Statement/Row
                            texts = answer_data.get('Text', [])
                            if isinstance(texts, list):
                                for i, text in enumerate(texts):
                                    if i < len(item_ids):
                                        item_id = item_ids[i]
                                        answers[f"{qid}_{item_id}"] = str(text) if text else None
                                        
                                        # Add label text if requested
                                        if include_text_labels and i < len(item_text):
                                            answers[f"{qid}_{item_id}_LABEL"] = item_text[i]
                            else:
                                # Single text field
                                answers[f"{qid}_TEXT"] = str(texts) if texts else None
                        else:
                            # Regular single text field
                            text = answer_data.get('Text', '')
                            answers[f"{qid}_TEXT"] = str(text) if text else None
                    elif selector == 'FORM':
                        texts = answer_data.get('Text', [])
                        
                        # Look for RowsID first, then StatementsID
                        row_ids = question.get('RowsID', []) or question.get('StatementsID', [])
                        row_text = question.get('Rows', []) or question.get('Statements', [])
                        
                        if not row_ids and row_text:
                            # If neither is defined, use Rows or Statements with numeric indices
                            row_ids = [str(i+1) for i in range(len(row_text))]
                        
                        for i, text_dict in enumerate(texts):
                            if i < len(row_ids):
                                row_id = row_ids[i]
                                for _, value in text_dict.items():
                                    answers[f"{qid}_{row_id}"] = str(value) if value else None
                                    
                                    # Add label text if requested
                                    if include_text_labels and i < len(row_text):
                                        answers[f"{qid}_{row_id}_LABEL"] = row_text[i]
                else:
                    # For any other question type, try to extract numeric values first
                    if ('Statements' in question or 'StatementsID' in question or 
                        'Rows' in question or 'RowsID' in question):
                        
                        # Look for StatementsID first, then RowsID
                        item_ids = question.get('StatementsID', []) or question.get('RowsID', [])
                        item_text = question.get('Statements', []) or question.get('Rows', [])
                        
                        if not item_ids and item_text:
                            # If neither is defined, use Statements or Rows with numeric indices
                            item_ids = [str(i+1) for i in range(len(item_text))]
                        
                        # Try to get numeric values first, then text
                        values = answer_data.get('Values', [])
                        if values and isinstance(values, list):
                            for i, value in enumerate(values):
                                if i < len(item_ids):
                                    item_id = item_ids[i]
                                    try:
                                        # Try to convert to numeric
                                        numeric_value = float(value) if value is not None else None
                                        answers[f"{qid}_{item_id}"] = numeric_value
                                    except (ValueError, TypeError):
                                        # If not numeric, use as text
                                        answers[f"{qid}_{item_id}"] = str(value) if value is not None else None
                                    
                                    # Add label text if requested
                                    if include_text_labels and i < len(item_text):
                                        answers[f"{qid}_{item_id}_LABEL"] = item_text[i]
                        else:
                            # Fall back to Text values
                            texts = answer_data.get('Text', [])
                            if isinstance(texts, list):
                                for i, text in enumerate(texts):
                                    if i < len(item_ids):
                                        item_id = item_ids[i]
                                        answers[f"{qid}_{item_id}"] = str(text) if text is not None else None
                            else:
                                # Single value/text field
                                text = answer_data.get('Text', '')
                                answers[qid] = str(text) if text else None
                    else:
                        # Single value case - try numeric first, then text
                        value = answer_data.get('Value')
                        if value is not None:
                            try:
                                # Try to convert to numeric
                                numeric_value = float(value)
                                answers[qid] = numeric_value
                            except (ValueError, TypeError):
                                # If not numeric, use text
                                answers[qid] = str(value)
                        else:
                            # Fall back to Text
                            text = answer_data.get('Text', '')
                            answers[qid] = str(text) if text else None
                    
            except Exception as e:
                print(f"Error processing question {qid} in {json_path}: {str(e)}")
    
    return answers

def extract_text_answers_from_mega_persona_json(json_path: str) -> Dict[str, Any]:
    """
    Extract text answers from a mega persona JSON template file, prioritizing text labels.
    
    Args:
        json_path: Path to the JSON template file
        
    Returns:
        Dictionary mapping QuestionIDs to their text answers
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        blocks = json.load(f)
    
    answers = {}
    
    # Extract PID from filename
    pid = os.path.basename(json_path).split('_')[1]
    answers['TWIN_ID'] = pid
    
    # Extract wave info from filename, if available
    filename_parts = os.path.basename(json_path).split('_')
    if "mega_persona.json" in os.path.basename(json_path): # Check if it's a mega_persona file
        answers['WAVE'] = 'mega_persona' # Or some other appropriate default
    elif len(filename_parts) > 4 and "A.json" in filename_parts[-1]:
        answers['WAVE'] = filename_parts[4].replace('A.json', '')
    else:
        answers['WAVE'] = 'unknown' # Default if format is unexpected
    
    # Handle both list and dict formats
    blocks_list = blocks if isinstance(blocks, list) else [blocks]
    
    for block in blocks_list:
        if not isinstance(block, dict):
            continue
            
        # Handle both direct questions and nested questions in Elements
        questions = []
        if 'Questions' in block:
            questions.extend(block['Questions'])
        elif 'Elements' in block:
            for element in block['Elements']:
                if isinstance(element, dict) and 'Questions' in element:
                    questions.extend(element['Questions'])
        else:
            # Handle case where block is directly a question
            questions.append(block)
            
        for question in questions:
            try:
                # Extract question info
                qid = question['QuestionID']
                question_type = question.get('QuestionType', '')
                settings = question.get('Settings', {})
                answer_data = question.get('Answers', {})
                
                if not answer_data:
                    continue
                    
                if question_type == 'MC':
                    # Handle multiple choice questions
                    selector = settings.get('Selector', '')
                    
                    if selector in ['SAVR', 'SAHR']:
                        # For single answer MC, prioritize SelectedText
                        selected_text = answer_data.get('SelectedText', '')
                        
                        if selected_text:
                            # Use SelectedText directly
                            answers[qid] = selected_text
                        else:
                            # Fall back to SelectedByPosition with lookup
                            selected = answer_data.get('SelectedByPosition', '')
                            if selected and question.get('Columns', []):
                                columns = question.get('Columns', [])
                                if isinstance(selected, str) and selected in columns:
                                    # Text is directly in SelectedByPosition
                                    answers[qid] = selected
                                elif isinstance(selected, (int, str)) and selected:
                                    # Try to convert numeric position to text
                                    try:
                                        pos = int(selected)
                                        if 0 < pos <= len(columns):
                                            answers[qid] = columns[pos-1]
                                        else:
                                            answers[qid] = None
                                    except ValueError:
                                        answers[qid] = None
                                else:
                                    answers[qid] = None
                            else:
                                answers[qid] = None
                    else:
                        # For multiple selection, use SelectedText array
                        selected_text = answer_data.get('SelectedText', [])
                        selected = answer_data.get('Selected', [])
                        
                        if selected_text:
                            # If we have SelectedText, use it
                            answers[f"{qid}_COMBINED_TEXT"] = "|".join(selected_text)
                            
                            # Also store individual selections with their text
                            for i, choice_id in enumerate(selected):
                                if i < len(selected_text):
                                    answers[f"{qid}_{choice_id}"] = selected_text[i]
                        else:
                            # If no SelectedText, try to look up from Choices
                            choices = question.get('Choices', [])
                            choices_id = question.get('ChoicesID', [])
                            
                            selected_labels = []
                            for choice_id in selected:
                                if choices and choices_id:
                                    try:
                                        idx = choices_id.index(choice_id)
                                        if idx < len(choices):
                                            text = choices[idx]
                                            answers[f"{qid}_{choice_id}"] = text
                                            selected_labels.append(text)
                                    except ValueError:
                                        answers[f"{qid}_{choice_id}"] = choice_id
                                        selected_labels.append(choice_id)
                                else:
                                    answers[f"{qid}_{choice_id}"] = choice_id
                                    selected_labels.append(choice_id)
                            
                            if selected_labels:
                                answers[f"{qid}_COMBINED_TEXT"] = "|".join(selected_labels)
                
                elif question_type == 'Matrix':
                    # Handle matrix questions with text emphasis
                    if settings.get('SubSelector') == 'SingleAnswer':
                        # For Likert-type scales
                        selected_text = answer_data.get('SelectedText', [])
                        selected = answer_data.get('SelectedByPosition', [])
                        columns = question.get('Columns', [])
                        
                        # Check for Statements/StatementsID first, then fall back to Rows/RowsID
                        if 'Statements' in question or 'StatementsID' in question:
                            statement_ids = question.get('StatementsID', [])
                            statements = question.get('Statements', [])
                            if not statement_ids and statements:
                                statement_ids = [str(i+1) for i in range(len(statements))]
                            
                            for i, statement_id in enumerate(statement_ids):
                                if selected_text and i < len(selected_text):
                                    # Use SelectedText directly
                                    answers[f"{qid}_{statement_id}"] = selected_text[i]
                                elif selected and i < len(selected) and columns:
                                    # Try to convert position to text
                                    value = selected[i]
                                    try:
                                        pos = int(value)
                                        if 0 < pos <= len(columns):
                                            answers[f"{qid}_{statement_id}"] = columns[pos-1]
                                        else:
                                            answers[f"{qid}_{statement_id}"] = None
                                    except (ValueError, TypeError):
                                        answers[f"{qid}_{statement_id}"] = None
                                else:
                                    answers[f"{qid}_{statement_id}"] = None
                        else:
                            # Use Rows/RowsID if no Statements are present
                            row_ids = question.get('RowsID', [])
                            rows = question.get('Rows', [])
                            if not row_ids and rows:
                                row_ids = [str(i+1) for i in range(len(rows))]
                            
                            for i, row_id in enumerate(row_ids):
                                if selected_text and i < len(selected_text):
                                    # Use SelectedText directly
                                    answers[f"{qid}_{row_id}"] = selected_text[i]
                                elif selected and i < len(selected) and columns:
                                    # Try to convert position to text
                                    value = selected[i]
                                    try:
                                        pos = int(value)
                                        if 0 < pos <= len(columns):
                                            answers[f"{qid}_{row_id}"] = columns[pos-1]
                                        else:
                                            answers[f"{qid}_{row_id}"] = None
                                    except (ValueError, TypeError):
                                        answers[f"{qid}_{row_id}"] = None
                                else:
                                    answers[f"{qid}_{row_id}"] = None
                    
                elif question_type == 'Slider':
                    # For slider questions, use the numeric values as text
                    # HSLIDER questions typically don't have text labels, so we use the values directly
                    values = answer_data.get('Values', [])
                    
                    # Check if this is a multi-slider question with Statements/Rows
                    has_multiple = (len(values) > 1 and 
                                   ('Statements' in question or 'StatementsID' in question or 
                                    'Rows' in question or 'RowsID' in question))
                    
                    if has_multiple:
                        # Look for StatementsID first, then RowsID
                        item_ids = question.get('StatementsID', []) or question.get('RowsID', [])
                        item_text = question.get('Statements', []) or question.get('Rows', [])
                        
                        if not item_ids and item_text:
                            # If neither is defined, use Statements or Rows with numeric indices
                            item_ids = [str(i+1) for i in range(len(item_text))]
                        
                        # Store each value in a separate column
                        for i, value in enumerate(values):
                            if i < len(item_ids):
                                item_id = item_ids[i]
                                # Convert the numeric value to string
                                answers[f"{qid}_{item_id}"] = str(value) if value is not None else None
                    else:
                        # Single slider - store as a single value
                        if values:  # Check if we have any values
                            # Convert the numeric value to string
                            answers[qid] = str(values[0]) if values[0] is not None else None
                
                elif question_type == 'TE':
                    # Handle text entry questions - preserve as text
                    selector = settings.get('Selector', '')
                    
                    if selector in ['SL', 'ML']:
                        # Check if this is a multi-text question with Statements/Rows
                        if ('Statements' in question or 'StatementsID' in question or 
                            'Rows' in question or 'RowsID' in question):
                            
                            # Look for StatementsID first, then RowsID
                            item_ids = question.get('StatementsID', []) or question.get('RowsID', [])
                            item_text = question.get('Statements', []) or question.get('Rows', [])
                            
                            if not item_ids and item_text:
                                # If neither is defined, use Statements or Rows with numeric indices
                                item_ids = [str(i+1) for i in range(len(item_text))]
                            
                            # Handle multiple text fields - one for each Statement/Row
                            texts = answer_data.get('Text', [])
                            if isinstance(texts, list):
                                for i, text in enumerate(texts):
                                    if i < len(item_ids):
                                        item_id = item_ids[i]
                                        answers[f"{qid}_{item_id}"] = str(text) if text else None
                            else:
                                # Single text field
                                answers[f"{qid}_TEXT"] = str(texts) if texts else None
                        else:
                            # Regular single text field
                            text = answer_data.get('Text', '')
                            answers[f"{qid}_TEXT"] = str(text) if text else None
                    elif selector == 'FORM':
                        texts = answer_data.get('Text', [])
                        
                        # Look for RowsID first, then StatementsID
                        row_ids = question.get('RowsID', []) or question.get('StatementsID', [])
                        row_text = question.get('Rows', []) or question.get('Statements', [])
                        
                        if not row_ids and row_text:
                            # If neither is defined, use Rows or Statements with numeric indices
                            row_ids = [str(i+1) for i in range(len(row_text))]
                        
                        for i, text_dict in enumerate(texts):
                            if i < len(row_ids):
                                row_id = row_ids[i]
                                for _, value in text_dict.items():
                                    answers[f"{qid}_{row_id}"] = str(value) if value else None
                else:
                    # For any other question type, check for multiple statements/rows
                    if ('Statements' in question or 'StatementsID' in question or 
                        'Rows' in question or 'RowsID' in question):
                        
                        # Look for StatementsID first, then RowsID
                        item_ids = question.get('StatementsID', []) or question.get('RowsID', [])
                        item_text = question.get('Statements', []) or question.get('Rows', [])
                        
                        if not item_ids and item_text:
                            # If neither is defined, use Statements or Rows with numeric indices
                            item_ids = [str(i+1) for i in range(len(item_text))]
                        
                        # Handle multiple text fields or values
                        raw_answers = answer_data.get('Text', []) or answer_data.get('Values', [])
                        if isinstance(raw_answers, list):
                            for i, value in enumerate(raw_answers):
                                if i < len(item_ids):
                                    item_id = item_ids[i]
                                    answers[f"{qid}_{item_id}"] = str(value) if value is not None else None
                        else:
                            # Single value/text field
                            answers[qid] = str(raw_answers) if raw_answers is not None else None
                    else:
                        # Single value case
                        text = answer_data.get('Text', '')
                        answers[qid] = str(text) if text else None
                    
            except Exception as e:
                print(f"Error processing question {qid} in {json_path}: {str(e)}")
    
    return answers

def process_mega_persona(config: dict):
    """
    Process mega persona JSON files and convert to CSV files.
    Always extracts both numeric and text formats.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing both numeric and text DataFrames
    """
    output_dir = config['pipeline']['output_dir']
    wave = config['mega_persona_extraction']['wave']
    output_csv_base = config['mega_persona_extraction']['output_csv']
    source_dir = config['pipeline'].get('source_dir', 'answer_blocks')
    
    # Generate filenames - numeric uses original name, text uses _label suffix
    numeric_filename = output_csv_base  # Keep original name for numeric
    text_filename = output_csv_base.replace('.csv', '_label.csv')
    
    # Check if we're using absolute or relative paths
    if os.path.isabs(output_dir):
        base_dir = output_dir
    else:
        # If it's a relative path, use the current working directory
        base_dir = output_dir
        
    # Print for debugging
    print(f"Looking for files in {base_dir}/{source_dir}")
    
    # Determine which JSON files to process based on wave and actual file format
    # First try the MegaPersona pattern (old format)
    json_pattern = f"{base_dir}/{source_dir}/MegaPersona_*_{wave}_A.json"
    json_files = glob.glob(json_pattern)
    
    # If no files found, try the pid pattern (new format)
    if not json_files:
        json_pattern = f"{base_dir}/{source_dir}/pid_*_wave4_Q_{wave}_A.json"
        json_files = glob.glob(json_pattern)
        
    # If still no files found, try looking directly in the CWD
    if not json_files and not os.path.isabs(output_dir):
        json_pattern = f"{source_dir}/pid_*_wave4_Q_{wave}_A.json"
        json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"Warning: No files found matching the pattern {json_pattern}")
        return None
    
    print(f"Found {len(json_files)} files to process")
    
    # Process each JSON file and extract answers in both formats
    all_numeric_answers = []
    all_text_answers = []
    
    for json_file in json_files:
        # Extract numeric answers
        numeric_answers = extract_answers_from_mega_persona_json(json_file, include_text_labels=False)
        all_numeric_answers.append(numeric_answers)
        
        # Extract text answers
        text_answers = extract_text_answers_from_mega_persona_json(json_file)
        all_text_answers.append(text_answers)
    
    # Convert to DataFrames
    df_numeric = pd.DataFrame(all_numeric_answers)
    df_text = pd.DataFrame(all_text_answers)
    
    # Save both CSV files
    numeric_path = os.path.join(base_dir, 'mega_persona_csv', numeric_filename)
    os.makedirs(os.path.dirname(numeric_path), exist_ok=True)
    df_numeric.to_csv(numeric_path, index=False)
    print(f"Saved numeric CSV file to {numeric_path}")
    
    text_path = os.path.join(base_dir, 'mega_persona_csv', text_filename)
    df_text.to_csv(text_path, index=False)
    print(f"Saved text CSV file to {text_path}")
    
    # Return both DataFrames in a dictionary
    return {
        'numeric': df_numeric,
        'text': df_text
    }

def main():
    parser = argparse.ArgumentParser(description='Extract mega persona data to CSV')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # When run as standalone script, set up default config structure
    if 'mega_persona_extraction' not in config:
        output_dir = config['pipeline']['output_dir']
        config = {
            'pipeline': {
                'output_dir': output_dir
            },
            'mega_persona_extraction': {
                'output_csv': 'mega_persona_data.csv'
            }
        }
    
    # Process mega persona files
    process_mega_persona(config)

if __name__ == "__main__":
    main() 