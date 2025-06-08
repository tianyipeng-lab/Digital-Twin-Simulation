#!/usr/bin/env python3
"""
json2csv.py - Unified script for converting JSON answer blocks to CSV format

This script combines functionality from convert_and_compare_responses.py and convert_csv_format.py
to provide a clean, modular interface for converting JSON survey data to various CSV formats.

Features:
- Convert JSON answer blocks to numeric CSV
- Format CSV to match benchmark structure
- Generate labeled versions of CSV files
- Create question mapping files
- Generate randdollar breakdown for pricing analysis
"""

import os
import json
import pandas as pd
import numpy as np
import argparse
import yaml
import re
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import csv
from collections import defaultdict
from enum import Enum
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ExtractionMode(Enum):
    """Extraction mode for prioritizing numeric vs text values."""
    NUMERIC = "numeric"
    TEXT = "text"


@dataclass
class QuestionContext:
    """Context information for processing a question."""
    qid: str
    question_type: str
    settings: Dict[str, Any]
    answer_data: Dict[str, Any]
    question: Dict[str, Any]
    mode: ExtractionMode
    include_labels: bool = False


class AnswerExtractor:
    """Main class for extracting answers from JSON files."""
    
    def __init__(self, mode: ExtractionMode = ExtractionMode.NUMERIC):
        self.mode = mode
        self.processors = {
            'MC': self._process_mc_question,
            'Matrix': self._process_matrix_question,
            'Slider': self._process_slider_question,
            'TE': self._process_te_question,
            'CS': self._process_cs_question
        }
    
    def extract_from_file(self, json_path: str, include_text_labels: bool = False) -> Dict[str, Any]:
        """Extract answers from a mega persona JSON file."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                blocks = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {json_path}: {e}")
            return {}
        
        # Extract metadata
        answers = self._extract_metadata(json_path)
        
        # Extract and process all questions
        questions = self._extract_questions(blocks)
        
        for question in questions:
            try:
                question_answers = self._process_question(question, include_text_labels)
                answers.update(question_answers)
            except Exception as e:
                logger.warning(f"Error processing question {question.get('QuestionID', 'unknown')}: {e}")
        
        return answers
    
    def _extract_metadata(self, json_path: str) -> Dict[str, str]:
        """Extract TWIN_ID and WAVE from filename."""
        answers = {}
        basename = os.path.basename(json_path)
        parts = basename.split('_')
        
        # Extract PID
        if len(parts) > 1:
            answers['TWIN_ID'] = parts[1]
        
        # Extract wave info
        if "mega_persona.json" in basename:
            answers['WAVE'] = 'mega_persona'
        elif len(parts) > 4 and "A.json" in parts[-1]:
            answers['WAVE'] = parts[4].replace('A.json', '')
        else:
            answers['WAVE'] = 'unknown'
        
        return answers
    
    def _extract_questions(self, blocks: Union[List, Dict]) -> List[Dict[str, Any]]:
        """Extract all questions from JSON blocks structure."""
        questions = []
        blocks_list = blocks if isinstance(blocks, list) else [blocks]
        
        for block in blocks_list:
            if not isinstance(block, dict):
                continue
                
            # Direct questions
            if 'Questions' in block:
                questions.extend(self._flatten_questions(block['Questions']))
            # Questions in Elements
            elif 'Elements' in block:
                for element in block['Elements']:
                    if isinstance(element, dict) and 'Questions' in element:
                        questions.extend(self._flatten_questions(element['Questions']))
            # Block is directly a question
            else:
                questions.append(block)
        
        return questions
    
    def _flatten_questions(self, questions_data: Union[List, Dict]) -> List[Dict]:
        """Flatten questions data structure."""
        if isinstance(questions_data, list):
            return questions_data
        elif isinstance(questions_data, dict):
            return list(questions_data.values())
        return []
    
    def _process_question(self, question: Dict[str, Any], include_labels: bool) -> Dict[str, Any]:
        """Process a single question."""
        qid = question.get('QuestionID', '')
        if not qid:
            return {}
            
        question_type = question.get('QuestionType', '')
        answer_data = question.get('Answers', {})
        
        if not answer_data:
            return {}
        
        ctx = QuestionContext(
            qid=qid,
            question_type=question_type,
            settings=question.get('Settings', {}),
            answer_data=answer_data,
            question=question,
            mode=self.mode,
            include_labels=include_labels
        )
        
        # Get processor for question type
        processor = self.processors.get(question_type, self._process_other_question)
        return processor(ctx)
    
    def _get_item_ids_and_text(self, question: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Extract item IDs and text from Statements/Rows."""
        if 'Statements' in question or 'StatementsID' in question:
            item_ids = question.get('StatementsID', [])
            item_text = question.get('Statements', [])
        else:
            item_ids = question.get('RowsID', [])
            item_text = question.get('Rows', [])
        
        if not item_ids and item_text:
            item_ids = [str(i+1) for i in range(len(item_text))]
        
        return item_ids, item_text
    
    def _process_mc_question(self, ctx: QuestionContext) -> Dict[str, Any]:
        """Process multiple choice questions."""
        answers = {}
        selector = ctx.settings.get('Selector', '')
        
        if selector in ['SAVR', 'SAHR']:  # Single answer
            if self.mode == ExtractionMode.NUMERIC:
                answers.update(self._process_mc_single_numeric(ctx))
            else:
                answers.update(self._process_mc_single_text(ctx))
        else:  # Multiple selection
            if self.mode == ExtractionMode.NUMERIC:
                answers.update(self._process_mc_multiple_numeric(ctx))
            else:
                answers.update(self._process_mc_multiple_text(ctx))
        
        return answers
    
    def _process_mc_single_numeric(self, ctx: QuestionContext) -> Dict[str, Any]:
        """Process single answer MC for numeric mode."""
        selected_position = ctx.answer_data.get('SelectedByPosition', '')
        
        if isinstance(selected_position, (int, str)) and selected_position:
            try:
                return {ctx.qid: int(selected_position)}
            except ValueError:
                # Fall back to text if position can't be converted
                selected_text = ctx.answer_data.get('SelectedText', '')
                return {ctx.qid: selected_text}
        else:
            selected_text = ctx.answer_data.get('SelectedText', '')
            return {ctx.qid: selected_text}
    
    def _process_mc_single_text(self, ctx: QuestionContext) -> Dict[str, Any]:
        """Process single answer MC for text mode."""
        selected_text = ctx.answer_data.get('SelectedText', '')
        
        if selected_text:
            return {ctx.qid: selected_text}
        
        # Fall back to position lookup
        selected = ctx.answer_data.get('SelectedByPosition', '')
        if not selected:
            return {ctx.qid: None}
        
        # Try to look up text from position
        columns = ctx.question.get('Columns', []) or ctx.question.get('Options', [])
        if not columns:
            return {ctx.qid: str(selected) if selected else None}
        
        if isinstance(selected, str) and selected in columns:
            return {ctx.qid: selected}
        
        try:
            pos = int(selected) - 1  # 1-indexed to 0-indexed
            if 0 <= pos < len(columns):
                return {ctx.qid: columns[pos]}
            else:
                return {ctx.qid: str(selected)}
        except (ValueError, TypeError):
            return {ctx.qid: str(selected) if selected else None}
    
    def _process_mc_multiple_numeric(self, ctx: QuestionContext) -> Dict[str, Any]:
        """Process multiple selection MC for numeric mode."""
        selected = ctx.answer_data.get('Selected', [])
        return {f"{ctx.qid}_{choice_id}": 1 for choice_id in selected}
    
    def _process_mc_multiple_text(self, ctx: QuestionContext) -> Dict[str, Any]:
        """Process multiple selection MC for text mode."""
        selected_text = ctx.answer_data.get('SelectedText', [])
        answers = {}
        
        if isinstance(selected_text, list):
            for i, text in enumerate(selected_text):
                if text:
                    answers[f"{ctx.qid}_{i+1}"] = text
        elif selected_text:
            answers[ctx.qid] = selected_text
        
        return answers
    
    def _process_matrix_question(self, ctx: QuestionContext) -> Dict[str, Any]:
        """Process matrix questions."""
        answers = {}
        item_ids, item_text = self._get_item_ids_and_text(ctx.question)
        
        if self.mode == ExtractionMode.NUMERIC:
            selected = ctx.answer_data.get('SelectedByPosition', [])
            for i, item_id in enumerate(item_ids):
                if selected and i < len(selected):
                    value = selected[i]
                    try:
                        answers[f"{ctx.qid}_{item_id}"] = int(value)
                    except (ValueError, TypeError):
                        # Fall back to text
                        selected_text = ctx.answer_data.get('SelectedText', [])
                        if selected_text and i < len(selected_text):
                            answers[f"{ctx.qid}_{item_id}"] = selected_text[i]
                        else:
                            answers[f"{ctx.qid}_{item_id}"] = None
                else:
                    answers[f"{ctx.qid}_{item_id}"] = None
        else:  # TEXT mode
            selected_text = ctx.answer_data.get('SelectedText', [])
            selected = ctx.answer_data.get('SelectedByPosition', [])
            columns = ctx.question.get('Columns', [])
            
            for i, item_id in enumerate(item_ids):
                if selected_text and i < len(selected_text):
                    answers[f"{ctx.qid}_{item_id}"] = selected_text[i]
                elif selected and i < len(selected) and columns:
                    # Look up text from position
                    try:
                        pos = int(selected[i]) - 1
                        if 0 <= pos < len(columns):
                            answers[f"{ctx.qid}_{item_id}"] = columns[pos]
                        else:
                            answers[f"{ctx.qid}_{item_id}"] = str(selected[i])
                    except (ValueError, TypeError):
                        answers[f"{ctx.qid}_{item_id}"] = str(selected[i]) if selected[i] else None
                else:
                    answers[f"{ctx.qid}_{item_id}"] = None
        
        return answers
    
    def _process_slider_question(self, ctx: QuestionContext) -> Dict[str, Any]:
        """Process slider questions."""
        answers = {}
        values = ctx.answer_data.get('Values', [])
        
        # Check if multi-slider
        has_multiple = (len(values) > 1 and 
                       ('Statements' in ctx.question or 'StatementsID' in ctx.question or 
                        'Rows' in ctx.question or 'RowsID' in ctx.question))
        
        if has_multiple:
            item_ids, item_text = self._get_item_ids_and_text(ctx.question)
            for i, value in enumerate(values):
                if i < len(item_ids):
                    item_id = item_ids[i]
                    if self.mode == ExtractionMode.NUMERIC:
                        answers[f"{ctx.qid}_{item_id}"] = float(value) if value is not None else None
                        if ctx.include_labels and i < len(item_text):
                            answers[f"{ctx.qid}_{item_id}_TEXT"] = item_text[i]
                    else:
                        answers[f"{ctx.qid}_{item_id}"] = str(value) if value is not None else None
        else:
            # Single slider
            if values:
                if self.mode == ExtractionMode.NUMERIC:
                    answers[ctx.qid] = float(values[0]) if values[0] is not None else None
                else:
                    answers[ctx.qid] = str(values[0]) if values[0] is not None else None
        
        return answers
    
    def _process_te_question(self, ctx: QuestionContext) -> Dict[str, Any]:
        """Process text entry questions."""
        answers = {}
        selector = ctx.settings.get('Selector', '')
        
        if selector in ['SL', 'ML']:
            # Check for multi-text
            if any(key in ctx.question for key in ['Statements', 'StatementsID', 'Rows', 'RowsID']):
                item_ids, item_text = self._get_item_ids_and_text(ctx.question)
                
                if self.mode == ExtractionMode.NUMERIC:
                    texts = ctx.answer_data.get('Text', [])
                    if isinstance(texts, list):
                        for i, text in enumerate(texts):
                            if i < len(item_ids):
                                item_id = item_ids[i]
                                answers[f"{ctx.qid}_{item_id}"] = str(text) if text is not None else None
                                if ctx.include_labels and i < len(item_text):
                                    answers[f"{ctx.qid}_{item_id}_LABEL"] = item_text[i]
                    else:
                        answers[f"{ctx.qid}_TEXT"] = str(texts) if texts else None
                else:  # TEXT mode
                    raw_answers = ctx.answer_data.get('Text', []) or ctx.answer_data.get('Values', [])
                    if isinstance(raw_answers, list):
                        for i, value in enumerate(raw_answers):
                            if i < len(item_ids):
                                item_id = item_ids[i]
                                answers[f"{ctx.qid}_{item_id}"] = str(value) if value is not None else None
                    else:
                        answers[f"{ctx.qid}_TEXT"] = str(raw_answers) if raw_answers is not None else None
            else:
                # Single text field
                text = ctx.answer_data.get('Text', '')
                answers[f"{ctx.qid}_TEXT"] = str(text) if text else None
        else:
            # Single text field
            text = ctx.answer_data.get('Text', '')
            answers[f"{ctx.qid}_TEXT"] = str(text) if text else None
        
        return answers
    
    def _process_cs_question(self, ctx: QuestionContext) -> Dict[str, Any]:
        """Process constant sum questions."""
        answers = {}
        item_ids, item_text = self._get_item_ids_and_text(ctx.question)
        
        values = ctx.answer_data.get('Values', [])
        if values and isinstance(values, list):
            for i, value in enumerate(values):
                if i < len(item_ids):
                    item_id = item_ids[i]
                    try:
                        numeric_value = float(value) if value is not None else None
                        answers[f"{ctx.qid}_{item_id}"] = numeric_value
                    except (ValueError, TypeError):
                        answers[f"{ctx.qid}_{item_id}"] = str(value) if value is not None else None
                    
                    if ctx.include_labels and i < len(item_text):
                        answers[f"{ctx.qid}_{item_id}_LABEL"] = item_text[i]
        else:
            # Fall back to text
            texts = ctx.answer_data.get('Text', [])
            if isinstance(texts, list):
                for i, text in enumerate(texts):
                    if i < len(item_ids):
                        item_id = item_ids[i]
                        answers[f"{ctx.qid}_{item_id}"] = str(text) if text is not None else None
            else:
                text = ctx.answer_data.get('Text', '')
                answers[ctx.qid] = str(text) if text else None
        
        return answers
    
    def _process_other_question(self, ctx: QuestionContext) -> Dict[str, Any]:
        """Process other question types."""
        answers = {}
        
        # Try to extract value based on mode
        if self.mode == ExtractionMode.NUMERIC:
            value = ctx.answer_data.get('Value')
            if value is not None:
                try:
                    answers[ctx.qid] = float(value)
                except (ValueError, TypeError):
                    answers[ctx.qid] = str(value)
            else:
                values = ctx.answer_data.get('Values', [])
                if values and isinstance(values, list) and values[0] is not None:
                    try:
                        answers[ctx.qid] = float(values[0])
                    except (ValueError, TypeError):
                        answers[ctx.qid] = str(values[0])
                else:
                    text = ctx.answer_data.get('Text', '')
                    answers[ctx.qid] = str(text) if text else None
        else:  # TEXT mode
            text = ctx.answer_data.get('Text', '')
            if text:
                answers[ctx.qid] = str(text)
            else:
                value = ctx.answer_data.get('Value', '')
                if value:
                    answers[ctx.qid] = str(value)
                else:
                    values = ctx.answer_data.get('Values', [])
                    if values and isinstance(values, list) and values[0] is not None:
                        answers[ctx.qid] = str(values[0])
                    else:
                        answers[ctx.qid] = None
        
        return answers


class JSONToCSVConverter:
    """Main converter class for JSON to CSV transformation using the original logic"""
    
    def __init__(self, config: dict):
        """Initialize converter with configuration"""
        self.config = self._process_config(config)
        self.question_mapping = {}
        self.question_types = {}
        self.benchmark_descriptions = {}  # Store descriptions from benchmark CSV
    
    def _process_config(self, config: dict) -> dict:
        """Process and validate configuration."""
        # Create a copy to avoid modifying the original
        processed = config.copy()
        
        # Substitute ${trial_dir} in all paths if trial_dir is defined
        if 'trial_dir' in processed:
            processed = self._substitute_trial_dir(processed)
        
        # Validate required fields
        self._validate_config(processed)
        
        return processed
    
    def _substitute_trial_dir(self, config: dict) -> dict:
        """Recursively substitute ${trial_dir} in all string values."""
        trial_dir = config['trial_dir']
        
        def substitute_in_value(value):
            if isinstance(value, str):
                return value.replace('${trial_dir}', trial_dir)
            elif isinstance(value, dict):
                return {k: substitute_in_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_in_value(v) for v in value]
            else:
                return value
        
        # Create new config with substitutions
        return {key: substitute_in_value(value) if key != 'trial_dir' else value 
                for key, value in config.items()}
    
    def _validate_config(self, config: dict):
        """Validate configuration has required fields."""
        if 'waves' not in config:
            raise ValueError("Configuration must contain 'waves' section")
        
    def process_wave(self, wave_name: str, wave_config: dict) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Process a single wave using the original logic
        
        Returns dict with 'numeric' and 'text' DataFrames
        """
        input_pattern = wave_config.get('input_pattern')
        if not input_pattern:
            logger.warning(f"No input pattern specified for {wave_name}")
            return None
            
        # Get actual files matching the pattern
        full_pattern = input_pattern.replace('{pid}', '*')
        json_files = sorted(glob.glob(full_pattern))
        
        if not json_files:
            logger.warning(f"No files found matching pattern: {full_pattern}")
            return None
            
        logger.info(f"Found {len(json_files)} files to process")
        
        # Apply max_personas limit if set
        if self.config.get('max_personas'):
            json_files = json_files[:self.config['max_personas']]
            logger.info(f"Limited to {len(json_files)} files due to max_personas setting")
        
        # Process each JSON file and extract answers in both formats
        all_numeric_answers = []
        all_text_answers = []
        
        numeric_extractor = AnswerExtractor(ExtractionMode.NUMERIC)
        text_extractor = AnswerExtractor(ExtractionMode.TEXT)
        
        for json_file in json_files:
            # Extract numeric answers
            numeric_answers = numeric_extractor.extract_from_file(json_file, include_text_labels=False)
            if numeric_answers:
                all_numeric_answers.append(numeric_answers)
            
            # Extract text answers
            text_answers = text_extractor.extract_from_file(json_file)
            if text_answers:
                all_text_answers.append(text_answers)
        
        if not all_numeric_answers:
            logger.warning(f"No valid answers extracted for {wave_name}")
            return None
        
        # Convert to DataFrames
        df_numeric = pd.DataFrame(all_numeric_answers)
        df_text = pd.DataFrame(all_text_answers)
        
        # Return both DataFrames in a dictionary
        return {
            'numeric': df_numeric,
            'text': df_text
        }
    
                        
    def format_to_benchmark(self, df: pd.DataFrame, benchmark_csv: str, 
                          column_mapping_file: Optional[str] = None) -> pd.DataFrame:
        """
        Format DataFrame to match benchmark CSV structure using the same logic as convert_csv_format.py
        
        Args:
            df: Input DataFrame
            benchmark_csv: Path to benchmark CSV file
            column_mapping_file: Optional manual column mapping file
            
        Returns:
            Formatted DataFrame matching benchmark structure
        """
        try:
            # Load benchmark structure
            benchmark_df = pd.read_csv(benchmark_csv, nrows=0)  # Only need column structure
            benchmark_columns = list(benchmark_df.columns)
        except Exception as e:
            logger.error(f"Failed to load benchmark CSV {benchmark_csv}: {e}")
            return df
        
        # Ensure TWIN_ID is first
        if 'TWIN_ID' in benchmark_columns:
            benchmark_columns.remove('TWIN_ID')
        benchmark_columns = ['TWIN_ID'] + benchmark_columns
        
        # Extract ImportId mappings and descriptions from wave4 CSV (like convert_csv_format.py does)
        importid_mapping, descriptions_by_column, raw_wave4_descriptions = self._extract_importid_mapping_with_descriptions(benchmark_csv)
        
        # Load manual column mapping if provided
        manual_mapping = self._load_column_mapping(column_mapping_file)
        
        # Combine mappings (manual takes precedence)
        final_mapping = {**importid_mapping, **manual_mapping}
        
        # Create reverse mapping (input_col -> wave4_col)
        reverse_mapping = {v: k for k, v in final_mapping.items() if v}
        
        # Build formatted data following convert_csv_format.py logic
        formatted_data = self._build_formatted_data_v2(df, benchmark_columns, reverse_mapping)
        
        # Create DataFrame from dictionary
        df_formatted = pd.DataFrame(formatted_data)
        
        # Ensure column order matches benchmark
        final_columns = [col for col in benchmark_columns if col in df_formatted.columns]
        additional_cols = sorted([col for col in df_formatted.columns if col not in benchmark_columns])
        
        # Store descriptions for later use
        self.benchmark_descriptions = descriptions_by_column
        
        return df_formatted[final_columns + additional_cols]
    
    def _extract_importid_mapping_with_descriptions(self, benchmark_csv: str) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
        """
        Extract ImportId to column name mapping and descriptions from wave4 CSV.
        Returns:
            - mapping: ImportId mapping (wave4_col -> import_id)
            - descriptions_by_column: Descriptions by column name
            - raw_wave4_descriptions: Raw descriptions from wave4 CSV
        """
        mapping = {}
        descriptions_by_column = {}
        raw_wave4_descriptions = {}
        
        try:
            # Read the first few rows to find ImportId row
            with open(benchmark_csv, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = []
                for i, row in enumerate(reader):
                    rows.append(row)
                    if i >= 5:  # Check first 6 rows
                        break
            
            if len(rows) < 2:
                return mapping, descriptions_by_column, raw_wave4_descriptions
            
            headers = rows[0]
            
            # Look for ImportId row (usually row 3, index 2)
            import_row_idx = None
            for idx in range(1, min(len(rows), 6)):
                row = rows[idx]
                # Check if this row contains ImportId data
                importid_count = sum(1 for cell in row if cell.startswith('{') and '"ImportId"' in cell)
                if importid_count > len(row) * 0.3:  # If >30% of cells have ImportId
                    import_row_idx = idx
                    break
            
            if import_row_idx is None:
                logger.warning("No ImportId row found in wave4 CSV")
                # Still try to get descriptions if possible
                if len(rows) > 1:
                    description_row = rows[1]
                    for col_idx, (header, desc) in enumerate(zip(headers, description_row)):
                        if col_idx < len(description_row):
                            descriptions_by_column[header] = desc
                            raw_wave4_descriptions[header] = desc
                return mapping, descriptions_by_column, raw_wave4_descriptions
            
            # Description row is typically the row before the ImportId row
            description_row_idx = import_row_idx - 1
            if description_row_idx >= 1 and description_row_idx < len(rows):
                description_row = rows[description_row_idx]
            else:
                description_row = [""] * len(headers)
            
            import_row = rows[import_row_idx]
            
            # Parse ImportId values and descriptions
            for col_idx, header in enumerate(headers):
                # Get description
                if col_idx < len(description_row):
                    desc = description_row[col_idx]
                    descriptions_by_column[header] = desc
                    raw_wave4_descriptions[header] = desc
                else:
                    descriptions_by_column[header] = ""
                    raw_wave4_descriptions[header] = ""
                
                # Get ImportId
                if col_idx < len(import_row):
                    cell = import_row[col_idx]
                    if cell.startswith('{') and '"ImportId"' in cell:
                        try:
                            import_data = json.loads(cell)
                            import_id = import_data.get('ImportId', '')
                            if import_id:
                                mapping[header] = import_id
                        except json.JSONDecodeError:
                            continue
                        
        except Exception as e:
            logger.warning(f"Failed to extract ImportId mapping from {benchmark_csv}: {e}")
        
        return mapping, descriptions_by_column, raw_wave4_descriptions
    
    def _extract_importid_mapping(self, benchmark_csv: str) -> Dict[str, str]:
        """Extract ImportId to column name mapping from wave4 CSV (backward compatibility)."""
        mapping, _, _ = self._extract_importid_mapping_with_descriptions(benchmark_csv)
        return mapping
    
    def _load_column_mapping(self, column_mapping_file: Optional[str]) -> Dict[str, str]:
        """Load manual column mapping from file."""
        if not column_mapping_file or not os.path.exists(column_mapping_file):
            return {}
        
        mapping = {}
        try:
            mapping_df = pd.read_csv(column_mapping_file)
            
            # Handle different formats
            if 'wave4_id' in mapping_df.columns and 'input_id' in mapping_df.columns:
                # Format: wave4_id,input_id
                for _, row in mapping_df.iterrows():
                    wave4_col = str(row['wave4_id']).strip()
                    input_col = str(row['input_id']).strip()
                    if wave4_col and input_col and pd.notna(wave4_col) and pd.notna(input_col):
                        mapping[wave4_col] = input_col
            elif 'wave4_column_name' in mapping_df.columns and 'input_column_name' in mapping_df.columns:
                # Format: wave4_column_name,input_column_name
                for _, row in mapping_df.iterrows():
                    wave4_col = str(row['wave4_column_name']).strip()
                    input_col = str(row['input_column_name']).strip()
                    if wave4_col and input_col and pd.notna(wave4_col) and pd.notna(input_col):
                        mapping[wave4_col] = input_col
            elif 'source_id' in mapping_df.columns and 'target_id' in mapping_df.columns:
                # Alternative format: source_id,target_id
                for _, row in mapping_df.iterrows():
                    target_col = str(row['target_id']).strip()
                    source_col = str(row['source_id']).strip()
                    if target_col and source_col and pd.notna(target_col) and pd.notna(source_col):
                        mapping[target_col] = source_col
        except Exception as e:
            logger.warning(f"Failed to load column mapping from {column_mapping_file}: {e}")
        
        return mapping
    
    def _build_formatted_data_v2(self, df: pd.DataFrame, benchmark_columns: List[str], 
                                reverse_mapping: Dict[str, str]) -> Dict[str, List]:
        """
        Build formatted data dictionary following convert_csv_format.py logic.
        
        Args:
            df: Input DataFrame
            benchmark_columns: Column order from benchmark
            reverse_mapping: Maps input column names to wave4 column names
        """
        # Initialize result with benchmark columns
        result_data = {col: [None] * len(df) for col in benchmark_columns}
        additional_data = {}
        
        # Find TWIN_ID column
        twin_id_col = None
        for col in df.columns:
            if col.upper() == 'TWIN_ID':
                twin_id_col = col
                break
        
        if twin_id_col and 'TWIN_ID' in result_data:
            result_data['TWIN_ID'] = df[twin_id_col].tolist()
        
        # Map columns
        for input_col in df.columns:
            if input_col == twin_id_col:
                continue
                
            if input_col in reverse_mapping:
                wave4_col = reverse_mapping[input_col]
                if wave4_col in result_data:
                    result_data[wave4_col] = df[input_col].tolist()
                else:
                    additional_data[wave4_col] = df[input_col].tolist()
            else:
                # Preserve unmapped columns
                if input_col.upper() != 'TWIN_ID':
                    additional_data[input_col] = df[input_col].tolist()
        
        # Combine additional columns
        for col, values in additional_data.items():
            if col not in result_data:
                result_data[col] = values
        
        return result_data
    
    def _build_formatted_data(self, df: pd.DataFrame, benchmark_columns: List[str], 
                            mapping: Dict[str, str]) -> Dict[str, List]:
        """Build formatted data dictionary."""
        formatted_data = {}
        
        for benchmark_col in benchmark_columns:
            if benchmark_col == 'TWIN_ID' and 'TWIN_ID' in df.columns:
                formatted_data[benchmark_col] = df['TWIN_ID'].tolist()
            elif benchmark_col in mapping and mapping[benchmark_col] in df.columns:
                # Use mapped column
                formatted_data[benchmark_col] = df[mapping[benchmark_col]].tolist()
            elif benchmark_col in df.columns:
                # Direct match
                formatted_data[benchmark_col] = df[benchmark_col].tolist()
            else:
                # Column not found - fill with empty values
                formatted_data[benchmark_col] = [''] * len(df)
        
        return formatted_data
        
    def generate_randdollar_breakdown(self, df: pd.DataFrame, wave4_csv: str, 
                                    output_path: str, csv_dir: str = None) -> None:
        """
        Generate detailed breakdown of randDollarString data for pricing analysis
        using the logic from convert_csv_format.py
        
        Args:
            df: DataFrame containing responses (not used in new implementation)
            wave4_csv: Path to wave4 CSV with randDollarString column
            output_path: Path to save breakdown CSV
            csv_dir: Directory containing formatted label CSV files
        """
        try:
            # Determine paths for labeled CSV files
            if csv_dir is None:
                csv_dir = os.path.dirname(output_path)
            
            csv_formatted_label_dir = os.path.join(csv_dir, "csv_formatted_label")
            wave13_labels_path = os.path.join(csv_formatted_label_dir, "responses_wave1_3_label_formatted.csv")
            wave4_labels_path = os.path.join(csv_formatted_label_dir, "responses_wave4_label_formatted.csv")
            llm_labels_path = os.path.join(csv_formatted_label_dir, "responses_llm_imputed_label_formatted.csv")
            
            # Find the randdollar column name
            randdollar_col_name = self._find_randdollar_column(wave4_csv)
            if not randdollar_col_name:
                logger.warning("No randdollar column found in wave4 CSV")
                self._create_empty_breakdown(output_path)
                return
            
            # Determine ImportId row index
            import_row_idx = self._find_import_row_index(wave4_csv)
            
            # Generate the breakdown using the function from convert_csv_format.py
            self._generate_randdollar_details_dataframe(
                wave4_csv_path=wave4_csv,
                twin_id_col_name_in_file="TWIN_ID",
                randdollar_col_name_in_file=randdollar_col_name,
                import_row_idx=import_row_idx,
                output_file_path=output_path,
                wave13_label_formatted_path=wave13_labels_path,
                wave4_label_formatted_path=wave4_labels_path,
                llm_label_formatted_path=llm_labels_path
            )
            
        except Exception as e:
            logger.error(f"Failed to generate randdollar breakdown: {e}")
            self._create_empty_breakdown(output_path)
    
    def _create_empty_breakdown(self, output_path: str):
        """Create empty breakdown file with expected columns."""
        breakdown_df = pd.DataFrame(columns=[
            'TWIN_ID', 'position', 'product_ID', 'price_numeric', 
            'input_column_name', 'wave4_column_name', 
            'response_wave3', 'response_wave4', 'response_llm'
        ])
        breakdown_df.to_csv(output_path, index=False)
    
    def _load_formatted_label_csv_for_lookup(self, file_path):
        """Helper to load a formatted label CSV, setting TWIN_ID as index."""
        if not file_path or not os.path.exists(file_path):
            # Suppress debug message for cleaner output
            pass
            return pd.DataFrame()
        try:
            # Skip the description row (row 1, 0-indexed) when loading data
            df = pd.read_csv(file_path, header=0, skiprows=[1], low_memory=False, 
                           keep_default_na=False, na_filter=False, dtype=str)
            if 'TWIN_ID' in df.columns:
                df.set_index('TWIN_ID', inplace=True)
                return df
            else:
                # TWIN_ID column not found - return without setting index
                pass
                return df
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}. Returning empty DataFrame.")
            return pd.DataFrame()
    
    def _find_randdollar_column(self, wave4_csv: str) -> Optional[str]:
        """Find the randdollar column name in wave4 CSV."""
        try:
            with open(wave4_csv, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                if headers:
                    # Look for common variations of randDollar column names
                    preferred_cols = ["randDollarsString", "randDollarString", "randDollarsString_test"]
                    for col in preferred_cols:
                        if col in headers:
                            logger.info(f"Using randdollar column: '{col}'")
                            return col
            return None
        except Exception as e:
            logger.error(f"Error finding randdollar column: {e}")
            return None
    
    def _find_import_row_index(self, wave4_csv: str) -> int:
        """Find the row index containing ImportId data."""
        try:
            with open(wave4_csv, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i >= 10:  # Check first 10 rows
                        break
                    for cell in row:
                        if 'ImportId' in cell:
                            logger.info(f"Found ImportId row at index {i}")
                            return i
            # Default to row 2 if not found
            return 2
        except Exception as e:
            logger.error(f"Error finding ImportId row: {e}")
            return 2
    
    def _generate_randdollar_details_dataframe(self, wave4_csv_path, twin_id_col_name_in_file, 
                                             randdollar_col_name_in_file, import_row_idx, 
                                             output_file_path, wave13_label_formatted_path, 
                                             wave4_label_formatted_path, llm_label_formatted_path):
        """
        Generate detailed DataFrame from parsing randDollarString for all TWIN_IDs.
        This is adapted from convert_csv_format.py's generate_randdollar_details_dataframe function.
        """
        try:
            # Read the wave4 CSV
            df = pd.read_csv(wave4_csv_path, header=0, low_memory=False, 
                           keep_default_na=False, na_filter=False, dtype=str)
            
            if twin_id_col_name_in_file not in df.columns:
                logger.error(f"TWIN_ID column '{twin_id_col_name_in_file}' not found")
                return
            if randdollar_col_name_in_file not in df.columns:
                logger.error(f"randDollar column '{randdollar_col_name_in_file}' not found")
                return
            
            # Load formatted label CSVs
            df_wave13_labels = self._load_formatted_label_csv_for_lookup(wave13_label_formatted_path)
            df_wave4_labels = self._load_formatted_label_csv_for_lookup(wave4_label_formatted_path)
            df_llm_labels = self._load_formatted_label_csv_for_lookup(llm_label_formatted_path)
            
            all_details = []
            diagnostic_count = 0
            max_diagnostics = 5
            
            for index, row_series in df.iterrows():
                rand_val = row_series.get(randdollar_col_name_in_file, "")
                
                # Skip non-data rows
                if not isinstance(rand_val, str) or not rand_val.strip():
                    continue
                if rand_val.lower() == randdollar_col_name_in_file.lower():
                    continue
                if rand_val.startswith('{') and '"ImportId"' in rand_val:
                    continue
                if index <= import_row_idx and import_row_idx > 0:
                    continue
                
                twin_id = row_series.get(twin_id_col_name_in_file, "")
                
                # Clean up the randDollar string
                if rand_val.startswith('`"""`;'):
                    rand_val = rand_val[len('`"""`;'):]
                elif rand_val.startswith('`"""`'):
                    rand_val = rand_val[len('`"""`'):]
                
                # Remove quotes
                if rand_val.startswith('"""') and rand_val.endswith('"""'):
                    rand_val = rand_val[3:-3]
                elif rand_val.startswith('"') and rand_val.endswith('"'):
                    rand_val = rand_val[1:-1]
                elif rand_val.startswith("'") and rand_val.endswith("'"):
                    rand_val = rand_val[1:-1]
                
                # Print diagnostics for first few rows
                # Commented out diagnostic logging for cleaner output
                # if diagnostic_count < max_diagnostics and index > import_row_idx:
                #     logger.debug(f"Row {index} - TWIN_ID {twin_id}: {rand_val[:100]}...")
                #     diagnostic_count += 1
                
                # Parse items
                items = rand_val.split(';')
                valid_position = 0
                
                for item in items:
                    item = item.strip()
                    if not item or item == '---':
                        continue
                    
                    if ':' in item:
                        parts = item.split(':', 1)
                        product_part = parts[0]
                        price_part = parts[1]
                        
                        # Extract product ID
                        product_match = re.search(r'(\d+)', product_part)
                        if not product_match:
                            continue
                        product_id = int(product_match.group(1))
                        
                        # Extract price
                        price = None
                        price_lower = price_part.strip().lower()
                        if price_lower == 'free' or '$0 (product is free)' in price_lower:
                            price = 0.0
                        else:
                            price_clean = re.sub(r'[^\d.]', '', price_part)
                            if price_clean and price_clean != '.':
                                try:
                                    price = float(price_clean)
                                except ValueError:
                                    # Invalid price - skip silently
                                    pass
                        
                        valid_position += 1
                        if valid_position > 40:  # Limit to 40 items
                            continue
                        
                        # Column names for lookup
                        wave4_col = f"{valid_position}_Q295"
                        input_col = f"QID9_{valid_position}"
                        
                        # Lookup values
                        val_wave13 = ""
                        val_wave4 = ""
                        val_llm = ""
                        
                        if not df_wave13_labels.empty and twin_id in df_wave13_labels.index:
                            if wave4_col in df_wave13_labels.columns:
                                val_wave13 = df_wave13_labels.loc[twin_id, wave4_col]
                        
                        if not df_wave4_labels.empty and twin_id in df_wave4_labels.index:
                            if wave4_col in df_wave4_labels.columns:
                                val_wave4 = df_wave4_labels.loc[twin_id, wave4_col]
                        
                        if not df_llm_labels.empty and twin_id in df_llm_labels.index:
                            if wave4_col in df_llm_labels.columns:
                                val_llm = df_llm_labels.loc[twin_id, wave4_col]
                        
                        all_details.append({
                            'TWIN_ID': twin_id,
                            'position': valid_position,
                            'product_ID': product_id,
                            'price_numeric': price,
                            'input_column_name': input_col,
                            'wave4_column_name': wave4_col,
                            'response_wave3': val_wave13,
                            'response_wave4': val_wave4,
                            'response_llm': val_llm
                        })
            
            if not all_details:
                logger.warning("No randDollarString details generated")
                self._create_empty_breakdown(output_file_path)
                return
            
            # Save the details
            details_df = pd.DataFrame(all_details)
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            details_df.to_csv(output_file_path, index=False, encoding='utf-8')
            logger.info(f"Generated randdollar breakdown with {len(details_df)} rows")
            
        except Exception as e:
            logger.error(f"Error in randdollar generation: {e}")
            self._create_empty_breakdown(output_file_path)
        
    def save_question_mapping(self, output_path: str):
        """Save question mapping to CSV file"""
        if not self.question_types:
            # Suppress this warning as it's not critical
            pass
            return
        
        mapping_data = []
        
        for qid, info in self.question_types.items():
            mapping_data.append({
                'question_id': qid,
                'question_type': info.get('type', ''),
                'question_text': info.get('text', ''),
                'description': info.get('description', '')
            })
            
        if mapping_data:
            pd.DataFrame(mapping_data).to_csv(output_path, index=False)
            logger.info(f"Saved question mapping to {output_path}")


def main():
    """Main entry point for the JSON to CSV converter."""
    parser = argparse.ArgumentParser(
        description="Convert JSON answer blocks to CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --config config.yaml --all                  # Generate all formats
  %(prog)s --config config.yaml --numeric-only         # Only numeric CSV
  %(prog)s --config config.yaml --format --labeled     # Format and label only
        """
    )
    
    # Arguments
    parser.add_argument('--config', required=True, help='Path to configuration YAML file')
    parser.add_argument('--numeric-only', action='store_true', 
                       help='Output only numeric CSV (skip formatting)')
    parser.add_argument('--format', action='store_true',
                       help='Format output to match benchmark CSV')
    parser.add_argument('--labeled', action='store_true',
                       help='Generate labeled version of CSV')
    parser.add_argument('--all', action='store_true',
                       help='Generate all output formats (default)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
        
    # Default to all outputs if no specific format requested
    if not (args.numeric_only or args.format or args.labeled):
        args.all = True
    
    try:    
        # Initialize converter
        converter = JSONToCSVConverter(config)
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        return 1
    
    # Track processed waves
    processed_waves = 0
    last_numeric_df = None
    
    # Process each wave specified in config
    # Use the converter's processed config which has substitutions applied
    waves = converter.config.get('waves', {})
    
    for wave_name, wave_config in waves.items():
        logger.info(f"Processing {wave_name}...")
        
        try:
            # Process wave and get both numeric and text DataFrames
            result = converter.process_wave(wave_name, wave_config)
            
            if result is None or result['numeric'].empty:
                logger.warning(f"No data found for {wave_name}")
                continue
                
            df_numeric = result['numeric']
            df_text = result['text']
            last_numeric_df = df_numeric
            processed_waves += 1
            
            # Save numeric version
            if args.numeric_only or args.all:
                output_path = wave_config.get('output_csv', f"{wave_name}_numeric.csv")
                _save_dataframe(df_numeric, output_path, "numeric CSV")
                
            # Save formatted version
            if (args.format or args.all) and 'benchmark_csv' in converter.config:
                output_path = wave_config.get('output_csv_formatted', f"{wave_name}_formatted.csv")
                df_formatted = converter.format_to_benchmark(
                    df_numeric, 
                    converter.config['benchmark_csv'],
                    converter.config.get('column_mapping')
                )
                _save_dataframe(df_formatted, output_path, "formatted CSV", add_description_row=True, 
                               descriptions_dict=converter.benchmark_descriptions)
                
            # Save labeled version
            if args.labeled or args.all:
                output_path = wave_config.get('output_csv_labeled', f"{wave_name}_labeled.csv")
                # Also format the labeled version if benchmark is available
                if 'benchmark_csv' in converter.config:
                    df_text_formatted = converter.format_to_benchmark(
                        df_text,
                        converter.config['benchmark_csv'],
                        converter.config.get('column_mapping')
                    )
                    _save_dataframe(df_text_formatted, output_path, "labeled CSV", add_description_row=True,
                                   descriptions_dict=converter.benchmark_descriptions)
                else:
                    _save_dataframe(df_text, output_path, "labeled CSV")
                
        except Exception as e:
            logger.error(f"Failed to process {wave_name}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            
    # Save question mapping
    if converter.config.get('save_question_mapping'):
        try:
            mapping_path = converter.config['question_mapping_output']
            os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
            converter.save_question_mapping(mapping_path)
        except Exception as e:
            logger.error(f"Failed to save question mapping: {e}")
        
    # Generate randdollar breakdown if requested
    if converter.config.get('generate_randdollar_breakdown') and 'benchmark_csv' in converter.config:
        if last_numeric_df is not None and not last_numeric_df.empty:
            try:
                # Extract csv_dir from the config
                csv_dir = None
                if 'waves' in converter.config:
                    # Get the output directory from one of the waves
                    for wave_config in converter.config['waves'].values():
                        if 'output_csv' in wave_config:
                            csv_dir = os.path.dirname(wave_config['output_csv'])
                            break
                
                converter.generate_randdollar_breakdown(
                    last_numeric_df,
                    converter.config['benchmark_csv'],
                    converter.config.get('randdollar_output', 'randdollar_breakdown.csv'),
                    csv_dir=csv_dir
                )
            except Exception as e:
                logger.error(f"Failed to generate randdollar breakdown: {e}")
    
    if processed_waves > 0:         
        logger.info(f"Conversion complete! Processed {processed_waves} waves.")
        return 0
    else:
        logger.warning("No waves were successfully processed.")
        return 1


def _save_dataframe(df: pd.DataFrame, output_path: str, description: str, add_description_row: bool = False, 
                    descriptions_dict: Optional[Dict[str, str]] = None):
    """Helper function to save a DataFrame with proper error handling."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if add_description_row:
            # Save with description row like convert_csv_format.py does
            headers = list(df.columns)
            # Use provided descriptions or empty strings
            if descriptions_dict:
                description_row = [descriptions_dict.get(h, "") for h in headers]
            else:
                description_row = [""] * len(headers)
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerow(description_row)
                for _, data_row in df.iterrows():
                    # Convert NaN to empty string
                    processed_row = ["" if pd.isna(val) or (isinstance(val, float) and np.isnan(val)) else val for val in data_row]
                    writer.writerow(processed_row)
        else:
            df.to_csv(output_path, index=False)
            
        logger.info(f"Saved {description} to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save {description} to {output_path}: {e}")

if __name__ == "__main__":
    import sys
    sys.exit(main())