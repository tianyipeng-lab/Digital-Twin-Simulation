import json
import os
import re
import argparse
import glob
import sys
from convert_persona_to_text import format_question_text
from tqdm import tqdm
GENERAL_LLM_INSTRUCTION = """Please answer the following questions as if you were taking this survey. The expected output is a JSON object and the format is provided in the end. 
---

"""

def strip_html(text: str) -> str:
    """Strip HTML tags from text and normalize whitespace"""
    if not text:
        return ""
    text = re.sub(r'<[^>]*>', ' ', text)
    text = text.replace('&nbsp;', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def format_instructions(include_reasoning=True):
    FORMAT_INSTRUCTIONS = """
### Format Instructions:
In order to facilitate the postprocessing, you should generate string that can be parsed into a valid JSON object with the following format:
{
    "Q1": {
    "Question Type": "XX",
    "Reasoning": "Concise reasoning for the answer",
    "Answers": {
        see below
    } 
    },
    "Q2": {
    "Question Type": "XX",
    "Reasoning": "Concise reasoning for the answer",
    "Answers": {
        see below
    } 
    },
    ....
}

The question type can be one of the following:
1. Matrix 
For Matrix questions, the answers should include two lists, one for the selected positions and one for the selected texts.
For example, 

Would you support or oppose...
Question Type: Matrix
Options:
1 = Strongly oppose
2 = Somewhat oppose
3 = Neither oppose nor support
4 = Somewhat support
5 = Strongly support
1. Placing a tax on carbon emissions?
Answer: [Masked]
2. Ensuring 40% of all new clean energy infrastructure development spending goes to low-income communities?
Answer: [Masked]
3. Federal investments to ensure a carbon-pollution free electricity sector by 2035?
Answer: [Masked]
4. A 'Medicare for All' system in which all Americans would get healthcare from a government-run plan?
Answer: [Masked]
5. A 'public option', which would allow Americans to buy into a government-run healthcare plan if they choose to do so?
Answer: [Masked]
6. Immigration reforms that would provide a path to U.S. citizenship for undocumented immigrants currently in the United States?
Answer: [Masked]
7. A law that requires companies to provide paid family leave for parents?
Answer: [Masked]
8. A 2% tax on the assets of individuals with a net worth of more than $50 million?
Answer: [Masked]
9. Increasing deportations for those in the US illegally?
Answer: [Masked]
10. Offering seniors healthcare vouchers to purchase private healthcare plans in place of traditional medicare coverage?
Answer: [Masked] 

Examples Answers:
{
    "Answers": {
        "SelectedByPosition": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "SelectedText": ["Strongly oppose", "Somewhat oppose", "Neither oppose nor support", "Somewhat support", "Strongly support", "Strongly oppose", "Somewhat oppose", "Neither oppose nor support", "Somewhat support", "Strongly support"]
    }
}

2. Single Choice
For Single Choice questions, the answers should include the selected position and the selected text.
For example,

Imagine that the U.S. is preparing for the outbreak of an unusual disease, which is expected to kill 600 people. Two alternative programs to combat the disease have been proposed. Assume that the exact scientific estimate of the consequences of the programs are as follows: If Program A is adopted, 400 people will die. If Program B is adopted, there is 1/3 probability that nobody people will die, and 2/3 probability that 600 people be die. Which of the two programs would you favor?
Question Type: Single Choice
Options:
1 - I strongly favor program A
2 - I favor program A
3 - I slightly favor program A
4 - I slightly favor program B
5 - I favor program B
6 - I strongly favor program B
Answer: [Masked]

Examples Answers:
{
    "Answers": {
        "SelectedByPosition": 1,
        "SelectedText": "I strongly favor program A"
    }
}
    
3. Slider 
For Slider questions, the answers should simply include the a list of answers. 
For example,

A panel of psychologist have interviewed and administered personality tests to 30 engineers and 70 lawyers, all successful in their respective fields. On the basis of this information, thumbnail descriptions of the 30 engineers and 70 lawyers have been written. Below is one description, chosen at random from the 100 available descriptions. Jack is a 45-year-old man. He is married and has four children. He is generally conservative, careful, and ambitious. He shows no interest in political and social issues and spends most of his free time on his many hobbies which include home carpentry, sailing, and mathematical puzzles. The probability that Jack is one of the 30 engineers in the sample of 100 is ___%. Please indicate the probability on a scale from 0 to 100.
Question Type: Slider
1. [No Statement Needed]
Answer: [Masked]

Examples Answers:
{
    "Answers": {
        "Values": ["55"],
    }
}

4. Text Entry
For Text Entry questions, the answers should simply include the text.
For example,

Question Type: Text Entry
Answer: [Masked]

Examples Answers:
{
    "Answers": {
        "Text": "70"
    }
}
"""
    if not include_reasoning:
        FORMAT_INSTRUCTIONS = """
### Format Instructions:
In order to facilitate the postprocessing, you should generate string that can be parsed into a valid JSON object with the following format:
{
    "Q1": {
    "Question Type": "XX",
    "Answers": {
        see below
    } 
    },
    "Q2": {
    "Question Type": "XX",
    "Answers": {
        see below
    } 
    },
    ....
}

The question type can be one of the following:
1. Matrix 
For Matrix questions, the answers should include two lists, one for the selected positions and one for the selected texts.
For example, 

Would you support or oppose...
Question Type: Matrix
Options:
1 = Strongly oppose
2 = Somewhat oppose
3 = Neither oppose nor support
4 = Somewhat support
5 = Strongly support
1. Placing a tax on carbon emissions?
Answer: [Masked]
2. Ensuring 40% of all new clean energy infrastructure development spending goes to low-income communities?
Answer: [Masked]
3. Federal investments to ensure a carbon-pollution free electricity sector by 2035?
Answer: [Masked]
4. A 'Medicare for All' system in which all Americans would get healthcare from a government-run plan?
Answer: [Masked]
5. A 'public option', which would allow Americans to buy into a government-run healthcare plan if they choose to do so?
Answer: [Masked]
6. Immigration reforms that would provide a path to U.S. citizenship for undocumented immigrants currently in the United States?
Answer: [Masked]
7. A law that requires companies to provide paid family leave for parents?
Answer: [Masked]
8. A 2% tax on the assets of individuals with a net worth of more than $50 million?
Answer: [Masked]
9. Increasing deportations for those in the US illegally?
Answer: [Masked]
10. Offering seniors healthcare vouchers to purchase private healthcare plans in place of traditional medicare coverage?
Answer: [Masked] 

Examples Answers:
{
    "Answers": {
        "SelectedByPosition": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        "SelectedText": ["Strongly oppose", "Somewhat oppose", "Neither oppose nor support", "Somewhat support", "Strongly support", "Strongly oppose", "Somewhat oppose", "Neither oppose nor support", "Somewhat support", "Strongly support"]
    }
}

2. Single Choice
For Single Choice questions, the answers should include the selected position and the selected text.
For example,

Imagine that the U.S. is preparing for the outbreak of an unusual disease, which is expected to kill 600 people. Two alternative programs to combat the disease have been proposed. Assume that the exact scientific estimate of the consequences of the programs are as follows: If Program A is adopted, 400 people will die. If Program B is adopted, there is 1/3 probability that nobody people will die, and 2/3 probability that 600 people be die. Which of the two programs would you favor?
Question Type: Single Choice
Options:
1 - I strongly favor program A
2 - I favor program A
3 - I slightly favor program A
4 - I slightly favor program B
5 - I favor program B
6 - I strongly favor program B
Answer: [Masked]

Examples Answers:
{
    "Answers": {
        "SelectedByPosition": 1,
        "SelectedText": "I strongly favor program A"
    }
}
    
3. Slider 
For Slider questions, the answers should simply include the a list of answers. 
For example,

A panel of psychologist have interviewed and administered personality tests to 30 engineers and 70 lawyers, all successful in their respective fields. On the basis of this information, thumbnail descriptions of the 30 engineers and 70 lawyers have been written. Below is one description, chosen at random from the 100 available descriptions. Jack is a 45-year-old man. He is married and has four children. He is generally conservative, careful, and ambitious. He shows no interest in political and social issues and spends most of his free time on his many hobbies which include home carpentry, sailing, and mathematical puzzles. The probability that Jack is one of the 30 engineers in the sample of 100 is ___%. Please indicate the probability on a scale from 0 to 100.
Question Type: Slider
1. [No Statement Needed]
Answer: [Masked]

Examples Answers:
{
    "Answers": {
        "Values": ["55"],
    }
}

4. Text Entry
For Text Entry questions, the answers should simply include the text.
For example,

Question Type: Text Entry
Answer: [Masked]

Examples Answers:
{
    "Answers": {
        "Text": "70"
    }
}
"""
    return FORMAT_INSTRUCTIONS

def process_json_file(input_file, output_dir, include_reasoning=True):
    """Process a single JSON file and save the result to output_dir"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # Check if content is wrapped in quotes and unescape if needed
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1].replace('\\"', '"').replace('\\\\', '\\')
            data = json.loads(content)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {input_file}: {e}")
        return
    
    # Extract PID from input filename
    input_basename = os.path.basename(input_file)
    pid_match = re.search(r'(pid_\d+)', input_basename)
    output_filename = f"{pid_match.group(1)}.txt" if pid_match else f"{os.path.splitext(input_basename)[0]}.txt"
    
    top_level_block_content_lines = [GENERAL_LLM_INSTRUCTION] # Start with general instruction

    count = 0
    for element in data: ## data is a list of blocks
        element_type = element.get("ElementType")
        if element_type == "Block":
            if element.get("Questions"): # Check if block has questions
                for question in element.get("Questions", []):
                    if question.get("QuestionType") == "DB":
                        top_level_block_content_lines.append(format_question_text(question, with_answers=False))
                    else:
                        count += 1
                        top_level_block_content_lines.append(f"Q{count}:\n" + format_question_text(question, with_answers=False))

    top_level_block_content_lines.append(format_instructions(include_reasoning))

    main_file_path = os.path.join(output_dir, output_filename)
    try:
        with open(main_file_path, 'w', encoding='utf-8') as mf:
            mf.write("".join(top_level_block_content_lines))
        return True
    except Exception as e:
        print(f"Error writing main prompts file {main_file_path}: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON survey to LLM prompt files, saving branches to separate files.")
    parser.add_argument("--input", help="Input folder containing JSON files", default="./data/mega_persona_json/answer_blocks")
    parser.add_argument("--output_dir", help="Output directory to save LLM prompt text files", default="./text_simulation/text_questions")
    parser.add_argument("--include_reasoning", action="store_true", help="Include reasoning in the output format")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Check if input is a directory or a file
    if os.path.isdir(args.input):
        # Find all files ending with _wave4_Q_wave4_A.json
        pattern = os.path.join(args.input, "*_wave4_Q_wave4_A.json")
        input_files = glob.glob(pattern)
        
        if not input_files:
            print(f"No files matching '*_wave4_Q_wave4_A.json' found in {args.input}")
            sys.exit(1)
            
        for input_file in tqdm(input_files):
            process_json_file(input_file, args.output_dir, args.include_reasoning)
    else:
        process_json_file(args.input, args.output_dir, args.include_reasoning) 