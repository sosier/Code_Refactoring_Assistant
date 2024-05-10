
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    ResNetModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    CodeLlamaTokenizer
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
import accelerate
import bitsandbytes
from transformers.modeling_outputs import SequenceClassifierOutput
import json
import time
import csv
import re
import pandas as pd
import shutil
from datasets import load_dataset

from utils import * 


# Function to extract the rule number from the output text
def extract_rule(output):
    # Extracting text after '### Response'
    response_text = output.split('[/INST]')[-1]
    # Finding the first number in the specified range
    found_numbers = re.findall(r'\b(?:1[0-6]|[1-9])\b', response_text)
    return found_numbers[0] if found_numbers else None

def parsed_csv(file_path, results):
  keys = ['prompt', 'output']
  with open(file_path, 'w', newline='') as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames=keys)
      writer.writeheader()
      for result in results:
          writer.writerow(result)

# def parse_results(data):
#     results = []
#     for output in data:
#         prompt = output['prompt']
#         output = output['output']
#         rule = output['rule']
#         result_dict = {'prompt': prompt,'output': output, 'rule': rule}
#         results.append(result_dict)
#     return results


def parsed_results_csv(file_path, results):
  keys = ['prompt', 'output', 'rule']
  with open(file_path, 'w', newline='') as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames=keys)
      writer.writeheader()
      for result in results:
          writer.writerow(result)

def extract_python_function(text):
    # Pattern to capture a Python function
    # - Starts with 'def' followed by any characters (the function name and arguments)
    # - Includes all indented lines after the 'def' line, allowing for empty lines
    # - Assumes an indentation of 4 spaces, but can be adjusted
    pattern = r"(def .+?:\n(?: {4,}.*\n|\n)*)"

    # Find all matches
    matches = re.findall(pattern, text, re.MULTILINE)

    return matches

def extract_after_substring(full_string, substring):
    index = full_string.find(substring)
    if index != -1:
        return full_string[index + len(substring):]
    else:
        return ""

def extract_between_markers(full_string, start_marker, end_marker):
    start_index = full_string.find(start_marker)
    end_index = full_string.find(end_marker, start_index + len(start_marker))
    if start_index != -1 and end_index != -1:
        return full_string[start_index + len(start_marker):end_index].strip()
    else:
        return ""

def extract_python_code(full_string):
    extracted_string = extract_after_substring(full_string, "[/INST]")
    out = extract_between_markers(extracted_string, "```", "```")
    if out != '':
      return out
    else:
      return extract_python_function(extracted_string)
