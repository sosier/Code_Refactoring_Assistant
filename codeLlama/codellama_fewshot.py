# -*- coding: utf-8 -*-

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


device = torch.device('cuda')

mbpp = load_dataset("mbpp")  # train, validation, and test
humaneval = load_dataset("openai_humaneval")  # test only

model_id = "codellama/CodeLlama-7b-Instruct-hf"
quantization_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto",
)

rule_dict = {
    1: "Use a formatted string",
    2: "Use a built-in (i.e., radians) function",
    3: "Use a logical operator instead of a nested if",
    4: "Use a for-loop instead of a while-loop",
    5: "Use list comprehension instead of a for-loop",
    6: "Use the map function instead of list comprehension",
    7: "Use a throwaway variable",
    8: "Use the enumerate function instead of the range function",
    9: "Use the zip function instead of the range function",
    10: "Use a ternary operator instead of an if-branch",
    11: "Merge repeated ifs",
    12: "Merge dictionary assignments",
    13: "Remove unnecessary calls to dict.items()",
    14: "Remove str() from calls to print()",
    15: "Flatten nested try",
    16: "Convert any to in."
}

def select_rules(prompts):
  output_texts = []
  for p in prompts:
    code_block = p
    system_instruction = "You are an expert programmer."
    rule_prompt = (
      f"Given the code below, pick the most suitable refactoring rule from the list of rules copied below.\n"
      f"Code: {code_block}\n"
      f"You MUST follow these requirements: \n"
      f"1) Only pick one rule and 2) Only output the number of the rule you have chosen\n"
      "List of Rules:\n"
      "1. Use a formatted string.\n"
      "2. Use a built-in (i.e., radians) function.\n"
      "3. Use a logical operator instead of a nested if.\n"
      "4. Use a for-loop instead of a while-loop.\n"
      "5. Use list comprehension instead of a for-loop.\n"
      "6. Use the map function instead of list comprehension."
      "7. Use a throwaway variable.\n"
      "8. Use the enumerate function instead of the range function.\n"
      "9. Use the zip function instead of the range function.\n"
      "10. Use a ternary operator instead of an if-branch.\n"
      " 11. Merge repeated ifs.\n"
      " 12. Merge dictionary assignments.\n"
      " 13. Remove unnecessary calls to dict.items().\n"
      " 14. Remove str() from calls to print().\n"
      " 15. Flatten nested try.\n"
      " 16. Convert any to in."
    )

    PROMPT = "<s>[INST] <<SYS>>\\n{system_instruction}\\n<</SYS>>\\n\\n{user}[/INST]"
    prompt = PROMPT.format(system_instruction = system_instruction, user=rule_prompt)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    output = model.generate(
        inputs["input_ids"],
        max_new_tokens=2048,
        do_sample=True,
        top_p=0.9,
        temperature=0.1,
        pad_token_id=tokenizer.eos_token_id
    )
    output = output[0].to(device)
    output_text = tokenizer.decode(output)
    # print(output_text)
    output_dict = {'prompt': p, 'output': output_text}
    output_texts.append(output_dict)
    print('run', end = '', flush = True)
  return output_texts

def generate_output_with_rule(prompts):
  output_texts = []
  total = len(prompts)
  count = 0
  for prompt, rule in prompts.items():
    count += 1
    print(count/total, end = ' ', flush = True)
    match = re.search(r'def (\w+)\(', prompt)
    if match:
      function_name = match.group(1)
    else:
      function_name=''
    taken_out = 'for chosen rule in chosen_rules'
    rule_prompt = (
      f"""
      Please refactor the following Python program to a more readable, efficient, and maintainable one:
      - The given program is correct but needs improvement
      - MAKE SURE TO follow these given refactoring rules: {rule_dict[rule]} 
      - DO NOT change the name of the program
      - DO NOT change the input or output behavior of the program (e.g., number of inputs / outputs, input / output types, etc.)
      - Put your response in a markdown code block
      - Respond with only the code block
      - Don't explain the changes made


      Again, do not change the name of the function in any way. The function name should remain "{function_name}".

      \nCode: \n{prompt}\n\n
      """.strip()
    )

    system_instruction = "You are an expert programmer."
    PROMPT = "<s>[INST] <<SYS>>\\n{system_instruction}\\n<</SYS>>\\n\\n{user}[/INST]"
    prompt = PROMPT.format(system_instruction = system_instruction, user=rule_prompt)
    # print(prompt)
    # break

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    output = model.generate(
        inputs["input_ids"],
        max_new_tokens=2048,
        do_sample=True,
        top_p=0.9,
        temperature=0.1,
        pad_token_id=tokenizer.eos_token_id
    )
    output = output[0].to(device)
    output_text = tokenizer.decode(output)
    # print(output_text)
    output_dict = {'prompt': prompt, 'output': output_text}
    output_texts.append(output_dict)
  return output_texts
 
def mbpp_rules():
    print(device)

    mbpp_prompts=[]
    for i in mbpp["test"]:
      concat_prompt = i['code']
      mbpp_prompts.append(concat_prompt)

    mbpp_outputs = select_rules(mbpp_prompts)
    parsed_csv('./mbpp_rules.csv', mbpp_outputs)


    # print("\n\nMBPP\n")
    # mbpp_prompts=[]
    # for idx,i in enumerate(mbpp["test"]):
    #   concat_prompt = i['text'] + '\n\n'+ i['code']
    #   mbpp_prompts.append([idx,concat_prompt])

    # mbpp_outputs = generate_output(mbpp_prompts)
    # parsed_csv('/content/mbpp_parsed.csv', mbpp_outputs)

def humaneval_rules():
    humaneval_prompts = []
    for i in humaneval["test"]:
      concat_prompt = i['prompt'] + i['canonical_solution']
      humaneval_prompts.append(concat_prompt)

    humaneval_outputs = select_rules(humaneval_prompts)
    #writes to 'prompt', 'output' csv 
    parsed_csv('./humaneval_rules.csv', humaneval_outputs)

    # print("HUMANEVAL\n")
    # humaneval_prompts = []
    # for idx,i in enumerate(humaneval["test"]):
    #   concat_prompt = i['prompt'] + i['canonical_solution']
    #   humaneval_prompts.append([idx,concat_prompt])

    # humaneval_outputs = generate_output(humaneval_prompts)
    # parsed_csv('/content/humaneval_parsed.csv', humaneval_outputs)

''' old code to parse for function and add back to csv'''
# csv_file_path = 'humaneval_parsed.csv'

# humaneval_output_values = []
# with open(csv_file_path, 'r') as file:
#     csv_reader = csv.DictReader(file)
#     for row in csv_reader:
#         humaneval_output_values.append(row['output'])

# humaneval_snippets=[]
# for i in humaneval_output_values:
#   humaneval_snippets.append(extract_python_code(i))

# # Step 1: Read the CSV file into a DataFrame
# csv_file_path = 'humaneval_parsed.csv'
# parsed ='humaneval_parsed_output.csv'
# df = pd.read_csv(csv_file_path)

# # Step 3: Add the new "City" column to the DataFrame
# df['parsed_output'] = humaneval_snippets

# # Step 4: Write the DataFrame back to the CSV file
# df.to_csv(parsed, index=False)
    

# csv_file_path = 'mbpp_parsed.csv'

# mbpp_output_values = []
# with open(csv_file_path, 'r') as file:
#     csv_reader = csv.DictReader(file)
#     for row in csv_reader:
#         mbpp_output_values.append(row['output'])

# mbpp_snippets=[]
# for i in mbpp_output_values:
#   mbpp_snippets.append(extract_python_code(i))
# print(mbpp_snippets)


# # Step 1: Read the CSV file into a DataFrame
# csv_file_path = 'mbpp_parsed.csv'
# parsed ='mbpp_parsed_output.csv'
# df = pd.read_csv(csv_file_path)

# # Step 3: Add the new "City" column to the DataFrame
# df['parsed_output'] = mbpp_snippets

# # Step 4: Write the DataFrame back to the CSV file
# df.to_csv(parsed, index=False)

'''
load the rules data and extract the rule
'''
def rule_filter():
    # Load the CSV file
    df = pd.read_csv('./mbpp_rules.csv')

    # Apply the function to the 'output' column
    df['rule'] = df['output'].apply(extract_rule)

    # Create a new DataFrame with the required columns including the original output
    new_df = df[['prompt', 'output', 'rule']]

    # Save the new DataFrame to a CSV file
    new_df.to_csv('./filtered_mbpp_rules_with_output.csv', index=False)

    print("New CSV file created with prompts, original outputs, and their corresponding rules.")


    # Load the CSV file
    df = pd.read_csv('./humaneval_rules.csv')

    # Apply the function to the 'output' column
    df['rule'] = df['output'].apply(extract_rule)

    # Create a new DataFrame with the required columns including the original output
    new_df = df[['prompt', 'output', 'rule']]

    # Save the new DataFrame to a CSV file
    new_df.to_csv('./filtered_humaneval_rules_with_output.csv', index=False)

    print("New CSV file created for humaneval_rules with prompts, original outputs, and their corresponding rules.")

"""#refactor_with_rules.ipynb

"""

def refactor_with_rule(humaneval_run=False, mbpp_run=False):
   
  # # Paths to your files
  mbpp_file_path = './filtered_mbpp_rules_with_output.csv'
  humaneval_file_path = './filtered_humaneval_rules_with_output.csv'

  # Load the CSV files
  mbpp_df = pd.read_csv(mbpp_file_path)
  humaneval_df = pd.read_csv(humaneval_file_path)

  # Initialize dictionaries to store prompts and rules
  mbpp_prompts_rules = {}
  humaneval_prompts_rules = {}

  # Iterate through the MBPP DataFrame and store data
  for index, row in mbpp_df.iterrows():
      mbpp_prompts_rules[row['prompt']] = row['rule']

  # Iterate through the HumanEval DataFrame and store data
  for index, row in humaneval_df.iterrows():
      humaneval_prompts_rules[row['prompt']] = row['rule']

  # If you want to see what's in the dictionaries, you can print them
  # print("MBPP Prompts and Rules:", mbpp_prompts_rules)
  # print("HumanEval Prompts and Rules:", humaneval_prompts_rules)
  if humaneval_run:
    humaneval_results=generate_output_with_rule(humaneval_prompts_rules)

    parsed_results_csv('./humaneval_rules_refactored.csv', humaneval_results)
  if mbpp_run: 
    mbpp_results=generate_output_with_rule(mbpp_prompts_rules)

    parsed_results_csv('./mbpp_rules_refactored.csv', mbpp_results)

refactor_with_rule(humaneval_run = False, mbpp_run = True)

# """#eval-starcoder-fewshot.ipynb"""

# import csv

# def parsed_csv(file_path, results):
#   keys = ['error', 'dataset', 'split', 'task_id', 'result', 'avg_test_time', 'passed_tests', 'compiled', 'loc', 'lloc', 'sloc', 'comments', 'multi', 'blank', 'single_comments', 'CC', 'h1', 'h2', 'N1', 'N2', 'vocabulary', 'length', 'calculated_length', 'volume', 'difficulty', 'effort', 'time', 'bugs', 'MI']

#   with open(file_path, 'w', newline='') as csvfile:
#       writer = csv.DictWriter(csvfile, fieldnames=keys)
#       writer.writeheader()
#       for result in results:
#           # print(result)
#           if type(result) == str:
#             writer.writerow({"error": result})
#           else:
#             writer.writerow(result)


# csv_file_path = 'humaneval_rules_refactored.csv'
# humaneval_output_values = []
# with open(csv_file_path, 'r') as file:
#     csv_reader = csv.DictReader(file)
#     for row in csv_reader:
#         humaneval_output_values.append(row['output'])

# humaneval_snippets=[]
# for i in humaneval_output_values:
#   humaneval_snippets.append(extract_after_substring(i, "### Response"))

# humaneval_bulk_output = bulk_evaluate(
#     dataset="openai_humaneval",
#     split="test",
#     code=humaneval_snippets, # one for each task in HumanEval test
# )

# parsed_csv('./humaneval_metrics.csv', humaneval_bulk_output)

# csv_file_path = 'mbpp_rules_refactored.csv'
# mbpp_output_values = []
# with open(csv_file_path, 'r') as file:
#     csv_reader = csv.DictReader(file)
#     for row in csv_reader:
#         mbpp_output_values.append(row['output'])

# mbpp_snippets=[]
# for i in mbpp_output_values:
#   mbpp_snippets.append(extract_after_substring(i, "### Response"))

# mbpp_bulk_output = bulk_evaluate(
#     dataset="mbpp",
#     split="test",
#     code=mbpp_snippets,
# )

# parsed_csv('./mbpp_metrics.csv', mbpp_bulk_output)