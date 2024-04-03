# 'Make this python function implementation better and WITHOUT any explanation. Just write the function implementation, and do not change any function definitions. Do not write anything but code, starting with the same function definition: '

# Import libraries
import requests
import json
from eval import DATA, evaluate, bulk_evaluate
import re
import csv

# Define csv writer - results
def write_to_csv(prompt, deepseek_output, result, filename):
    with open(filename, 'a', newline='') as csvfile:  # Use 'a' mode for appending
        fieldnames = ['prompt', 'deepseek_output', 'result']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # If the file is empty, write header
        if csvfile.tell() == 0:
            writer.writeheader()
        
        writer.writerow({'prompt': prompt, 'deepseek_output': deepseek_output, 'result': result})

# Define csv writer - no results
def write_to_csv_2(prompt, deepseek_output, filename):
    with open(filename, 'a', newline='') as csvfile:  # Use 'a' mode for appending
        fieldnames = ['prompt', 'deepseek_output']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # If the file is empty, write header
        if csvfile.tell() == 0:
            writer.writeheader()
        
        writer.writerow({'prompt': prompt, 'deepseek_output': deepseek_output})


# Get all data
mbpp = DATA["mbpp"]  # train, validation, and test
humaneval = DATA["openai_humaneval"]  # test only

# Get number of prompts
num_prompts = len(humaneval["test"])

# Keep track of prompts and generated code
prompts = []
LIST_OF_CODE_SNIPPETS = []

# Set the system instruction
system_instruction = "Refactor the given Python program to a more readable, efficient, and maintainable one. You can assume that the given program is semantically correct. Do not change the external behavior of the program, and keep the syntactic and semantic correctness. Python programs should be in a code block. Do not explain anything in natural language."

# For each prompt
for i in range(0, num_prompts):
    print('Working on prompt number', i, 'of', num_prompts)

    # Get the prompt
    prompt = humaneval["test"][i]["prompt"] + humaneval["test"][i]["canonical_solution"]

    # Add prompt to list of prompts
    prompts.append(prompt)

    # Set up URL for POST request
    url = "https://api.deepseek.com/v1/chat/completions"

    # Create payload (what we send to deepseek for answer)
    payload = json.dumps({
    "messages": [
        {
        "content": system_instruction + prompt,
        "role": "system"
        },
        # {
        #   "content": "Hi",
        #   "role": "user"
        # }
    ],
    "model": "deepseek-coder",
    "frequency_penalty": 0,
    "max_tokens": 2048,
    "presence_penalty": 0,
    "stop": None,
    "stream": False,
    "temperature": 1,
    "top_p": 1
    })
    headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': 'Bearer sk-47c911d337d54a53a9a5ada234244bb2'
    }

    # Get generate code
    response = requests.request("POST", url, headers=headers, data=payload)
    response_data = json.loads(response.text)
    content = response_data['choices'][0]['message']['content']

    # # Evaluate whether or not the generated code is correct
    # print('Prompt was:', prompt)
    # print('Response was:', content)
    # print('Correct?:', evaluate(
    #     dataset="openai_humaneval",
    #     split="test",
    #     task_id=0,
    #     code=content
    # )['result'])

    # Add code to the list of generated code
    LIST_OF_CODE_SNIPPETS.append(content)

print('Done with getting content!')

# Write each pair without its result to CSV
for prompt, deepseek_output in zip(prompts, LIST_OF_CODE_SNIPPETS):
    write_to_csv_2(prompt, deepseek_output, 'without_results_output.csv')

print('Done with writing non-result to CSV!')

# Bulk Evaluate instead
results = bulk_evaluate(
    dataset="openai_humaneval",
    split="test",
    code=LIST_OF_CODE_SNIPPETS, # one for each task in HumanEval test
    # Run in parallel using 4 cores
    # Entering None will use all cores on your machine:
    num_processes=8
)

temp = []
for i in results:
    if i == 'ERROR':
        temp.append('False')
    else:
        temp.append(i['passed_tests'])
results = temp
print('Results:\n', results)


# Write each pair with its result to a CSV
for prompt, deepseek_output, result in zip(prompts, LIST_OF_CODE_SNIPPETS, results):
    write_to_csv(prompt, deepseek_output, result, 'results_output.csv')
