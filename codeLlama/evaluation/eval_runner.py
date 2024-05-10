import csv 
import pandas as pd

from eval import DATA, evaluate, bulk_evaluate
from eval_utils import *


mbpp = DATA["mbpp"]  # train, validation, and test
humaneval = DATA["openai_humaneval"]  # test only

def test():
    '''
    Function to test the cloud evaluation sandbox machine
    '''

    canonical_solution = humaneval["test"][0]["prompt"] + humaneval["test"][0]["canonical_solution"]
    print(type(canonical_solution))
    print(evaluate(
        dataset="openai_humaneval",
        split="test",
        task_id=0,
        code=canonical_solution  # or alternatively YOUR_CODE_HERE
    ))

    canonical_solution = mbpp["test"][0]["code"]
    print(evaluate(
        dataset="mbpp",
        split="test",
        task_id=0,
        code=canonical_solution  # or alternatively YOUR_CODE_HERE
    ))

'''

Humaneval output testing 

'''
def humaneval_few_eval():
    csv_file_path = 'humaneval_rules_refactored.csv'
    humaneval_output_values = []
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            humaneval_output_values.append(row['output'])
    
    humaneval_snippets=[]
    for i in humaneval_output_values:
        extracted = extract_python_code(i)
        if len(extracted) < 1:
            humaneval_snippets.append(extract_after_substring(i, "[/INST]"))
        else:
            humaneval_snippets.append(extracted)

        
    humaneval_bulk_output = bulk_evaluate(
        dataset="openai_humaneval",
        split="test",
        code=humaneval_snippets, # one for each task in HumanEval test
    )


    parsed_eval_csv('./humaneval_few_metrics.csv', humaneval_bulk_output)

def humaneval_zero_eval():
    csv_file_path = 'humaneval_zero.csv'
    humaneval_output_values = []
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            humaneval_output_values.append(row['output'])
    
    humaneval_snippets=[]
    for i in humaneval_output_values:
        extracted = extract_python_code(i)
        if len(extracted) < 1:
            humaneval_snippets.append(extract_after_substring(i, "[/INST]"))
        else:
            humaneval_snippets.append(extracted)

        
    humaneval_bulk_output = bulk_evaluate(
        dataset="openai_humaneval",
        split="test",
        code=humaneval_snippets, # one for each task in HumanEval test
    )


    parsed_eval_csv('./humaneval_zero_metrics.csv', humaneval_bulk_output)

'''

MBPP output testing

'''
def mbpp_few_eval():
    csv_file_path = 'mbpp_rules_refactored.csv'
    mbpp_output_values = []
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            mbpp_output_values.append(row['output'])


    mbpp_snippets=[]
    for i in mbpp_output_values:
        extracted = extract_python_code(i)
        if len(extracted) < 1:
            mbpp_snippets.append(extract_after_substring(i, "[/INST]"))
        else:
            mbpp_snippets.append(extracted)
        
    
    mbpp_bulk_output = bulk_evaluate(
        dataset="mbpp",
        split="test",
        code=mbpp_snippets,
    )

    parsed_eval_csv('./mbpp_few_metrics.csv', mbpp_bulk_output)

def mbpp_zero_eval():
    csv_file_path = 'mbpp_zero.csv'
    mbpp_output_values = []
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            mbpp_output_values.append(row['output'])

    mbpp_snippets=[]
    for i in mbpp_output_values:
        extracted = extract_python_code(i)
        if len(extracted) < 1:
            mbpp_snippets.append(extract_after_substring(i, "[/INST]"))
        else:
            mbpp_snippets.append(extracted)
        
    
    mbpp_bulk_output = bulk_evaluate(
        dataset="mbpp",
        split="test",
        code=mbpp_snippets,
    )

    parsed_eval_csv('./mbpp_zero_metrics.csv', mbpp_bulk_output)

def model_format():
    #changes csv format for results collation
    for file_path in ["./mbpp_few_metrics.csv", "./humaneval_few_metrics.csv","./mbpp_zero_metrics.csv", "./humaneval_zero_metrics.csv"]:
        # Load the CSV file
        df = pd.read_csv(file_path)
        print(file_path)

        # Add a new column with a constant value
        df['model'] = "codeLLama_fewshot"


        # Save the updated DataFrame back to the CSV file
        df.to_csv(file_path, index=False)
