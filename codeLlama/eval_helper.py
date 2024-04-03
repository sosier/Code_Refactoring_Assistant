from eval import DATA, bulk_evaluate
import csv

mbpp = DATA["mbpp"]  # train, validation, and test
humaneval = DATA["openai_humaneval"]  # test only

csv_file_path = 'humaneval_parsed_output.csv'
humaneval_output = []
with open(csv_file_path, 'r') as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        humaneval_output.append(row['parsed_output'])

humaneval_results = bulk_evaluate(
    dataset="openai_humaneval",
    split="test",
    code=humaneval_output, # one for each task in HumanEval test
    # Run in parallel using 4 cores
    # Entering None will use all cores on your machine:
    num_processes=4
)

print(humaneval_results)