import csv 
import re

"""
Utility functions for evaluating, extracting, and saving code
"""

def parsed_eval_csv(file_path, results):
  keys = ['error', 'dataset', 'split', 'task_id', 'result', 'avg_test_time', 'passed_tests', 'compiled', 'loc', 'lloc', 'sloc', 'comments', 'multi', 'blank', 'single_comments', 'CC', 'h1', 'h2', 'N1', 'N2', 'vocabulary', 'length', 'calculated_length', 'volume', 'difficulty', 'effort', 'time', 'bugs', 'MI']
       
  with open(file_path, 'w', newline='') as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames=keys)
      writer.writeheader()
      for result in results:
          # print(result)
          if type(result) == str:
            writer.writerow({"error": result})
          else:
            writer.writerow(result)

def extract_python_function(text):
    # Pattern to capture a Python function
    # - Starts with 'def' followed by any characters (the function name and arguments)
    # - Includes all indented lines after the 'def' line, allowing for empty lines
    # - Assumes an indentation of 4 spaces, but can be adjusted
    pattern = r"(def .+?:\n(?: {4,}.*\n|\n)*)"

    # Find all matches
    matches = re.findall(pattern, text, re.MULTILINE)
    return matches[0] if matches else ""

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
    

