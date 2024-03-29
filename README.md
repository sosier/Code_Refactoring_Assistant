# Code_Refactoring_Assistant
 
## Code Evaluation

### Initial Installation / Setup

1. Move the `eval.py` to where your code is located or your code here

2. Install required packages:

```bash
pip install datasets  # Huggingface datasets package
pip install google-cloud
pip install google-cloud-run
pip install google-cloud-logging
pip install multiprocess  # For bulk evaluation
pip install radon  # More code complexity metrics
```

3. Verify you have access to the `safe-eval` job (sandboxed evaluation environment) on Google Cloud. You should see it at this link: [https://console.cloud.google.com/run/jobs?authuser=2&project=code-refactoring-assistant](https://console.cloud.google.com/run/jobs?authuser=2&project=code-refactoring-assistant)

4. Install and initialize the Google Cloud CLI by following the instructions [here](https://cloud.google.com/sdk/docs/install).
  - Make sure to complete all required steps including the `gcloud init` one

5. Set up your Google Cloud ADC (Applicaiton Default Credentials) by following the steps [here](https://cloud.google.com/docs/authentication/provide-credentials-adc#google-idp). The Google Cloud Python packages use these credential to interact with Google Cloud resources.


### Usage

Import the relevant functions / data:

```python
from eval import DATA, evaluate, bulk_evaluate

mbpp = DATA["mbpp"]  # train, validation, and test
humaneval = DATA["openai_humaneval"]  # test only
```
 - `DATA` is coming from Huggingface behind-the-scenes and is provided for convenience because the evaluation functions need this data loaded
 - `evaluate` and `bulk_evaluate` are the primary evaluation functions and will assess code snippet(s) for correctness, simplicity, and efficiency (example usage below)
   - `evaluate` evaluates a single code snippet on a single task from either MBPP or HumanEval
   - `bulk_evaluate` evaluates many code snippets in parallel, specifically one code snippet per task in a "split" of MBPP or HumanEval (e.g. the HumanEval test split, the MBPP train split, etc.)
   - If you need a different bulk evaluation approach, investigate the internals of the `bulk_evaluate` function and use it as a starting point. Because of the Google Cloud overhead a single `evaluate` call is pretty slow (~20-30s), but because it's on Google Cloud it can be more or less infinitely parallezied to greatly speed up how long it takes to process a batch of evaluations.

`evaluate` Example (running a code snippet on the first HumanEval task):

```python
canonical_solution = humaneval["test"][0]["prompt"] + humaneval["test"][0]["canonical_solution"]
evaluate(
    dataset="openai_humaneval",
    split="test",
    task_id=0,
    code=canonical_solution  # or alternatively YOUR_CODE_HERE
)
```

Example Output:

```
{'dataset': 'openai_humaneval',
 'split': 'test',
 'task_id': 0,
 'result': 'passed',
 'avg_test_time': 0.00019492872000000715,
 'passed_tests': True,
 'compiled': True,
 'loc': 19,
 'lloc': 10,
 'sloc': 9,
 'comments': 0,
 'multi': 7,
 'blank': 3,
 'single_comments': 0,
 'CC': 5,
 'h1': 3,
 'h2': 6,
 'N1': 3,
 'N2': 6,
 'vocabulary': 9,
 'length': 9,
 'calculated_length': 20.264662506490406,
 'volume': 28.529325012980813,
 'difficulty': 1.5,
 'effort': 42.793987519471216,
 'time': 2.377443751081734,
 'bugs': 0.009509775004326938,
 'MI': 95.60592320426878}
```

`bulk_evaluate` Example (evaluating code snippets for each HumanEval task):

```python
bulk_evaluate(
    dataset="openai_humaneval",
    split="test",
    code=LIST_OF_CODE_SNIPPETS, # one for each task in HumanEval test
    # Run in parallel using 4 cores
    # Entering None will use all cores on your machine:
    num_processes=4
)
```
