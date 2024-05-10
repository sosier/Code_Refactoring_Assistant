
from eval import DATA, evaluate, bulk_evaluate

mbpp = DATA["mbpp"]  # train, validation, and test
humaneval = DATA["openai_humaneval"]  # test only

canonical_solution = humaneval["test"][0]["prompt"] + humaneval["test"][0]["canonical_solution"]
print(type(canonical_solution))
evaluate(
    dataset="openai_humaneval",
    split="test",
    task_id=0,
    code=canonical_solution  # or alternatively YOUR_CODE_HERE
)

canonical_solution = mbpp["test"][0]["code"]
evaluate(
    dataset="mbpp",
    split="test",
    task_id=0,
    code=canonical_solution  # or alternatively YOUR_CODE_HERE
)