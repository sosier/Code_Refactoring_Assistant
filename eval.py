from datasets import load_dataset  # Huggingface datasets package
from google.cloud import run_v2, logging_v2
from multiprocess import Pool
import radon
from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit

# Load datasets
DATA = {
    "mbpp": load_dataset("mbpp"),
    "openai_humaneval": load_dataset("openai_humaneval")
}

def safely_check_correctness(code, test_code):
    """
    Safely checks correctness of untrusted code vs. untrusted test code by
    calling out to a sandboxed (serverless Docker) environment on Google Cloud
    to execute the code / tests. Estimated time is ~20-30s, but because the
    environment is serverless on Google Cloud can in theory perform many runs
    / evaluations in parallel
    """
    assert type(code) == str and type(test_code) == str

    project = "code-refactoring-assistant"
    location = "us-central1"
    job = "safe-eval"

    # Create client for running job
    client = run_v2.JobsClient()

    # Initialize request argument(s)
    request = run_v2.RunJobRequest(
        name=f"projects/{project}/locations/{location}/jobs/{job}",
        overrides=run_v2.types.RunJobRequest.Overrides(
            container_overrides=[
                run_v2.types.RunJobRequest.Overrides.ContainerOverride(
                    args=[code, test_code]
                )
            ]
        )
    )

    # Send the request
    operation = client.run_job(request=request)

    # Wait
    print("Executing...")

    # Get results
    response = operation.result()
    execution_name = response.name.split("/")[-1]

    # Create client for getting result
    client = logging_v2.Client(project=project)

    # Fetch the actual result (stdout)
    # (by default is a generator so we have to call next on it)
    result = next(client.list_entries(
        resource_names=[f"projects/{project}"],
        filter_=f'''
            resource.type = "cloud_run_job"
            AND resource.labels.job_name = "{job}"
            AND labels."run.googleapis.com/execution_name" = "{execution_name}"
            AND log_name="projects/{project}/logs/run.googleapis.com%2Fstdout"
        ''',
        max_results=1
    )).payload
    
    return result

def build_MBPP_tests(task):
    return (
        (task["test_setup_code"] + "\n\n" if task["test_setup_code"] else "")
        + "\n".join(
            task["test_list"]
            + task["challenge_test_list"]
        )
    )

def build_humaneval_tests(task):
    return (
        task["test"] + "\n" +
        f"check({task['entry_point']})"
    )

def build_tests(dataset, task):
    assert dataset in ("mbpp", "openai_humaneval")
    if dataset == "mbpp":
        return build_MBPP_tests(task)
    else:  # HumanEval
        return build_humaneval_tests(task)

def analyze_simplicity(code, multiline_str_comment=True):
    return dict(
        **radon.raw.analyze(code)._asdict(),
        CC=cc_visit(code)[0].complexity,
        **h_visit(code).total._asdict(),
        MI=mi_visit(code, multi=multiline_str_comment)
    )

def evaluate(dataset, split, task_id, code):
    assert dataset in DATA
    assert split in DATA[dataset]
    assert DATA[dataset][split][task_id]
    assert type(code) == str

    task = DATA[dataset][split][task_id]
    test_code = build_tests(dataset, task)

    return dict(
        dataset=dataset,
        split=split,
        task_id=task_id,
        **safely_check_correctness(code, test_code),
        **analyze_simplicity(code)
    )

def bulk_evaluate(dataset, split, code, num_processes):
    """
    num_processes = # of processes to run in parallel. Max = the number of cores
        on your computer. If None then the number returned by os.cpu_count()
        is used
    """
    assert dataset in DATA
    assert split in DATA[dataset]
    
    num_tasks = len(DATA[dataset][split])
    assert type(code) == list
    assert len(code) == num_tasks
    assert all(type(snippet) == str for snippet in code)

    assert(
        (type(num_processes) == int and num_processes >= 1)
        or num_processes is None
    )

    # Wrap evaluate in a try except so that we don't lose all our results if
    # there's an unexpected error on the Google Cloud side (e.g. an evaluation
    # times out, etc.):
    def _evaluate(i):
        try:
            return evaluate(dataset, split, task_id=i, code=code[i])
        except:
            return "ERROR"

    # For efficiency run the individual evaluations in parallel:
    results = Pool(num_processes).map(_evaluate, range(num_tasks))

    return results
