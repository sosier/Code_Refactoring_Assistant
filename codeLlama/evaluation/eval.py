from datasets import load_dataset  # Huggingface datasets package
from google.cloud import run_v2, logging_v2
import radon
from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit
import time


'''

General evaluation file used for all models in this project

'''



# Load datasets
DATA = {
    "mbpp": load_dataset("mbpp"),
    "openai_humaneval": load_dataset("openai_humaneval")
}
# Google Cloud constants
GOOGLE_CLOUD_PROJECT = "code-refactoring-assistant"
GOOGLE_CLOUD_LOCATION = "us-central1"
GOOGLE_CLOUD_JOB = "safe-eval"

def _send_safe_eval_job(
    jobs_client,
    code,
    test_code,
    project=GOOGLE_CLOUD_PROJECT,
    location=GOOGLE_CLOUD_LOCATION,
    job=GOOGLE_CLOUD_JOB
):
    assert type(code) == str and type(test_code) == str

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
    operation = jobs_client.run_job(request=request)
    return operation

def _fetch_safe_eval_job_result(
    logging_client,
    execution_name,
    project=GOOGLE_CLOUD_PROJECT,
    job=GOOGLE_CLOUD_JOB
):
    # Fetch the actual result (stdout)
    # Run in a while loop because sometimes logging result isn't immediately
    # available
    logging_result = []
    i = 0
    while len(logging_result) < 1:
        if i > 0:
            time.sleep(5)  # Wait 5 seconds
        elif i > 60/5 * 10:  # If it's been 10 minutes give up and let it error
            break

        # By default returns a generator so we have to turn it into a list
        logging_result = list(logging_client.list_entries(
            resource_names=[f"projects/{project}"],
            filter_=f'''
                resource.type = "cloud_run_job"
                AND resource.labels.job_name = "{job}"
                AND labels."run.googleapis.com/execution_name" = "{execution_name}"
                AND log_name="projects/{project}/logs/run.googleapis.com%2Fstdout"
            ''',
            max_results=1
        ))
        i += 1

    result = logging_result[0].payload
    return result

def safely_check_correctness(code, test_code):
    """
    Safely checks correctness of untrusted code vs. untrusted test code by
    calling out to a sandboxed (serverless Docker) environment on Google Cloud
    to execute the code / tests. Estimated time is ~20-30s, but because the
    environment is serverless on Google Cloud can in theory perform many runs
    / evaluations in parallel
    """
    # Create client for running job
    client = run_v2.JobsClient()

    # Send the request
    operation = _send_safe_eval_job(
        jobs_client=client,
        code=code,
        test_code=test_code,
        project=GOOGLE_CLOUD_PROJECT,
        location=GOOGLE_CLOUD_LOCATION,
        job=GOOGLE_CLOUD_JOB
    )

    # Wait
    print("Executing...")

    # Get results
    response = operation.result()
    execution_name = response.name.split("/")[-1]

    # Create client for getting result
    client = logging_v2.Client(project=GOOGLE_CLOUD_PROJECT)

    # Fetch the actual result (stdout)
    return _fetch_safe_eval_job_result(
        logging_client=client,
        execution_name=execution_name,
        project="code-refactoring-assistant",
        job="safe-eval"
    )

def _bulk_send_safe_eval_jobs(code, test_code):
    assert type(code) == list and type(test_code) == list
    assert all(type(snippet) == str for snippet in code)
    assert all(type(snippet) == str for snippet in test_code)
    assert len(code) == len(test_code)
    
    # Create client for running job
    client = run_v2.JobsClient()

    print("Sending", len(code), "jobs for execution...")
    execution_ids = []

    for i, (code_instance, tests_instance) in enumerate(zip(code, test_code)):
        if i > 0:
            # Wait 2 seconds between requests to avoid hitting the "Job run
            # requests per minute per region" rate limit:
            time.sleep(2)

        # Send the request
        operation = _send_safe_eval_job(
            jobs_client=client,
            code=code_instance,
            test_code=tests_instance,
            project=GOOGLE_CLOUD_PROJECT,
            location=GOOGLE_CLOUD_LOCATION,
            job=GOOGLE_CLOUD_JOB
        )
        execution_name = operation.metadata.name.split("/")[-1]
        execution_ids.append(execution_name)

    print(len(execution_ids), "jobs sent")
    return execution_ids

def _bulk_fetch_safe_eval_job_results(execution_ids):
    # Create client for getting result
    client = logging_v2.Client(project=GOOGLE_CLOUD_PROJECT)

    print("Fetching", len(execution_ids), "job results...")
    results = []

    for i, execution_name in enumerate(execution_ids):
        if i > 0:
            if i % 10 == 0:
                print(f"{len(results)}/{len(execution_ids)} fetched")
            
            # Wait 2 seconds between requests to avoid hitting the
            # "logging.googleapis.com/read_requests: Read requests per minute
            # per user" rate limit:
            time.sleep(2)
        
        # Fetch the actual result (stdout)
        result = _fetch_safe_eval_job_result(
            logging_client=client,
            execution_name=execution_name,
            project=GOOGLE_CLOUD_PROJECT,
            job=GOOGLE_CLOUD_JOB
        )
        results.append(result)
    
    print(len(results), "job results fetched")
    return results

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

    # Evaluate Correctness / Efficiency
    execution_results = safely_check_correctness(code, test_code)
    
    # Evaluate Code Simplicity
    if execution_results["compiled"]:
        simplicity_results = analyze_simplicity(code)
    else:
        simplicity_results = {}

    return dict(
        dataset=dataset,
        split=split,
        task_id=task_id,
        **execution_results,
        **simplicity_results
    )

def bulk_evaluate(dataset, split, code):
    assert dataset in DATA
    assert split in DATA[dataset]
    
    num_tasks = len(DATA[dataset][split])
    assert type(code) == list
    assert len(code) == num_tasks
    assert all(type(snippet) == str for snippet in code)
    test_code = [
        build_tests(dataset, DATA[dataset][split][i])
        for i in range(num_tasks)
    ]

    # Perform evaluations vs. test code on Google Cloud:
    execution_ids = _bulk_send_safe_eval_jobs(code, test_code)
    execution_results = _bulk_fetch_safe_eval_job_results(execution_ids)
    
    # Calculate simplicity metrics
    simplicity_results = [
        analyze_simplicity(code_instance)
        if execution_result["compiled"]
        else {}
        for code_instance, execution_result in zip(code, execution_results)
    ]

    # Merge all results together:
    results = [
        dict(
            dataset=dataset,
            split=split,
            task_id=i,
            **execution_result,
            **simplicity_result
        )
        for i, (execution_result, simplicity_result)
        in enumerate(zip(execution_results, simplicity_results))
    ]

    return results
