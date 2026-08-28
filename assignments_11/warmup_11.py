# Prefect Orchestration
# --- Prefect Question 1 ---
# @task is a unit of work inside a flow.  Tasks are individual steps that can be tracked, retried, logged and executed independently within the flow
# @flow is the orchestration layer.  It is the entry point of the pipeline and controls the structure of work.

# I would not decorate a Celsius to Fahrenheit helper function with @task because it is just simple calculation with no need for retries or tracking.

# --- Prefect Question 2 ---
# @task(retries = 3, retry_delay_seconds = 30)

# --- Prefect Question 3 ---
# I would click into the failed "transform" task in the Prefect UI. In
# the task run view, I would check the Logs section to see the error
# message and stack trace. I would also look at the Inputs and Outputs
# tabs to confirm what data was passed into the task. This helps me
# understand whether the failure came from bad input data, an API
# error, or a coding issue inside the task.


# Production Patterns
# --- Production Question 1 ---
# raise_for_status() checks the HTTP response and automatically raises
# an exception if the status code indicates an error (like 4xx or
# 5xx). This is better than manually checking status codes because
# raising an exception properly fails the Prefect task, which stops
# downstream tasks from running and shows the error clearly in the UI
# with a stack trace.

# If the API returns a 500 error and you only print an error message,
# the task may still "succeed" and downstream tasks will continue
# running with bad or missing data. With raise_for_status(), the task
# fails immediately and downstream tasks are skipped, which prevents
# incorrect or corrupted data from flowing through the pipeline.

# --- Production Question 2 ---
# overwrite=True ensures that when the pipeline is re-run after a
# failure, the new output file replaces any existing file with the
# same name in Azure Blob Storage. This prevents conflicts or errors
# caused by trying to upload a file that already exists.

# Without overwrite=True, the second run could fail during the upload
# step, or you could end up with duplicate/partial outputs depending
# on how the upload is handled. In a crash-and-retry scenario, this
# makes the pipeline less reliable because you would have to manually
# clean up old blobs before re-running.

# --- Production Question 3 ---
# Write a task stub -- just the function signature, decorator, and a single log line -- that uses get_run_logger() to log an INFO message saying how many records were loaded. The function should accept records (a list) and blob_path (a string) as arguments.
from prefect import task, get_run_logger

@task
def log_loaded_records(records: list, blob_path: str):
    logger = get_run_logger()
    logger.info(f"Loaded {len(records)} records from {blob_path}")