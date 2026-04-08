import pandas as pd
import numpy as np
from prefect import task, flow
from prefect.logging import get_run_logger

# Pipelines Question 2
# Rebuild the pipeline from Q1 using Prefect. Copy your three functions from Pipeline Question 1 (create_series, clean_data, summarize_data) into this file and turn them into Prefect tasks using @task.
# Turn data_pipeline() into a Prefect flow using @flow. Inside the flow, call the three tasks in order and return the summary dictionary.

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

# takes a NumPy array and returns a pandas Series with the name "values"
@task(retries=3, retry_delay_seconds=3)
def create_series(arr):
    logger = get_run_logger()
    logger.info(f"Creating series from {arr}")
    values = pd.Series(arr)
    logger.info("Series created successfully")
    return values

# takes the Series, removes any NaN values using .dropna(), and returns the cleaned Series
@task(retries=3, retry_delay_seconds=3)
def clean_data(series):
    logger = get_run_logger()
    logger.info("Dropping all NaN values")
    cleaned_series = series.dropna()
    logger.info("Successfully dropped all NaN values")
    return cleaned_series

# takes the cleaned Series and returns a dictionary with four keys: "mean", "median", "std", and "mode". For mode, use series.mode()[0] to get a single value.
@task(retries=3, retry_delay_seconds=3)
def summarize_data(series):
    # had to use lambdas to ensure only received the first value of mode so we could use to_dict()
    logger = get_run_logger()
    logger.info("Summarizing the data and finding Mean, Median, Standard Deviation and Mode")
    data_summary = series.agg(mean = "mean",
                            median = "median",
                            std = "std",
                            mode = lambda x : x.mode()[0] if not x.mode().empty else None).to_dict()
    logger.info("Summary computed successfully")
    return(data_summary)

# calls the three functions above in sequence and returns the summary dictionary.
@flow
def pipeline_flow(arr):
    logger = get_run_logger()
    values = create_series(arr)
    cleaned_values = clean_data(values)
    data_summary = summarize_data(cleaned_values)
    for k, v in data_summary.items():
        logger.info(f"{k} : {v}")
    return data_summary

if __name__ == "__main__":
    pipeline_flow(arr)
    
# This pipeline is simple -- just three small functions on a handful of numbers. Why might Prefect be more overhead than it is worth here?
# Prefect has a lot more set up and run time than the 3 original functions we wrote.  It ends up taking more time and effort to set up and run Prefect.

# Describe some realistic scenarios where a framework like Prefect could still be useful, even if the pipeline logic itself stays simple like in this case.
# Some things that Prefect will help is logging the steps so we can see if things ran successfully or where it broke.  Prefect can also retry the steps that do fail instead of running the entire code again.  Another thing is we can set Prefect to run on a schedule even if it is a simple task so we don't have to keep remembering.