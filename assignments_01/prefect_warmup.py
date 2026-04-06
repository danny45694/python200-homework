from prefect import task, flow
from prefect.logging import get_run_logger
import pandas as pd
import numpy as np
from scipy import stats




arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])


#Q2 Rebuild the pipeline from Q1 using Prefect. Do the following:



#Define Tasks
@task
def create_series(arr):
    values = pd.Series(arr, name="values")
    return values


@task
def clean_data(series):
    values = series.dropna()
    return values


@task
def summarize_data(series):
    summary = {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }
    return summary

#Define Flow
@flow
def data_pipeline(arr):
    logger = get_run_logger()
    values = create_series(arr)
    cleaned = clean_data(values)
    summary = summarize_data(cleaned)
    logger.info(summary)


#To run the flow

if __name__ == "__main__":
    data_pipeline(arr)



"""
Questions


1. Creating and setting up prefect is more work and for a task that is 1 time thing, it is unnecessary.


2. If used for a regular task, regardless of simplicity, it would be good to use Prefect. Prefect provides tools that enable things like re-running the code or continuing with the subsequent functions of a script in the event something fails.  

"""
