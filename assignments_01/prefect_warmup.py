from prefect import task, flow
from prefect.logging import get_run_logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats
from scipy import stats
import seaborn as sns




arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])


#Q2 Rebuild the pipeline from Q1 using Prefect. Do the following:

"""  1. Turn the previous 3 functions into Prefect tasks using @task. 
# 2. Turn data_pipeline() into a Prefect flow using @flow. 
# 3. Call the 3 tasks in order and return the summary dictionary. Add this block at the bottom of the file so the flow runs when you execute the script.  """

if __name__ == "__main__":
    pipeline_flow()

""" 
4. Run the workflow from the terminal. Values should match that of Q1.
5. Finally: Add a comment block at the bottom of prefect_warmup.py. Answer these 2 questions.

    1. This pipeline is simple -- just three small functions on a handful of numbers. Why might Prefect be more overhead than it is worth here?

    2. Describe some realistic scenarios where a framework like Prefect could still be useful, even if the pipeline logic itself stays simple like in this case.

 """


#Define Tasks
@task
def create_series(arr):
    values = pd.Series(arr)
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
    clean_data(values)
    summary = summarize_data(values)
    logger.info(summary)


#print(data_pipeline(arr))


#To run the flow

if __name__ == "__main__":
    data_pipeline(arr)



"""
Questions


1. Creating and setting up prefect is more work and for a task that is 1 time thing, it is unnecessary.


2. If used for a regular task, regardless of simplicity, it would be good to use Prefect. Prefect provides tools that enable things like re-running the code or continuing with the subsequent functions of a script in the event something fails.  

"""
