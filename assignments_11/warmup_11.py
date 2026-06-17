

# -------------------------------- Prefect Orchestration --------------------------------


# Q1

"""

1. Difference between a task and a flow like a computer. A task is a single component while a flow is the overall system/collective comprised of components(tasks).

Tasks handle retries, caching, dependency tracking etc. Tasks should be simple and focused.

Flow handles state tracking for the whole workflow and supports parameters, schedules and deployments.

I would not decorate the Celsius to Fahrenheit helper function. Prefect is designed for things that may fail, like an API or querying a databse. They are more focused on I/O actions. Adding @task to simple, lightweight functions that perform quick calculations is counter-productive. 

"""


# Q2


"""

@task(retries=3, retry_delay_seconds=30)
def call_api:
    continue

"""

#Q3


"""

To see what happened, you can click on the specific flow/task and review the logs tab to see the exception message that caused the failure.

"""


# --------------------------------- Production Patterns -------------------------------


# Q1

"""

raise_for_status raises exceptions that stop execution. It stops the program from continuing when something goes wrong, makes you deal with the responses early, and makes error handling structured and consistent. It is so tasks downstream don't run. print won't stop program from continuing.


"""

# Q2

"""
Overwrite = True protects you from the program crashing if an existing file already exists. It automatically replaces old data with new data in the same location, saving you from needing to manually delete the old file. Without the overwrite = True, your program will crash/halt each time you run it, even if the pipeline code is bugfree. 

"""

# Q3 

"""

@task()
def signature(records, blob_path):

    logger = get_run_logger()
    logger.info(f"Loaded {len(records)}")

"""