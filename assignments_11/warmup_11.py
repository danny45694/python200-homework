

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

To see what happened, you can click on the specific flow and the following page will provide detailed information about individual tasks and logs. 

Here you can see Info, warning and error logs. If it failed in the transform step, first thing is to check the code in the transform step. I'd check if the input is correct, if correct, the error is isolated to the transform area. If input is incorrect, the extract location is where the bug resides. 

I can narrow down from there. 

"""


# --------------------------------- Production Patterns -------------------------------


# Q1

"""

raise_for_status surface errors cleanly. It stops the program from continuing when something goes wrong, makes you deal with the responses early, and makes error handling structured and consistent. 


"""

# Q2

"""
Overwrite = True protects you from the program crashing if an existing file already exists. It automatically replaces old data with new data in the same location, saving you from needing to manually delete the old file. Without the overwrite = True, your program will crash/halt each time you run it, even if the pipeline code is bugfree. 

"""

# Q3 

"""

@task()
def signature(records, blob_path):

    raw = container.download_blob(blob_path).readall()
    data = json.loads(raw.decode("utf-8"))

    logger = get_run_logger()
    logger.info(f"Loaded {len(records)}")

"""