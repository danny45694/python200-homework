

#                Video Link                       https://youtu.be/fF-lffnYKDQ




import requests
import json
import os
from datetime import date
from dotenv import load_dotenv
from openai import OpenAI
from prefect import task, flow
from prefect.logging import get_run_logger
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource.subscriptions import SubscriptionClient


load_dotenv()

ACCOUNT_URL = "https://danielctd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"


credential = DefaultAzureCredential()
client = SubscriptionClient(credential)



@task(retries=2, retry_delay_seconds=10)
def extract_task():
    logger = get_run_logger()

    url = ("https://api.open-meteo.com/v1/forecast?latitude=35.2271&longitude=-80.8431&hourly=temperature_2m,precipitation&forecast_days=7")

    response = requests.get(url) #Call Open-Meteo API for 7 days of hourly temperature_2m and precipitation data
    response.raise_for_status() #Raise status

    data = response.json() #Already a dict
    
    logger.info("Successfully called Open-Meteo API and returned JSON dict")
    return data




# Transform Task
def user_message(record): 
    return (
    f"Temperature: {record['temperature_2m']}C, "
    f"Precipitation: {record['precipitation']}mm"
    )


@task
def transform_task(data):
    logger = get_run_logger()
    hourly = data["hourly"]

    records = []
    for i in range(len(hourly["time"])):
        record = {
            "temperature_2m": hourly["temperature_2m"][i],
            "precipitation": hourly["precipitation"][i],
        }
        records.append(record)

    SYSTEM_PROMPT = (
        "You are classifying hourly weather conditions for outdoor running. "
        "Given a temperature in Celsius and a precipitation amount in mm, "
        "classify the conditions as exactly one of: good, marginal, or bad. "
        "Reply with that one word only -- no punctuation, no explanation."
    )

    VALID_LABELS = {"good", "marginal", "bad"}

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    enriched = []
    for i, record in enumerate(records):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message(record)},
            ]
        )

        raw_label = response.choices[0].message.content.strip().lower()
        label = raw_label if raw_label in VALID_LABELS else "unknown"
        enriched.append({**record, "conditions": label})
        if (i + 1) % 6 == 0:
            logger.info("6 records processed") #Output message every 6 records
        if (i + 1) % 24 == 0:
            logger.info("Processed 24 records") #Output message once 24 records reached
            return enriched
        

@task 
def load_task(enriched):
    logger = get_run_logger()

    ACCOUNT_URL = "https://danielctd2026sa.blob.core.windows.net"

    container = ContainerClient(
    account_url=ACCOUNT_URL,
    container_name="pipeline-data",
    credential=credential
)
    
    #Date and blob_path
    today = date.today().isoformat()
    blob_path = f"final/{today}/weather_etl.json"

    #Load
    payload = json.dumps(enriched).encode("utf-8")

    #Upload
    container.upload_blob(blob_path, payload, overwrite=True)
    logger.info(f"Uploaded to {blob_path}. Total bytes uploaded: {len(payload)}")
    return blob_path
    
@flow(log_prints=True)
def etl_pipeline():
    logger = get_run_logger()

    #Task 1: Load Task
    logger.info("Starting Pipeline: Extracting data")
    data = extract_task()

    #Task 2: Transform Data
    logger.info("Beginning transformation of Data")
    enriched = transform_task(data)

    #Task 3: Load task and upload
    logger.info("Now loading task and uploading to Azure cloud")
    blob_path = load_task(enriched)

    # Completion message
    logger.info(f"Data successfully processed and uploaded to Azure path {blob_path}")

if __name__ == "__main__":
    etl_pipeline()