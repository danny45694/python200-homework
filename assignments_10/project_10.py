import json
import os
from datetime import date
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential

load_dotenv()

account_url = "https://danielctd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"


credential = DefaultAzureCredential()

container = ContainerClient(
    account_url=account_url,
    container_name="pipeline-data",
    credential=credential
)


# Task 1

#Find upload name
for blob in container.list_blobs():
    blob_path = (f"{blob.name}")
    print(f" {blob.name} ({blob.size} bytes)")

# Copied uploaded file name

#Download
raw = container.download_blob(blob_path).readall()
data = json.loads(raw.decode("utf-8"))

hourly = data["hourly"]

records = []
for i in range(len(hourly["time"])):
    record = {
        "time": hourly["time"][i],
        "temperature_2m": hourly["temperature_2m"][i],
        "precipitation": hourly["precipitation"][i],
    }
    records.append(record)

print(f"Loaded {len(records)} hourly records")

# Task 2


SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

#Transform

VALID_LABELS = {"good", "marginal", "bad"}

def make_user_message(record):
    return (
        f"Temperature: {record['temperature_2m']}C, "
        f"Precipitation: {record['precipitation']}mm"
    )

client = OpenAI(api_key=os.environ["OPEN_API_KEY"])
enriched = []
for i, record in enumerate(records):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_message(record)},
        ]
    )
    raw_label = response.choices[0].message.content.strip().lower()
    label = raw_label if raw_label in VALID_LABELS else "unknown"
    enriched.append({**record, "conditions": label})
    if (i + 1) % 6 == 0:
        print("6 records processed and running.")
    if (i + 1) % 24 == 0:
        print(f"  Processed {i + 1} records...")
    

# Load
processed_path = f"processed/{blob_path}"
container.upload_blob(processed_path, json.dumps(enriched).encode("utf-8"), overwrite=True)
print(f"Uploaded to {processed_path}")


# --------------------------------- Step 3: Write ---------------------------------------

processed_path = f"processed/{blob_path}"
payload = json.dumps(enriched).encode("utf-8")
container.upload_blob(processed_path, payload, overwrite=True)
print(f"Uploaded {len(payload)} bytes to {processed_path}")



# -------------------------------- Step 4: Spot-Check ----------------------------------

raw = container.download_blob(payload).readall()
data = json.loads(raw.decode("utf-8"))["hourly"]

df = pd.Dataframe(json.loads(raw.decode("utf-8"))["hourly"])
print(f"\nFirst 5 rows:")
print(df.head())


# ------------------------------- Step 5: Save Output ----------------------------------

outputs = "outputs"
file_name = "first_10_records.json"

full_path = os.path.join(outputs, file_name)

with open(full_path, "w", encoding="utf-8") as file:
    json.dump(data, file)