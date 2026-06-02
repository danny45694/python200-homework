import requests
import json
import io
import pandas as pd
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource.subscriptions import SubscriptionClient
from datetime import date
import os


account_url = "https://danielctd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"



credential = DefaultAzureCredential()
client = SubscriptionClient(credential)

#Verify connection
for sub in client.subscriptions.list():
    print(sub.display_name)

container = ContainerClient(
    account_url=account_url,
    container_name="pipeline-data",
    credential=credential
)

url = (
    "https://api.open-meteo.com/v1/forecast?latitude=35.2271&longitude=-80.8431&hourly=temperature_2m,precipitation&forecast_days=7"
)


response = requests.get(url)
response.raise_for_status()

# ---------------------------------- Step 2: Serialize -----------------------------

data = response.json()
payload = json.dumps(data).encode('utf-8')

# -----------------------------------Step 3: Load -----------------------------------


today = date.today().isoformat()
blob_path = f"raw/{today}/weather.json"

#Uploading Blob
"""
1st argument is blob name (path in container), 2nd is content as bytes. Overwrite=True parameter means if it exists, overwrite it. If not, if it exists, error is thrown"
"""

container.upload_blob(blob_path, payload, overwrite=True)
print(f"Uploaded to {blob_path}")



# --------------------------------- Step 4: Verify -------------------------------------

print("\nBlobs in container:")
for blob in container.list_blobs():
    print(f" {blob.name} ({blob.size} bytes)")



# --------------------------------- Step 5: Read back ----------------------------------
outputs = "outputs"
file_name = "weather_raw.json"

full_path = os.path.join(outputs, file_name)
print(full_path)




raw = container.download_blob(blob_path).readall()
data = json.loads(raw.decode("utf-8"))["hourly"]

df = pd.DataFrame(json.loads(raw.decode("utf-8"))["hourly"])
print(f"\nFirst 5 rows:")
print(df.head())

with open(full_path, "w", encoding="utf-8") as file:
    json.dump(data, file)