import json
import os
from datetime import date
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential


account_url = "https://danielctd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"


credential = DefaultAzureCredential()

container = ContainerClient(
    account_url=account_url,
    container_name="pipeline-data",
    credential=credential
)


# Task 1

for blob in container.list_blobs():
    print(f" {blob.name} ({blob.size} bytes)")

"""
blob_path

raw = container.download_blob(blob_path).readall()
data = json.loads(raw.decode("utf-8"))["hourly"]
"""