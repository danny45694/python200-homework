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
Container = "pipeline-data"