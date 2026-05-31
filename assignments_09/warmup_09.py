from azure.storage.blob import ContainerClient
from azure.identity import DefaulyAzureCredential


# ------------------------------ Azure Authentication ---------------------------------

#Q1 

"""
When running a python script that uses DefaultAzureCredential locally, it requires a Managed identity. It is a special type of service principal that Azure manages.

To use it, you need the azure-identity and azure-mgmt-resource packages. Installed with:

uv pip install azure-identity azure-mgmt-resource

Then you run the az login command. Once that is done, DefaultAzureCred. uses it automatically.

"""

#Q2

"""
A deployed pipeline can't use the az login because there is no human around to run az login. It instead uses a managed identity. The way DefaultAzureCredential is setup, it automatically switches from using 'az login' credentials to the managed identity or service principal data without modifying the underlying code. 

"""

#Q3

"""
The 2 most likely causes is:

1. Az login not setup or session expired

2. Session expired

I would use the code below to verify az login is in place. If this works without errors, it is not the az login. If it does, we try freshing the session. If that fails, I would try refreshing the session.

    from azure.identity import DefaultAzureCredential
    from azure.mgmt.resource import SubscriptionClient

    credential = DefaultAzureCredential()
    client = SubscriptionClient(credential)

    for sub in client.subscriptions.list():
        print(sub.display_name)
 

"""


# ---------------------------------- Blob Storage --------------------------------------

#Q1 

"""
three-level hierarachy of the Azure Blob Storage system

Storage account > Container > blob

Storage account is the overall account
Container - Similar to a top-level folder. Like C:/ daniel or C:/ CTD. 
blob - Individual files. Stored by name. They are stored in bytes. You have to encode strings before uploading and decode after downloading.

"""

#Q2

"""

1. REST API return JSON payload each hour - Blob Storage. You will need to reprocess them later so better jsut t osave it. 
2. Relational database. While it has higher costs, the files are ready to query immediately with SQL and low maintenance.
3. Store in blob storage. Cheaper overall, faster and don't need to process it to be query ready."""



#Q3

def list_container(container_client: ContainerClient) -> None:
    
    blob_list = container_client.list_blobs()
    for blob in blob_list:
        print(f"{blob.name} - {blob.size} ")


#Q4

def upload_text(container_client, blob_name, text):

    data_bytes = text.encode('utf-8')
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(data_bytes, overwrite=True)

    return None