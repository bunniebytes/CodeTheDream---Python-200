# Azure Authentication

# --- Azure Authentication Question 1 ---
# In a comment block, answer: when you run a Python script locally that uses DefaultAzureCredential, what does it rely on to authenticate? What command must you have run first, and how does DefaultAzureCredential know to use it?

# When running a Python script locally with DefaultAzureCredential, it
# typically relies on credentials from the Azure CLI for
# authentication.
# Before running the script, we first log in using "az login".  This
# command authenticates a local Azure CLI session and stores access
# tokens/credentials on the machine.
# DefaultAzureCredential automatically checks multiple authentication
# methods in a specific order. When it detects that the Azure CLI is
# logged in, it uses the cached Azure CLI credentials to authenticate
# requests to Azure services.

# --- Azure Authentication Question 2 ---
# In a comment block, answer: why can't a deployed pipeline (running on an Azure VM or container) use az login for authentication? What does it use instead, and why does the same Python code work without changes?

# A deployed pipeline running on an Azure VM or container cannot use "az login" because there is no interactive use session to perform browser based authentification.  Manual login is not possible or secure because pipelines are supposed to run unattended.
# It uses a service principal, managed identity assigned to the Azure resource (VM, App Service, or container).  Azure injects temporary credentials into the environment automatically.
# DefaultAzureCredential detects this environment and automatically
# retrieves a token from the managed identity without requiring any
# code changes.
# The Python code works without having to change anything because
# DefaultAzureCredential is designed to link multiple authentication
# methods (local CLI, environment variables, managed identity, etc.)
# and automatically selects the correct one based on where the code is
# running.

# --- Azure Authentication Question 3 ---
# You run a script that creates a DefaultAzureCredential and immediately gets an AuthenticationError. In a comment block, describe the two most likely causes and how you would diagnose each.

# 1. User is not authenticated with supported credential source
#    locally.  An example is Azure CLI has not been logged in.  This
#    would be diagnosed by running "az account show" to see if a valid
#    session exists.  It there is no subscription or it fails, "az
#    login" should be run.
# 2. The Python environment is not able to access a valid credential
#    source.  An example is Azure CLI is not installed or the wrong
#    environment is being used.  To diagnose this Azure CLI needs to
#    be confirmed if it is installed and running with "az --version".
#    Also need to confirm that the correct Python virtual environment
#    is being used.  Also check that "azure-identity" is installed in
#    the environment.
# Both cases DefaultAzureCredential fails because it tries multiple
# authentication methods in order and none of them succeeded.

# Blob Storage
# --- Blob Storage Question 1 ---
# In a comment block, describe the three-level hierarchy of Azure Blob Storage in your own words. Give a concrete analogy that maps each level to something familiar (a filesystem, a filing cabinet, etc.).

# 1. Storage Account - This is the top level container for all storage resources, acting like the filing cabinet.
# 2. Container - This is the subdivision inside the storage account used to organize the blobs, like the folders or drawers in the filin cabinet.
# 3. Blob - This is the stored file or data object, like a file or document inside the folder or drawer.
# Analogy - A filing cabinet (Storage Account), One drawer in the cabinet (Container), One file inside the drawer (Blob)

# --- Blob Storage Question 2 ---
# For each scenario below, write one sentence in a comment block saying whether you would use Blob Storage or a relational database (like Azure SQL), and why.

    # A REST API returns a JSON payload each hour. You need to store the raw responses for reprocessing later.
    # Your pipeline produces a table of 50 million customer transactions that your analytics team queries by date range and customer ID every day.
    # A computer vision model produces image embeddings as NumPy arrays. You need to save them between pipeline runs.

# I would use Blob storage for storing the JSONs because it does well storing large amounts of unstructured raw data for later reprocessing.
# I would use a relational database like Azure SQL for the customer transactions because the data is structured and it needs to be efficiently queried by date range and customer ID.
# I would use Blob storage for image embeddings as NumPy arrays because they are binary/unstructured data objects that only need to persist between the pipeline runs.

# --- Blob Storage Question 3 ---
# Write a function list_container(container_client) that prints the name and size (in bytes) of every blob in the container, one per line. The function should take a ContainerClient object as its only argument and return nothing.

def list_container(container_client):
    blobs = container_client.list_blobs()
    for blob in blobs:
        print(f"{blob.name} : {blob.size} bytes")

# --- Blob Storage Question 4 ---
# Write a function upload_text(container_client, blob_name, text) that encodes a Python string as UTF-8 and uploads it as a blob, overwriting any existing blob with the same name. The function should take a ContainerClient, a blob name string, and a text string, and return nothing.

def upload_text(container_client, blob_name, text):
    encoded_text = text.encode("utf-8")
    container_client.upload_blob(name = blob_name,
                                 data = encoded_text,
                                 overwrite = True)