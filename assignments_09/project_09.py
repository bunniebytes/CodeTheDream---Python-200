# Video Link = https://youtu.be/9qanofDcMxE

import requests
import json
import os
import pandas as pd
from datetime import date
from dotenv import load_dotenv

from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient

load_dotenv()

account = os.getenv("AZURE_ACCOUNT_URL")
container = os.getenv("AZURE_CONTAINER")

if account and container:
    print("Azure configuration loaded successfully.")
else:
    print("Warning: missing Azure config in .env file.")

credential = DefaultAzureCredential()

container_client = ContainerClient(
    account_url = account,
    container_name = container,
    credential = credential
)

output_path = "./outputs"
# Checks if the outputs folder exists, if not it creates it
os.makedirs(output_path, exist_ok=True)

# --- Step 1: Extract ---
def get_weather():
    # longitude and latitude of San Francisco
    latitude = 37.773972
    longitude = -122.431297
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        "&hourly=temperature_2m,precipitation"
        "&forecast_days=7"
    )

    response = requests.get(url)
    
    # raises exception for HTTP errors
    response.raise_for_status()
    data = response.json()
    
    return data

# --- Step 2: Serialize ---
def serialize_data(data):
    serialized_bytes = json.dumps(data).encode("utf-8")
    print(f"Serialized {len(serialized_bytes)} bytes")
    
    return serialized_bytes

# --- Step 3: Load ---
def load_data(container_client, data):
    today = date.today().isoformat()
    blob_path = f"raw/{today}/weather.json"
    
    container_client.upload_blob(name = blob_path,
                                 data = data,
                                 overwrite = True)
    
    print(f"Uploaded {len(data)} bytes to {blob_path}")
    
    return blob_path
    
# --- Step 4: Verify ---
def list_container(container_client):
    blobs = container_client.list_blobs()
    for blob in blobs:
        print(f"{blob.name} : {blob.size} bytes")
        
# --- Step 5: Read Back ---
def verify_data(container_client, blob_path):
    # Download blob
    blob_client = container_client.get_blob_client(blob_path)
    downloaded_bytes = blob_client.download_blob().readall()
    
    # Parse JSON
    data = json.loads(downloaded_bytes.decode("utf-8"))
    with open(f"{output_path}/weather_raw.json", "w", encoding = "utf-8") as file:
        json.dump(data, file)
    
    # loads hourly data into dataframe
    hourly_df = pd.DataFrame(data["hourly"])
    
    # prints first 5 rows of dataframe
    print(hourly_df.head(5))

def main(container_client):
    data = get_weather()
    serialized_data = serialize_data(data)
    blob_path = load_data(container_client, serialized_data)
    list_container(container_client)
    verify_data(container_client, blob_path)
    
if __name__ == "__main__":
    main(container_client)