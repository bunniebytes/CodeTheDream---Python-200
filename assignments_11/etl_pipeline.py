# Video link = https://youtu.be/GSaey0yUxLo

import requests
import json
import os
from datetime import date
from dotenv import load_dotenv

from prefect import task, flow
from prefect.logging import get_run_logger

from dotenv import load_dotenv
from openai import OpenAI

from azure.identity import DefaultAzureCredential
from azure.storage.blob import ContainerClient

if load_dotenv():
    print('Successfully loaded environment variables from .env')
else:
    print('Warning: could not load environment variables from .env')

api_key = os.getenv("OPENAI_API_KEY")
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

# --- Extract ---
@task(retries=2, retry_delay_seconds=10)
def get_weather():
    logger = get_run_logger()
    logger.info("Getting weather forecast for next 7 days.")
    
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
    
    logger.info("Weather forecast successfully extracted.")
    
    return data

# --- Transform ---
@task
def reshape_data(data):
    logger = get_run_logger()
    logger.info("Transforming the data to per hour lists")
    
    hourly = data["hourly"]

    records = []

    for time, temp, precip in zip(
        hourly["time"],
        hourly["temperature_2m"],
        hourly["precipitation"]
    ):
        records.append({
            "time": time,
            "temperature_2m": temp,
            "precipitation": precip
        })

    return records

SYSTEM_PROMPT = (
"You are classifying hourly weather conditions for outdoor running.  "
"Given a temperature in Celsius and a precipitation amount in mm, "
"classify the conditions as exactly one of: good, marginal, or bad.  "
"Reply with that one word only -- no punctuation, no explanation."
)

VALID_LABELS = {"good", "marginal", "bad"}

@task
def classify_weather_records(records):
    logger = get_run_logger()
    client = OpenAI()
    logger.info('OpenAI client created.')
    
    logger.info("Classifying weather using LLM")
    results = []

    for i, record in enumerate(records[:24], start=1):

        user_message = (
            f"Temperature: {record['temperature_2m']}C, Precipitation: {record['precipitation']}mm"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
        )

        label = response.choices[0].message.content.strip().lower()

        if label not in VALID_LABELS:
            label = "unknown"

        results.append({
            **record,
            "conditions": label
        })

        if i % 6 == 0:
            logger.info(f"Processed {i} records")

    return results

# --- Load ---
@task
def serialize_data(data):
    logger = get_run_logger()
    logger.info("Serializing data")
    serialized_bytes = json.dumps(data).encode("utf-8")
    logger.info(f"Serialized {len(serialized_bytes)} bytes")
    
    return serialized_bytes

@task()
def upload_data(container_client, serialized_data):
    logger = get_run_logger()
    today = date.today().isoformat()
    blob_path = f"final/{today}/weather_etl.json"
    logger.info(f"Uploading data to {blob_path}")
    
    container_client.upload_blob(name = blob_path,
                                 data = serialized_data,
                                 overwrite = True)
    
    logger.info(f"Uploaded {len(serialized_data)} bytes to {blob_path}")
    
    return blob_path

@flow(log_prints=True)
def main():
    logger = get_run_logger()
    logger.info("Starting pipeline")
    
    data = get_weather()
    reshaped_data = reshape_data(data)
    classified_data = classify_weather_records(reshaped_data)
    serialized_data = serialize_data(classified_data)
    blob_path = upload_data(container_client, serialized_data)
    
    logger.info(f"Pipeline complete. Final blob path: {blob_path}")
    
if __name__ == "__main__":
    main()