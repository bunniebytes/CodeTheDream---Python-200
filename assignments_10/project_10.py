# --- Step 6: Reflect ---
# Using an LLM for this is probably unnecessary. This could be done
# with simple rules using temperature and precipitation thresholds. A
# deterministic approach would be faster, cheaper, and more
# consistent. The downside is rules are less flexible if the
# definition of good/bad changes. For this case, a rule-based approach
# would likely be better.

# Video
# https://youtu.be/wbpIAhiTIo4 

import json
import os
import pandas as pd
from datetime import date

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
    
client = OpenAI()
print('OpenAI client created.')

credential = DefaultAzureCredential()

container_client = ContainerClient(
    account_url = account,
    container_name = container,
    credential = credential
)

output_path = "./outputs"
# Checks if the outputs folder exists, if not it creates it
os.makedirs(output_path, exist_ok=True)

# --- Step 1: Read ---
def load_raw_data():
    today = date.today().isoformat()
    blob_path = f"raw/{today}/weather.json"
 
    try:
        raw = container_client.download_blob(blob_path).readall()
        return json.loads(raw.decode("utf-8"))

    except Exception:
        print("Using fallback dataset")

        with open(
            "/weather_raw.json",
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)
    
    return data

def reshape_weather(data):
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

# --- Step 2: Transform ---
SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

VALID_LABELS = {"good", "marginal", "bad"}

def classify_weather_records(records, client):
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
            print(f"Processed {i} records")

    return results

# --- Step 3: Write ---

def load_data(container_client, data):
    today = date.today().isoformat()
    blob_path = f"processed/{today}/weather_classified.json"
    
    container_client.upload_blob(name = blob_path,
                                 data = data,
                                 overwrite = True)
    
    print(f"Uploaded {len(data)} bytes to {blob_path}")
    
    return blob_path

def serialize_data(data):
    serialized_bytes = json.dumps(data).encode("utf-8")
    print(f"Serialized {len(serialized_bytes)} bytes")
    
    return serialized_bytes

# --- Step 4: Spot-Check ---
def verify_data(container_client, blob_path):
    # Download blob
    blob_client = container_client.get_blob_client(blob_path)
    downloaded_bytes = blob_client.download_blob().readall()
    
    # Parse JSON
    data = json.loads(downloaded_bytes.decode("utf-8"))
    
    df = pd.DataFrame(data)
    
    print("Condition counts:")
    print(df["conditions"].value_counts())

    print("\nFirst 5 rows:")
    print(df.head())
    
    return df

# --- Step 5: Save Output ---
def save_json(data):
    with open(f"{output_path}/first_10_records.json", "w", encoding = "utf-8") as file:
        json.dump(data, file)

def main(container_client):
    data = load_raw_data()
    reshaped_data = reshape_weather(data)
    classified_records = classify_weather_records(reshaped_data, client)
    serialized_data = serialize_data(classified_records)
    blob_path = load_data(container_client, serialized_data)
    df = verify_data(container_client, blob_path)
    save_json(classified_records[:10])

if __name__ == "__main__":
    main(container_client)