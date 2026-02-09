from azure.storage.blob import BlobServiceClient

import pandas as pd
import io
import json
import os
from datetime import datetime

# Azurite connection string (default)
AZURITE_CONNECTION_STRING = "UseDevelopmentStorage=true"

CONTAINER_NAME = "datasets"
BLOB_NAME = "All_Diets.csv"
OUTPUT_DIR = "task3_serverless/simulated_nosql"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "results.json")


def process_nutritional_data_from_azurite():
    print(f"[{datetime.now()}] Starting serverless data processing")

    # Connect to Azurite
    blob_service_client = BlobServiceClient.from_connection_string(
        AZURITE_CONNECTION_STRING
    )

    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    blob_client = container_client.get_blob_client(BLOB_NAME)

    # Download CSV from Azurite
    stream = blob_client.download_blob().readall()
    df = pd.read_csv(io.BytesIO(stream))

    print(f"[{datetime.now()}] CSV loaded from Azurite")

    # Calculate averages
    avg_macros = (
        df.groupby("Diet_type")[["Protein(g)", "Carbs(g)", "Fat(g)"]]
        .mean()
        .reset_index()
    )

    # Simulated NoSQL storage (JSON)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(avg_macros.to_dict(orient="records"), f, indent=2)

    print(f"[{datetime.now()}] Results saved to {OUTPUT_FILE}")
    return "Data processed and stored successfully."


if __name__ == "__main__":
    print(process_nutritional_data_from_azurite())
