# scripts/download_data.py

import os
import requests
import json
import subprocess

BASE_URI = "ipfs://bafybeibawzhxy5iu4jtinkldgczwt43jsufah36m4zl5b7zykfsj5sx3uu/"
LOCAL_IPFS_GATEWAY = "http://localhost:8080/ipfs/"
OUTPUT_DIR = "../data/anky_genesis_collection"

def download_metadata_and_images(max_supply=8888):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for token_id in range(120, max_supply + 1):
        metadata_url = f"{LOCAL_IPFS_GATEWAY}bafybeibawzhxy5iu4jtinkldgczwt43jsufah36m4zl5b7zykfsj5sx3uu/{token_id}"
        response = requests.get(metadata_url)
        metadata = response.json()

        # Save metadata
        metadata_path = os.path.join(OUTPUT_DIR, f"{token_id}.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        # Download image
        image_url = metadata["image"].replace("ipfs://", LOCAL_IPFS_GATEWAY)
        image_response = requests.get(image_url)
        image_path = os.path.join(OUTPUT_DIR, f"{token_id}.png")
        with open(image_path, "wb") as f:
            f.write(image_response.content)

        print(f"Downloaded {token_id}/{max_supply}")

if __name__ == "__main__":
    download_metadata_and_images()
