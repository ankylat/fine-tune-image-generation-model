# scripts/structure_metadata.py

import json
import os

DATA_DIR = "../data/anky_genesis_collection"

def structure_metadata(data_dir):
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

    for json_file in json_files:
        with open(os.path.join(data_dir, json_file), "r") as f:
            metadata = json.load(f)

        structured_metadata = {
            "name": metadata.get("name", ""),
            "description": metadata.get("description", ""),
            "attributes": metadata.get("attributes", []),
            "image": json_file.replace('.json', '.png')
        }

        with open(os.path.join(data_dir, json_file), "w") as f:
            json.dump(structured_metadata, f, indent=2)

    print(f"Structured {len(json_files)} metadata files.")

if __name__ == "__main__":
    structure_metadata(DATA_DIR)