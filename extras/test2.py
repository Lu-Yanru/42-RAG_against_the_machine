import json
from src.ingest.loader import Loader

with open("data/processed/lexical/chunk_metadata.json", encoding="utf-8") as f:
    old_paths = {entry["file_path"] for entry in json.load(f)}

loader = Loader("data/raw")
current_paths = {f.file_path for f in loader.py_files}

missing = current_paths - old_paths
print(f"{len(missing)} current .py files have no entry in chunk_metadata.json")
for p in sorted(missing)[:10]:
    print(" ", repr(p))
