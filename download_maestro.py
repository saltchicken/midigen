import requests
import zipfile
import io
import shutil
from pathlib import Path
import os

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
DATASET_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
DOWNLOAD_DIR = Path("temp_download")
FINAL_DATA_DIR = Path("midi_data_base")

def download_and_extract():
    print(f"Downloading Maestro Dataset (MIDI only)...")
    print(f"URL: {DATASET_URL}")
    
    # 1. Download
    response = requests.get(DATASET_URL)
    response.raise_for_status()
    print("Download complete. Unzipping...")

    # 2. Unzip in memory
    z = zipfile.ZipFile(io.BytesIO(response.content))
    z.extractall(DOWNLOAD_DIR)
    print("Unzip complete.")

    # 3. Organize
    # Maestro unzips into subfolders by year. We want to flatten this.
    if FINAL_DATA_DIR.exists():
        shutil.rmtree(FINAL_DATA_DIR)
    FINAL_DATA_DIR.mkdir()

    print(f"Organizing files into '{FINAL_DATA_DIR}'...")
    midi_files = list(DOWNLOAD_DIR.rglob("*.midi")) + list(DOWNLOAD_DIR.rglob("*.mid"))
    
    count = 0
    for file in midi_files:
        # Renaissance/Baroque/Classical pieces are great for base models
        dest = FINAL_DATA_DIR / f"maestro_{count:04d}.mid"
        shutil.move(str(file), str(dest))
        count += 1
        
    # 4. Cleanup
    shutil.rmtree(DOWNLOAD_DIR)
    print(f"Success! {count} MIDI files ready for base training.")

if __name__ == "__main__":
    download_and_extract()
