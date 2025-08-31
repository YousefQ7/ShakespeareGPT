#!/usr/bin/env python3
"""
Download model files for ShakespeareGPT deployment.
This script downloads the model files from Google Drive.
"""

import os
import gdown
from pathlib import Path

# Google Drive file IDs (you'll get these from your uploads)
GOOGLE_DRIVE_IDS = {
    "checkpoint.pt": "1T9NJVNIBrZdAVZTG5hgw98Wg2Nisf74w",
    "train.txt": "14qw8JNBZ5sJiJjYOhzD23Ks5ARlpHYuT"
}

# Expected file sizes for verification (Google Drive compressed sizes)
EXPECTED_SIZES = {
    "checkpoint.pt": 124318937,  # ~118.6MB (Google Drive compressed)
    "train.txt": 541096898        # ~516MB (Google Drive compressed)
}

def download_from_gdrive(file_id: str, filename: str, expected_size: int) -> bool:
    """Download a file from Google Drive and verify its size."""
    print(f"📥 Downloading {filename} from Google Drive...")
    
    try:
        # Download using gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filename, quiet=False)
        
        # Verify file size (allow ±1MB variation)
        if os.path.exists(filename):
            actual_size = os.path.getsize(filename)
            size_diff = abs(actual_size - expected_size)
            size_diff_mb = size_diff / (1024 * 1024)
            
            if size_diff_mb <= 1.0:  # Allow ±1MB difference
                print(f"✅ {filename} downloaded successfully ({actual_size} bytes)")
                return True
            else:
                print(f"❌ {filename} size mismatch: expected ~{expected_size}, got {actual_size} (diff: {size_diff_mb:.1f}MB)")
                return False
        else:
            print(f"❌ {filename} was not downloaded")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        return False

def main():
    """Download all model files."""
    print("🚀 Starting model file download from Google Drive...")
    
    success = True
    for filename, file_id in GOOGLE_DRIVE_IDS.items():
        if file_id == "YOUR_CHECKPOINT_FILE_ID_HERE" or file_id == "YOUR_TRAIN_TXT_FILE_ID_HERE":
            print(f"❌ Please update the file ID for {filename} in the script")
            success = False
            continue
            
        expected_size = EXPECTED_SIZES[filename]
        
        if os.path.exists(filename):
            actual_size = os.path.getsize(filename)
            if actual_size == expected_size:
                print(f"✅ {filename} already exists with correct size")
                continue
            else:
                print(f"🔄 {filename} exists but size is wrong, re-downloading...")
                os.remove(filename)
        
        if not download_from_gdrive(file_id, filename, expected_size):
            success = False
    
    if success:
        print("🎉 All model files downloaded successfully!")
    else:
        print("❌ Some model files failed to download")
        exit(1)

if __name__ == "__main__":
    main()
