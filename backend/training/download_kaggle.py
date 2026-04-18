"""Download the Kaggle ASL Alphabet dataset into data/asl_alphabet_train/.

Requires the kaggle CLI and an API token. Set up once:
    1. https://www.kaggle.com/settings/account -> "Create New API Token"
    2. Save kaggle.json to:
         Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json
         macOS/Linux: ~/.kaggle/kaggle.json
    3. pip install kaggle

Then:
    python training/download_kaggle.py

If the CLI isn't available, this script prints manual download instructions.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
TARGET = DATA / "asl_alphabet_train"

DATASET = "grassknoted/asl-alphabet"


def manual_instructions():
    print("""
The `kaggle` CLI is not installed or not configured. Either:

  (a) Install + configure:
      pip install kaggle
      # put kaggle.json from https://www.kaggle.com/settings/account into ~/.kaggle/
      python training/download_kaggle.py

  (b) Download manually:
      1. Go to https://www.kaggle.com/datasets/grassknoted/asl-alphabet
      2. Click "Download" (~1GB zip)
      3. Extract into: {data}
      4. You should end up with: {target}/A/, {target}/B/, ...
""".format(data=DATA, target=TARGET))


def main():
    if importlib.util.find_spec("kaggle") is None:
        manual_instructions()
        sys.exit(1)

    DATA.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET} to {DATA}")

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    try:
        api.dataset_download_files(DATASET, path=str(DATA), unzip=True, quiet=False)
    except Exception as e:
        print(f"Kaggle API call failed: {e}")
        manual_instructions()
        sys.exit(1)

    if not TARGET.exists():
        nested = DATA / "asl_alphabet_train" / "asl_alphabet_train"
        if nested.exists():
            for item in nested.iterdir():
                item.rename(TARGET / item.name)
            nested.rmdir()

    if TARGET.exists():
        classes = sorted(p.name for p in TARGET.iterdir() if p.is_dir())
        print(f"Ready: {TARGET} has {len(classes)} classes: {classes}")
    else:
        print(f"Download finished but {TARGET} not found. Check {DATA} manually.")


if __name__ == "__main__":
    main()
