import os
import shutil
import zipfile
from pathlib import Path

raw_dir = Path("data/raw")
zip_path = Path("safebit_aml_datasets.zip")
art_zip = Path(r"C:\Users\bhuva\.gemini\antigravity-ide\brain\27eb4491-392f-4ac1-b938-042b73917f55\safebit_aml_datasets.zip")

print("Zipping datasets from:", raw_dir.resolve())
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for file_path in raw_dir.rglob("*"):
        if file_path.is_file():
            arcname = file_path.relative_to(raw_dir)
            zf.write(file_path, arcname)
            print(f"  + {arcname} ({file_path.stat().st_size:,} bytes)")

size_bytes = zip_path.stat().st_size
print(f"Successfully generated {zip_path} ({size_bytes:,} bytes / {size_bytes / (1024*1024):.2f} MB)")

shutil.copy(zip_path, art_zip)
print(f"Copied to artifact directory: {art_zip}")
