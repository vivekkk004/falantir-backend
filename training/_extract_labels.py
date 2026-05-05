"""Extract only the Labels CSVs from archive.zip."""
import os
import zipfile
from pathlib import Path

ARCHIVE = "dataset/archive.zip"
DEST_ROOT = Path("dataset/Labels")
DEST_ROOT.mkdir(parents=True, exist_ok=True)

extracted = 0
with zipfile.ZipFile(ARCHIVE) as z:
    for member in z.namelist():
        norm = member.replace("\\", "/")
        if "Labels/" in norm and norm.lower().endswith(".csv"):
            filename = norm.split("/")[-1]
            dest = DEST_ROOT / filename
            # Only extract once even if there are duplicates in the zip
            if dest.exists():
                continue
            with z.open(member) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            extracted += 1

print(f"Extracted {extracted} CSV files to {DEST_ROOT}")
print()
print("dataset/Labels/ contents:")
for f in sorted(os.listdir(DEST_ROOT)):
    size = os.path.getsize(DEST_ROOT / f)
    print(f"  {f:<30} {size:>8} bytes")
