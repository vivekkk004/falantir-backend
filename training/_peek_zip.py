"""Quick inspection of archive.zip structure."""
import sys
import zipfile

ZIP_PATH = sys.argv[1] if len(sys.argv) > 1 else "dataset/archive.zip"

with zipfile.ZipFile(ZIP_PATH) as z:
    names = z.namelist()

top_dirs = set()
second_level = set()
for n in names:
    parts = n.replace("\\", "/").strip("/").split("/")
    if len(parts) >= 1 and parts[0]:
        top_dirs.add(parts[0])
    if len(parts) >= 2 and parts[1]:
        second_level.add(parts[0] + "/" + parts[1])

print(f"Total entries: {len(names)}")
print()
print("Top-level folders:")
for d in sorted(top_dirs):
    print(f"  {d}")
print()

labels_files = [n for n in names if "label" in n.lower() or n.lower().endswith(".csv")]
print(f"CSV/label files ({len(labels_files)} total):")
for lf in labels_files[:20]:
    print(f"  {lf}")
print()

normal = [n for n in names if "normal" in n.lower()]
print(f"'Normal' related entries ({len(normal)} total):")
for n in normal[:10]:
    print(f"  {n}")
