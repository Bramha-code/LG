import os
import shutil
import re
from pathlib import Path

# Source and destination directories
source_dir = Path(r"C:\Users\Bramha.nimbalkar\Desktop\LG-2\audio")
dest_dir = Path(r"C:\Users\Bramha.nimbalkar\Desktop\LG-2\audio_classified")

# Create destination directory
dest_dir.mkdir(exist_ok=True)

# Get all wav files
wav_files = list(source_dir.glob("*.wav"))
print(f"Found {len(wav_files)} wav files")

# Dictionary to track classes and file counts
class_counts = {}

for wav_file in wav_files:
    filename = wav_file.name

    # Extract class name (everything before the last underscore followed by numbers)
    # Pattern: ClassName_timestamp.wav
    match = re.match(r"(.+?)_\d+\.wav$", filename)

    if match:
        class_name = match.group(1)
    else:
        # Fallback: use the part before the first underscore
        class_name = filename.split("_")[0] if "_" in filename else "unknown"

    # Clean class name (replace spaces with underscores for folder names)
    class_folder_name = class_name.replace(" ", "_")

    # Create class folder
    class_folder = dest_dir / class_folder_name
    class_folder.mkdir(exist_ok=True)

    # Copy file to class folder
    dest_path = class_folder / filename
    shutil.copy2(wav_file, dest_path)

    # Track counts
    class_counts[class_folder_name] = class_counts.get(class_folder_name, 0) + 1

# Print summary
print(f"\nOrganized into {len(class_counts)} classes:")
print("-" * 40)
for class_name, count in sorted(class_counts.items()):
    print(f"{class_name}: {count} files")
print("-" * 40)
print(f"Total files: {sum(class_counts.values())}")
print(f"\nOutput directory: {dest_dir}")
