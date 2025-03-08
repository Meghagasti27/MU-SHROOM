import os
import zipfile

# Paths to your datasets
data_folders = [
    "/kaggle/input/test-labeled",
    "/kaggle/input/test-unlabeled",
    "/kaggle/input/train-data"
]

output_path = "/kaggle/working/"  # Extracting files here

# Unzip all files
for folder in data_folders:
    if os.path.exists(folder):  # Check if the folder exists
        for file in os.listdir(folder):
            if file.endswith(".zip"):
                file_path = os.path.join(folder, file)
                extract_path = os.path.join(output_path, os.path.splitext(file)[0])  # Extract to a subfolder

                if not os.path.exists(extract_path):  # Avoid re-extracting
                    try:
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            zip_ref.extractall(extract_path)
                        print(f"Extracted: {file} to {extract_path}")
                    except zipfile.BadZipFile:
                        print(f"Error: Corrupted zip file - {file}")
                else:
                    print(f"Skipping extraction (already exists): {file}")

print("All datasets processed successfully!")

