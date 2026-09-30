from huggingface_hub import hf_hub_download
import zipfile
import os

print("Starting PlantVillage dataset download...")

# Download the official PlantVillage data.zip
zip_path = hf_hub_download(
    repo_id="mohanty/PlantVillage",
    filename="data.zip",
    repo_type="dataset"
)

print("\nDataset ZIP downloaded!")
print("ZIP location:", zip_path)

# Create extraction folder
extract_path = os.path.join("data", "plantvillage_raw")
os.makedirs(extract_path, exist_ok=True)

print("\nExtracting dataset...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)

print("\nPlantVillage dataset extracted successfully!")
print("Location:", extract_path)