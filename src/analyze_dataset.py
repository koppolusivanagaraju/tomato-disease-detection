import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# Dataset location
DATASET_PATH = "data/plantvillage_raw/raw/color"
OUTPUT_PATH = "outputs"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# --------------------------------------------------
# 1. Find all classes
# --------------------------------------------------

classes = sorted([
    folder for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
])

print("Number of classes:", len(classes))

# --------------------------------------------------
# 2. Count images in every class
# --------------------------------------------------

image_extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

records = []

for class_name in classes:

    class_path = os.path.join(DATASET_PATH, class_name)

    images = [
        file for file in os.listdir(class_path)
        if file.endswith(image_extensions)
    ]

    records.append({
        "class": class_name,
        "image_count": len(images)
    })

df = pd.DataFrame(records)

# Sort by image count
df = df.sort_values("image_count", ascending=False)

# Save CSV
csv_path = os.path.join(OUTPUT_PATH, "class_distribution.csv")
df.to_csv(csv_path, index=False)

# --------------------------------------------------
# 3. Print dataset statistics
# --------------------------------------------------

total_images = df["image_count"].sum()

print("\n========== DATASET REPORT ==========")
print("Number of classes :", len(df))
print("Total images      :", total_images)
print("Largest class     :", df.iloc[0]["class"])
print("Largest count     :", df.iloc[0]["image_count"])
print("Smallest class    :", df.iloc[-1]["class"])
print("Smallest count    :", df.iloc[-1]["image_count"])

print("\nClass distribution:")
print(df.to_string(index=False))

# --------------------------------------------------
# 4. Create class distribution graph
# --------------------------------------------------

plt.figure(figsize=(14, 10))

plt.barh(df["class"], df["image_count"])

plt.xlabel("Number of Images")
plt.ylabel("Plant Disease Class")
plt.title("PlantVillage Class Distribution")

plt.tight_layout()

graph_path = os.path.join(
    OUTPUT_PATH,
    "class_distribution.png"
)

plt.savefig(graph_path, dpi=150)
plt.close()

print("\nClass distribution graph saved to:")
print(graph_path)

# --------------------------------------------------
# 5. Create sample image grid
# --------------------------------------------------

sample_classes = classes[:12]

fig, axes = plt.subplots(
    3,
    4,
    figsize=(14, 10)
)

axes = axes.flatten()

for i, class_name in enumerate(sample_classes):

    class_path = os.path.join(DATASET_PATH, class_name)

    images = [
        file for file in os.listdir(class_path)
        if file.endswith(image_extensions)
    ]

    if not images:
        continue

    image_path = os.path.join(
        class_path,
        images[0]
    )

    image = Image.open(image_path)

    axes[i].imshow(image)
    axes[i].set_title(class_name, fontsize=8)
    axes[i].axis("off")

plt.tight_layout()

sample_path = os.path.join(
    OUTPUT_PATH,
    "sample_images.png"
)

plt.savefig(sample_path, dpi=150)
plt.close()

print("Sample images saved to:")
print(sample_path)

print("\nDataset analysis completed successfully!")