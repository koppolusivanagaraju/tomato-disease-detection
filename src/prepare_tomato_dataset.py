import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = r"D:\s2\data\plantvillage_raw\raw\color"

OUTPUT_DIR = r"D:\s2\data\tomato"

CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot"
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# COLLECT IMAGES
# ============================================================

rows = []

for class_id, class_name in enumerate(CLASSES):

    class_dir = os.path.join(BASE_DIR, class_name)

    if not os.path.exists(class_dir):
        print(f"ERROR: Folder not found: {class_dir}")
        continue

    for filename in os.listdir(class_dir):

        file_path = os.path.join(class_dir, filename)

        if os.path.isfile(file_path):

            extension = os.path.splitext(filename)[1].lower()

            if extension in IMAGE_EXTENSIONS:

                rows.append({
                    "image_path": file_path,
                    "class_name": class_name,
                    "class_id": class_id
                })


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(rows)

print("\n========================================")
print("TOMATO DATASET")
print("========================================")

print(f"Total images: {len(df)}")
print(f"Number of classes: {len(CLASSES)}")

print("\nClass distribution:")
print(df["class_name"].value_counts())


# ============================================================
# STRATIFIED SPLIT
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["class_id"],
    random_state=42
)

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["class_id"],
    random_state=42
)


# ============================================================
# SAVE FILES
# ============================================================

train_df.to_csv(
    os.path.join(OUTPUT_DIR, "train.csv"),
    index=False
)

validation_df.to_csv(
    os.path.join(OUTPUT_DIR, "validation.csv"),
    index=False
)

test_df.to_csv(
    os.path.join(OUTPUT_DIR, "test.csv"),
    index=False
)

pd.DataFrame({
    "class_id": range(len(CLASSES)),
    "class_name": CLASSES
}).to_csv(
    os.path.join(OUTPUT_DIR, "classes.csv"),
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("DATASET SPLIT COMPLETED")
print("========================================")

print(f"Training images   : {len(train_df)}")
print(f"Validation images : {len(validation_df)}")
print(f"Testing images    : {len(test_df)}")

print("\nFiles created:")
print("data/tomato/train.csv")
print("data/tomato/validation.csv")
print("data/tomato/test.csv")
print("data/tomato/classes.csv")