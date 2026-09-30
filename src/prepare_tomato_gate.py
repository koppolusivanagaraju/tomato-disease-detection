import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = r"D:\s2\data\plantvillage_raw\raw\color"
OUTPUT_DIR = r"D:\s2\data\tomato_gate"

TOMATO_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot"
]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".webp", ".bmp", ".tif", ".tiff"
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

rows = []

# ============================================================
# TOMATO = 1
# ============================================================

for class_name in TOMATO_CLASSES:

    class_dir = os.path.join(BASE_DIR, class_name)

    for filename in os.listdir(class_dir):

        path = os.path.join(class_dir, filename)

        if os.path.isfile(path):

            ext = os.path.splitext(filename)[1].lower()

            if ext in IMAGE_EXTENSIONS:

                rows.append({
                    "image_path": path,
                    "label": 1
                })


# ============================================================
# NON-TOMATO = 0
# ============================================================

non_tomato_count = 0

for class_name in os.listdir(BASE_DIR):

    if class_name in TOMATO_CLASSES:
        continue

    class_dir = os.path.join(BASE_DIR, class_name)

    if not os.path.isdir(class_dir):
        continue

    for filename in os.listdir(class_dir):

        path = os.path.join(class_dir, filename)

        if os.path.isfile(path):

            ext = os.path.splitext(filename)[1].lower()

            if ext in IMAGE_EXTENSIONS:

                rows.append({
                    "image_path": path,
                    "label": 0
                })

                non_tomato_count += 1

                # Keep the gate dataset balanced
                if non_tomato_count >= 7231:
                    break

    if non_tomato_count >= 7231:
        break


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(rows)

print("\n========================================")
print("TOMATO / NON-TOMATO GATE DATASET")
print("========================================")

print(f"Total images: {len(df)}")

print("\nClass distribution:")
print(df["label"].value_counts())


# ============================================================
# SPLIT
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["label"],
    random_state=42
)

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)


# ============================================================
# SAVE
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

print("\n========================================")
print("GATE DATASET CREATED")
print("========================================")

print(f"Training   : {len(train_df)}")
print(f"Validation : {len(validation_df)}")
print(f"Testing    : {len(test_df)}")