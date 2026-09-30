import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# Paths
# ============================================================

DATASET_PATH = "data/plantvillage_raw/raw/color"
OUTPUT_PATH = "data"

# ============================================================
# 1. Find all classes
# ============================================================

classes = sorted([
    folder
    for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
])

print("Number of classes:", len(classes))

# Create class ID mapping
class_to_id = {
    class_name: index
    for index, class_name in enumerate(classes)
}

# ============================================================
# 2. Collect all image paths
# ============================================================

image_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG"
)

records = []

for class_name in classes:

    class_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    for filename in os.listdir(class_path):

        if filename.endswith(image_extensions):

            image_path = os.path.join(
                class_path,
                filename
            )

            records.append({
                "image_path": image_path,
                "class_name": class_name,
                "class_id": class_to_id[class_name]
            })

# Create DataFrame
df = pd.DataFrame(records)

print("\nTotal images:", len(df))

# ============================================================
# 3. Split into train and temporary set
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["class_id"],
    random_state=42
)

# ============================================================
# 4. Split temporary set into validation and test
# ============================================================

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["class_id"],
    random_state=42
)

# ============================================================
# 5. Reset indexes
# ============================================================

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# ============================================================
# 6. Save CSV files
# ============================================================

train_path = os.path.join(
    OUTPUT_PATH,
    "train.csv"
)

val_path = os.path.join(
    OUTPUT_PATH,
    "validation.csv"
)

test_path = os.path.join(
    OUTPUT_PATH,
    "test.csv"
)

classes_path = os.path.join(
    OUTPUT_PATH,
    "classes.csv"
)

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

# Save class mapping
class_df = pd.DataFrame({
    "class_id": list(range(len(classes))),
    "class_name": classes
})

class_df.to_csv(
    classes_path,
    index=False
)

# ============================================================
# 7. Print results
# ============================================================

print("\n========== DATASET SPLIT ==========")

print("Total images      :", len(df))
print("Training images   :", len(train_df))
print("Validation images :", len(val_df))
print("Testing images    :", len(test_df))

print("\nClass mapping saved to:")
print(classes_path)

print("\nDataset splits saved:")
print(train_path)
print(val_path)
print(test_path)

print("\nDataset preparation completed successfully!")