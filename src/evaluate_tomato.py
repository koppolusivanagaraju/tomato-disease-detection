import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

MODEL_PATH = "models/tomato_disease_cnn.keras"
TEST_CSV = "data/tomato/test.csv"
CLASSES_PATH = "data/tomato/classes.csv"

OUTPUT_DIR = "outputs/tomato"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

print("\nLoading tomato disease model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

test_df = pd.read_csv(
    TEST_CSV
)

classes_df = pd.read_csv(
    CLASSES_PATH
)

class_names = classes_df["class_name"].tolist()

print("Model loaded successfully.")
print(f"Test images: {len(test_df)}")
print(f"Classes: {len(class_names)}")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    return image, label


# ============================================================
# TEST DATASET
# ============================================================

test_dataset = tf.data.Dataset.from_tensor_slices(
    (
        test_df["image_path"].values,
        test_df["class_id"].values
    )
)

test_dataset = test_dataset.map(
    load_image,
    num_parallel_calls=2
)

test_dataset = test_dataset.batch(
    BATCH_SIZE
)

test_dataset = test_dataset.prefetch(1)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    test_dataset,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)

y_true = test_df["class_id"].values


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n========================================")
print("TOMATO DISEASE MODEL EVALUATION")
print("========================================")

print(f"Test Accuracy : {accuracy * 100:.2f}%")
print(f"Precision     : {precision * 100:.2f}%")
print(f"Recall        : {recall * 100:.2f}%")
print(f"F1 Score      : {f1 * 100:.2f}%")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
)

print("\nClassification Report:")
print(report)

with open(
    os.path.join(
        OUTPUT_DIR,
        "classification_report.txt"
    ),
    "w"
) as f:

    f.write(report)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "test_accuracy": float(accuracy),
    "weighted_precision": float(precision),
    "weighted_recall": float(recall),
    "weighted_f1": float(f1),
    "test_images": len(test_df),
    "num_classes": len(class_names)
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "test_metrics.json"
    ),
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(
    figsize=(10, 8)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Tomato Disease Confusion Matrix"
)

plt.colorbar()

tick_marks = np.arange(
    len(class_names)
)

plt.xticks(
    tick_marks,
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    tick_marks,
    class_names
)

plt.xlabel(
    "Predicted Class"
)

plt.ylabel(
    "True Class"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
),
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETED
# ============================================================

print("\n========================================")
print("EVALUATION COMPLETED SUCCESSFULLY")
print("========================================")

print("\nFiles created:")

print(
    "outputs/tomato/test_metrics.json"
)

print(
    "outputs/tomato/classification_report.txt"
)

print(
    "outputs/tomato/confusion_matrix.png"
)