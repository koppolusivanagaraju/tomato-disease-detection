import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

MODEL_PATH = "models/tomato_gate.keras"
TEST_CSV = "data/tomato_gate/test.csv"

OUTPUT_DIR = "outputs/tomato_gate"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD MODEL AND TEST DATA
# ============================================================

print("\nLoading tomato gate model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

test_df = pd.read_csv(
    TEST_CSV
)

print("Model loaded successfully.")
print(f"Test images: {len(test_df)}")


# ============================================================
# IMAGE LOADING
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
        test_df["label"].values
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
# PREDICTION
# ============================================================

print("\nGenerating gate predictions...")

probabilities = model.predict(
    test_dataset,
    verbose=1
).flatten()

y_true = test_df["label"].values

# Probability >= 0.5 means Tomato
y_pred = (
    probabilities >= 0.5
).astype(int)


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
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("TOMATO / NON-TOMATO GATE EVALUATION")
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
    target_names=[
        "Non-Tomato",
        "Tomato"
    ],
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
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "test_images": len(test_df)
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
    figsize=(7, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Tomato / Non-Tomato Gate Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    [0, 1],
    ["Non-Tomato", "Tomato"]
)

plt.yticks(
    [0, 1],
    ["Non-Tomato", "Tomato"]
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    ),
    dpi=150
)

plt.close()


print("\n========================================")
print("GATE EVALUATION COMPLETED")
print("========================================")

print("\nFiles created:")
print("outputs/tomato_gate/test_metrics.json")
print("outputs/tomato_gate/classification_report.txt")
print("outputs/tomato_gate/confusion_matrix.png")