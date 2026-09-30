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
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ==============================
# CONFIGURATION
# ==============================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

MODEL_PATH = "models/plant_disease_mobilenetv2.keras"
TEST_CSV = "data/test.csv"
CLASSES_CSV = "data/classes.csv"

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# LOAD CLASS NAMES
# ==============================

classes_df = pd.read_csv(CLASSES_CSV)

print("\nClasses file:")
print(classes_df.head())

# Get class names safely
if "class_name" in classes_df.columns:
    class_names = classes_df["class_name"].tolist()
else:
    class_names = classes_df.iloc[:, 1].tolist()

print("\nNumber of classes:", len(class_names))


# ==============================
# LOAD TEST DATA
# ==============================

test_df = pd.read_csv(TEST_CSV)

print("\nTest images:", len(test_df))
print("Test CSV columns:", list(test_df.columns))


# ==============================
# IMAGE PREPROCESSING
# ==============================

def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
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


# ==============================
# CREATE TEST DATASET
# ==============================

test_dataset = tf.data.Dataset.from_tensor_slices(
    (
        test_df["image_path"].values,
        test_df["class_id"].values
    )
)

test_dataset = test_dataset.map(
    load_and_preprocess_image,
    num_parallel_calls=2
)

test_dataset = test_dataset.batch(
    BATCH_SIZE
)

test_dataset = test_dataset.prefetch(1)


# ==============================
# LOAD TRAINED MODEL
# ==============================

print("\nLoading trained model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ==============================
# PREDICTIONS
# ==============================

print("\nRunning predictions on test dataset...")
print("Please wait...")

y_true = []
y_pred = []

for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ==============================
# METRICS
# ==============================

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


# ==============================
# PRINT RESULTS
# ==============================

print("\n" + "=" * 60)
print("PLANT DISEASE MODEL EVALUATION")
print("=" * 60)

print(f"Test Accuracy : {accuracy * 100:.2f}%")
print(f"Precision     : {precision * 100:.2f}%")
print(f"Recall        : {recall * 100:.2f}%")
print(f"F1 Score      : {f1 * 100:.2f}%")

print("=" * 60)


# ==============================
# CLASSIFICATION REPORT
# ==============================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
)

print("\nClassification Report:\n")
print(report)


# ==============================
# SAVE METRICS
# ==============================

metrics = {
    "test_accuracy": float(accuracy),
    "weighted_precision": float(precision),
    "weighted_recall": float(recall),
    "weighted_f1": float(f1),
    "test_images": int(len(y_true)),
    "num_classes": int(len(class_names))
}

with open(
    os.path.join(OUTPUT_DIR, "test_metrics.json"),
    "w"
) as f:
    json.dump(
        metrics,
        f,
        indent=4
    )


# ==============================
# SAVE CLASSIFICATION REPORT
# ==============================

with open(
    os.path.join(OUTPUT_DIR, "classification_report.txt"),
    "w"
) as f:

    f.write(
        "Plant Disease Detection - Classification Report\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        f"Test Accuracy : {accuracy * 100:.2f}%\n"
    )

    f.write(
        f"Precision     : {precision * 100:.2f}%\n"
    )

    f.write(
        f"Recall        : {recall * 100:.2f}%\n"
    )

    f.write(
        f"F1 Score      : {f1 * 100:.2f}%\n\n"
    )

    f.write(
        report
    )


# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(
    figsize=(18, 16)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Plant Disease Classification - Confusion Matrix"
)

plt.colorbar()

tick_marks = np.arange(
    len(class_names)
)

plt.xticks(
    tick_marks,
    class_names,
    rotation=90,
    fontsize=7
)

plt.yticks(
    tick_marks,
    class_names,
    fontsize=7
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
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ==============================
# FINISHED
# ==============================

print("\nEvaluation completed successfully.")

print("\nFiles created:")

print("outputs/test_metrics.json")
print("outputs/classification_report.txt")
print("outputs/confusion_matrix.png")