import os
import json
import pandas as pd
import tensorflow as tf

from model import build_baseline_cnn


# ============================================================
# Configuration
# ============================================================

EPOCHS = 10
BATCH_SIZE = 32

MODEL_PATH = "models/baseline_cnn.keras"


# ============================================================
# Load CSV files
# ============================================================

train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/validation.csv")

print("Training images:", len(train_df))
print("Validation images:", len(val_df))


# ============================================================
# Image preprocessing
# ============================================================

def load_and_preprocess_image(image_path, label):

    image = tf.io.read_file(image_path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        (224, 224)
    )

    image = tf.cast(
        image,
        tf.float32
    ) / 255.0

    return image, label


# ============================================================
# Create TensorFlow dataset
# ============================================================

def create_dataset(dataframe, shuffle=False):

    image_paths = dataframe["image_path"].values
    labels = dataframe["class_id"].values

    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=1000,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=2
    )

    dataset = dataset.batch(
        BATCH_SIZE
    )

    dataset = dataset.prefetch(1)

    return dataset


train_dataset = create_dataset(
    train_df,
    shuffle=True
)

validation_dataset = create_dataset(
    val_df
)


# ============================================================
# Build model
# ============================================================

model = build_baseline_cnn(
    num_classes=38
)

print("\nModel created successfully!")

model.summary()


# ============================================================
# Create models directory
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


# ============================================================
# Callbacks
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)


# ============================================================
# Train
# ============================================================

print("\n========== STARTING TRAINING ==========\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)


# ============================================================
# Save training history
# ============================================================

history_path = "models/training_history.json"

with open(history_path, "w") as file:

    json.dump(
        history.history,
        file
    )


print("\n========== TRAINING COMPLETED ==========")

print("Best model saved to:")
print(MODEL_PATH)

print("\nTraining history saved to:")
print(history_path)