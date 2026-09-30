import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

MODEL_PATH = "models/plant_disease_mobilenetv2.keras"

TRAIN_CSV = "data/train.csv"
VALIDATION_CSV = "data/validation.csv"
CLASSES_CSV = "data/classes.csv"

os.makedirs("models", exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

train_df = pd.read_csv(TRAIN_CSV)
validation_df = pd.read_csv(VALIDATION_CSV)
classes_df = pd.read_csv(CLASSES_CSV)

NUM_CLASSES = len(classes_df)

print("Training images:", len(train_df))
print("Validation images:", len(validation_df))
print("Number of classes:", NUM_CLASSES)

# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_image(path, label):

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
    )

    image = preprocess_input(image)

    return image, label


# ============================================================
# TRAIN DATASET
# ============================================================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        train_df["image_path"].values,
        train_df["class_id"].values
    )
)

train_dataset = train_dataset.shuffle(
    2000,
    seed=42
)

train_dataset = train_dataset.map(
    load_image,
    num_parallel_calls=2
)

train_dataset = train_dataset.batch(
    BATCH_SIZE
)

train_dataset = train_dataset.prefetch(1)

# ============================================================
# VALIDATION DATASET
# ============================================================

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (
        validation_df["image_path"].values,
        validation_df["class_id"].values
    )
)

validation_dataset = validation_dataset.map(
    load_image,
    num_parallel_calls=2
)

validation_dataset = validation_dataset.batch(
    BATCH_SIZE
)

validation_dataset = validation_dataset.prefetch(1)

# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_df["class_id"].values
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(class_weights_array)
}

print("Class weights calculated.")

# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.10),
    ],
    name="data_augmentation"
)

# ============================================================
# MOBILENETV2
# ============================================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# ============================================================
# BUILD MODEL
# ============================================================

inputs = layers.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dense(
    256,
    activation="relu"
)(x)

x = layers.Dropout(
    0.4
)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel created successfully.")

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=1,
    min_lr=1e-6,
    verbose=1
)

# ============================================================
# TRAIN
# ============================================================

print("\nStarting MobileNetV2 transfer learning...")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)

# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

with open(
    "models/mobilenetv2_training_history.json",
    "w"
) as f:

    json.dump(
        history.history,
        f,
        indent=4
    )

print("\n" + "=" * 60)
print("TRANSFER LEARNING TRAINING COMPLETED")
print("=" * 60)

print("\nBest model:")
print(MODEL_PATH)

print("\nTraining history:")
print("models/mobilenetv2_training_history.json")