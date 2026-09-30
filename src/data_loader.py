import pandas as pd
import tensorflow as tf

# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 38

# ============================================================
# Load CSV files
# ============================================================

train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/validation.csv")
test_df = pd.read_csv("data/test.csv")

print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Testing images:", len(test_df))


# ============================================================
# Image loading and preprocessing
# ============================================================

def load_and_preprocess_image(image_path, label):

    image = tf.io.read_file(image_path)

    image = tf.image.decode_jpeg(
        image,
        channels=3
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(image, tf.float32) / 255.0

    return image, label


# ============================================================
# Create dataset
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


# ============================================================
# Create datasets
# ============================================================

train_dataset = create_dataset(
    train_df,
    shuffle=True
)

validation_dataset = create_dataset(
    val_df
)

test_dataset = create_dataset(
    test_df
)

print("\nTensorFlow datasets created successfully!")

print("Training batches:",
      tf.data.experimental.cardinality(train_dataset).numpy())

print("Validation batches:",
      tf.data.experimental.cardinality(validation_dataset).numpy())

print("Testing batches:",
      tf.data.experimental.cardinality(test_dataset).numpy())

print("\nData pipeline setup successful!")