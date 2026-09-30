import os
import tensorflow as tf

DATASET_PATH = "data/plantvillage_raw/raw/color"

# Get the first class
classes = sorted([
    folder
    for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
])

first_class = classes[0]

class_path = os.path.join(
    DATASET_PATH,
    first_class
)

# Get the first image
images = [
    file for file in os.listdir(class_path)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
]

image_path = os.path.join(
    class_path,
    images[0]
)

print("Class:", first_class)
print("Image:", image_path)

# Read image
image = tf.io.read_file(image_path)

# Decode image
image = tf.image.decode_image(
    image,
    channels=3,
    expand_animations=False
)

print("Original shape:", image.shape)

# Resize
image = tf.image.resize(
    image,
    (224, 224)
)

# Normalize
image = image / 255.0

print("New shape:", image.shape)
print("Minimum pixel:", tf.reduce_min(image).numpy())
print("Maximum pixel:", tf.reduce_max(image).numpy())

print("\nSingle image test successful!")