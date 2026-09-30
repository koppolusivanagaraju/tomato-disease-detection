import tensorflow as tf
from tensorflow.keras import layers, models


def build_baseline_cnn(num_classes=38):

    model = models.Sequential([

        # Input
        layers.Input(shape=(224, 224, 3)),

        # CNN Block 1
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu"
        ),
        layers.MaxPooling2D((2, 2)),

        # CNN Block 2
        layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),
        layers.MaxPooling2D((2, 2)),

        # CNN Block 3
        layers.Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),
        layers.MaxPooling2D((2, 2)),

        # Reduce feature maps
        layers.GlobalAveragePooling2D(),

        # Fully connected layer
        layers.Dense(
            128,
            activation="relu"
        ),

        # Reduce overfitting
        layers.Dropout(0.5),

        # 38 classes
        layers.Dense(
            num_classes,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


model = build_baseline_cnn()

model.summary()