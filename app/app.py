import os
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)

GATE_MODEL_PATH = "models/tomato_gate.keras"
DISEASE_MODEL_PATH = "models/tomato_disease_cnn.keras"

CLASSES_PATH = "data/tomato/classes.csv"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tomato Disease Detection",
    page_icon="🍅",
    layout="centered"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    gate_model = tf.keras.models.load_model(
        GATE_MODEL_PATH
    )

    disease_model = tf.keras.models.load_model(
        DISEASE_MODEL_PATH
    )

    return gate_model, disease_model


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_data
def load_classes():

    classes_df = pd.read_csv(
        CLASSES_PATH
    )

    return classes_df["class_name"].tolist()


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):

    image = image.resize(
        IMAGE_SIZE
    )

    image_array = np.array(
        image
    )

    if image_array.ndim == 2:
        image_array = np.stack(
            [image_array] * 3,
            axis=-1
        )

    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]

    image_array = image_array.astype(
        np.float32
    ) / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# PREDICT
# ============================================================

def predict_image(
    image,
    gate_model,
    disease_model,
    class_names
):

    image_array = preprocess_image(
        image
    )

    # --------------------------------------------------------
    # STEP 1: TOMATO / NON-TOMATO
    # --------------------------------------------------------

    tomato_probability = float(
        gate_model.predict(
            image_array,
            verbose=0
        )[0][0]
    )

    # --------------------------------------------------------
    # NOT TOMATO
    # --------------------------------------------------------

    if tomato_probability < 0.50:

        return {
            "is_tomato": False,
            "tomato_probability": tomato_probability,
            "results": []
        }

    # --------------------------------------------------------
    # STEP 2: TOMATO DISEASE
    # --------------------------------------------------------

    predictions = disease_model.predict(
        image_array,
        verbose=0
    )[0]

    top_indices = np.argsort(
        predictions
    )[-3:][::-1]

    results = []

    for index in top_indices:

        results.append(
            (
                class_names[index],
                float(predictions[index])
            )
        )

    return {
        "is_tomato": True,
        "tomato_probability": tomato_probability,
        "results": results
    }


# ============================================================
# APPLICATION
# ============================================================

st.title(
    "🍅 Tomato Disease Detection"
)

st.write(
    "Upload a tomato leaf image to identify the "
    "predicted disease using deep learning."
)

st.caption(
    "The system first checks whether the image is "
    "a tomato leaf and then predicts the disease."
)

st.divider()


# ============================================================
# LOAD MODELS
# ============================================================

try:

    gate_model, disease_model = load_models()

    class_names = load_classes()

except Exception as e:

    st.error(
        "Unable to load the trained models or class names."
    )

    st.exception(e)

    st.stop()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📷 Upload a leaf image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
        "bmp",
        "tif",
        "tiff"
    ]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader(
        "Uploaded Image"
    )

    st.image(
        image,
        caption="Selected leaf image",
        width="stretch"
    )

    if st.button(
        "🔍 Predict Disease",
        type="primary"
    ):

        with st.spinner(
            "Analyzing image..."
        ):

            prediction = predict_image(
                image,
                gate_model,
                disease_model,
                class_names
            )

        # ====================================================
        # NON-TOMATO RESULT
        # ====================================================

        if not prediction["is_tomato"]:

            st.warning(
                "⚠️ This image does not appear to be "
                "a tomato leaf."
            )

            st.info(
                "Please upload a clear tomato leaf image. "
                "The disease classifier supports five "
                "tomato classes."
            )

        # ====================================================
        # TOMATO RESULT
        # ====================================================

        else:

            results = prediction["results"]

            predicted_class = results[0][0]

            confidence = results[0][1]

            tomato_probability = (
                prediction["tomato_probability"]
            )

            st.success(
                "✅ Tomato leaf detected!"
            )

            st.subheader(
                "🌿 Prediction Result"
            )

            st.markdown(
                f"### {predicted_class}"
            )

            st.metric(
                "Disease Prediction Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.metric(
                "Tomato Detection Confidence",
                f"{tomato_probability * 100:.2f}%"
            )

            # =================================================
            # LOW CONFIDENCE WARNING
            # =================================================

            if confidence < 0.70:

                st.warning(
                    "⚠️ The disease prediction has low "
                    "confidence. Try uploading a clearer "
                    "tomato leaf image."
                )

            else:

                st.success(
                    "The model is confident in this "
                    "disease prediction."
                )

            # =================================================
            # TOP 3
            # =================================================

            st.subheader(
                "📊 Top 3 Predictions"
            )

            for rank, (
                class_name,
                probability
            ) in enumerate(
                results,
                start=1
            ):

                st.write(
                    f"**{rank}. {class_name}**"
                )

                st.progress(
                    probability,
                    text=f"{probability * 100:.2f}%"
                )

            # =================================================
            # INFORMATION
            # =================================================

            st.info(
                "This model was trained on the PlantVillage "
                "dataset using five tomato classes: Healthy, "
                "Early Blight, Late Blight, Leaf Mold, and "
                "Septoria Leaf Spot. Results may be less "
                "reliable for real-world field images that "
                "differ significantly from the training data."
            )


# ============================================================
# SUPPORTED CLASSES
# ============================================================

with st.expander(
    "📋 Supported Tomato Classes"
):

    for class_name in class_names:

        st.write(
            "• " + class_name
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Tomato Disease Detection • CNN • TensorFlow • "
    "Keras • Streamlit • PlantVillage"
)