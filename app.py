
import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os

# Page configuration
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: white;
}
h1, h2, h3 {
    color: #60a5fa;
}
[data-testid="stMetric"] {
    background: #17243a;
    padding: 18px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# Load trained CNN model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "models/digit_model.keras"
    )

if not os.path.exists("models/digit_model.keras"):
    st.error("Trained model not found!")
    st.stop()

model = load_model()

# Header
st.title("🔢 Handwritten Digit Recognition")
st.write(
    "Draw any digit from 0 to 9 and let our CNN model recognize it."
)

st.divider()

# Dashboard metrics
col1, col2, col3 = st.columns(3)

col1.metric("AI Model", "CNN")
col2.metric("Dataset", "MNIST")
col3.metric("Input Image", "28 × 28")

st.divider()

# Drawing section
st.subheader("✍️ Draw Your Digit")

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

left, right = st.columns([1, 1])

with left:
    st.write("Draw inside the black box:")

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.canvas_key}"
    )

    predict_button = st.button(
        "🔍 Predict Digit",
        use_container_width=True
    )

    clear_button = st.button(
        "🗑️ Clear Canvas",
        use_container_width=True
    )

    if clear_button:
        st.session_state.canvas_key += 1
        st.rerun()

# Prediction section
with right:
    st.subheader("🤖 Prediction Result")

    if predict_button:
        if canvas_result.image_data is not None:
            img = canvas_result.image_data

            # Convert drawing into grayscale
            pil_img = Image.fromarray(
                img.astype("uint8")
            ).convert("L")

            # Resize to MNIST dimensions
            pil_img = pil_img.resize(
                (28, 28),
                Image.Resampling.LANCZOS
            )

            # Normalize image
            image_array = np.array(
                pil_img
            ).astype("float32") / 255.0

            # Add batch and channel dimensions
            image_array = image_array.reshape(
                1, 28, 28, 1
            )

            # Predict digit
            predictions = model.predict(
                image_array,
                verbose=0
            )[0]

            predicted_digit = int(
                np.argmax(predictions)
            )

            confidence = float(
                predictions[predicted_digit] * 100
            )

            st.success(
                f"Predicted Digit: {predicted_digit}"
            )

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(
                confidence / 100
            )

            # Display all probabilities
            st.subheader("Digit Probabilities")

            chart_data = pd.DataFrame({
                "Digit": list(range(10)),
                "Confidence (%)": predictions * 100
            })

            st.bar_chart(
                chart_data.set_index("Digit")
            )

            st.image(
                pil_img,
                caption="Processed 28 × 28 Input",
                width=180
            )

        else:
            st.warning("Please draw a digit first.")

    else:
        st.info(
            "Your prediction will appear here after clicking Predict Digit."
        )

st.divider()

# Model evaluation images
st.subheader("📊 Model Performance")

accuracy_path = "models/training_accuracy.png"
matrix_path = "models/confusion_matrix.png"

chart1, chart2 = st.columns(2)

with chart1:
    st.write("Training Accuracy Graph")
    if os.path.exists(accuracy_path):
        st.image(accuracy_path, use_container_width=True)
    else:
        st.warning("Accuracy graph not found.")

with chart2:
    st.write("Confusion Matrix")
    if os.path.exists(matrix_path):
        st.image(matrix_path, use_container_width=True)
    else:
        st.warning("Confusion matrix not found.")

st.divider()

st.caption(
    "Handwritten Digit Recognition using "
    "Convolutional Neural Network (CNN)"
)
