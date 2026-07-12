import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="COVID-19 Chest X-Ray Detection",
    page_icon="🫁",
    layout="wide"
)

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

model = load_model()

# ---------------------------------------------------
# Title
# ---------------------------------------------------
st.title("🫁 COVID-19 Detection from Chest X-Ray")

st.write(
    """
Upload a Chest X-ray image and the trained CNN model will predict whether
the patient is **COVID-19 Positive** or **Normal**.
"""
)

# ---------------------------------------------------
# Upload Image
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

classes = ["COVID", "Normal"]

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=350)

    img = image.resize((299, 299))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    probability = prediction[0][0]

    if probability < 0.5:
        result = classes[0]
        confidence = (1 - probability) * 100
    else:
        result = classes[1]
        confidence = probability * 100

    st.markdown("---")
    st.subheader("Prediction")

    if result == "COVID":
        st.error(f"🦠 {result}")
    else:
        st.success(f"✅ {result}")

    st.progress(float(confidence / 100))

    st.write(f"Confidence : **{confidence:.2f}%**")

# ---------------------------------------------------
# Developer Corner
# ---------------------------------------------------
# ----------------------------------------------------
# Developer Corner
# ----------------------------------------------------

st.markdown("---")
st.header("💻 Developer Info")

st.markdown("""
**Developer:** AKSh
""")




