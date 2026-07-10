import streamlit as st
import numpy as np
from PIL import Image
import joblib
import time

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Gender Classifier",
    page_icon="👤",
    layout="centered"
)

# Custom CSS for a cleaner, richer UI
st.markdown("""
    <style>
    .prediction-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    div[data-testid="stImage"] img {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Load Model (Cached for performance)
# -------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load("gender_model.pkl")
    except FileNotFoundError:
        return None

model = load_model()
IMG_SIZE = 64

# -------------------------
# UI Header
# -------------------------
st.title("👤 Male vs Female Classifier")
st.markdown("Upload a facial image, and the model will predict the gender using Logistic Regression.")
st.divider()

if model is None:
    st.error("⚠️ **Model file not found!** Please run `train_model.py` first to generate `gender_model.pkl`.")
    st.stop()

# -------------------------
# Upload Image
# -------------------------
uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    # Use columns to split the image and the results side-by-side
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("### Uploaded Image")
        # Read and display image
        image = Image.open(uploaded_file)
        image = image.convert("RGB")
        st.image(image, use_container_width=True)

    with col2:
        st.markdown("### Analysis")
        
        # Add a small loading spinner for UI polish
        with st.spinner("Analyzing image..."):
            time.sleep(0.5) # Artificial delay for effect
            
            # Preprocess image
            resized = image.resize((IMG_SIZE, IMG_SIZE))
            resized_arr = np.array(resized).flatten()

            # Prediction
            prediction = model.predict([resized_arr])[0]
            probability = model.predict_proba([resized_arr])[0]

            # 0 aligns with "Male", 1 aligns with "Female" based on the array order in the training script
            if prediction == 0:
                st.success("👨 Prediction: **MALE**")
            else:
                st.success("👩 Prediction: **FEMALE**")

            # Display rich probabilities
            st.markdown("#### Confidence Scores")
            
            male_prob = probability[0] * 100
            female_prob = probability[1] * 100

            st.write(f"👨 **Male Probability:** {male_prob:.2f}%")
            st.progress(int(probability[0] * 100))

            st.write(f"👩 **Female Probability:** {female_prob:.2f}%")
            st.progress(int(probability[1] * 100))