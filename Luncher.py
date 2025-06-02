import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model('D:\Brain-Tumor-Detection-master\.ipynb_checkpoints\your_model.h5')  # Update with your model filename

# Constants
IMG_WIDTH, IMG_HEIGHT = 240, 240

def preprocess_image(img):
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img = np.array(img)
    img = img / 255.0  # normalize
    img = np.expand_dims(img, axis=0)  # model expects batch
    return img

# Title
st.title("🧠 Brain Tumor Detection")

# File uploader
uploaded_file = st.file_uploader("Upload a brain MRI image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = preprocess_image(image)

    # Predict
    prediction = model.predict(img_array)[0][0]  # assumes sigmoid output

    # Output
    if prediction > 0.5:
        st.error(f"🚨 Tumor Detected with {prediction*100:.2f}% confidence.")
    else:
        st.success(f"✅ No Tumor Detected. Confidence: {(1-prediction)*100:.2f}%.")
