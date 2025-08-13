import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load your trained model
# model = tf.keras.models.load_model("cat_dog_classifier.h5")  # change to your filename
model = tf.keras.models.load_model("cat_dog_classifier.h5", compile=False)
# Prediction function


def predict_image(img_pil):
    img_pil = img_pil.resize((250, 250))  # resize to match model input
    img_array = image.img_to_array(img_pil)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize

    prediction = model.predict(img_array)[0][0]
    label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    return label, confidence


# Streamlit UI
st.title("🐾 Cat vs Dog Classifier")
st.write("Upload an image to check whether it's a cat or a dog.")

# Option 1: Upload Image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_pil = Image.open(uploaded_file).convert("RGB")
    label, confidence = predict_image(img_pil)
    st.image(img_pil, caption=f"Prediction: {label} ({confidence:.2f} confidence)")
