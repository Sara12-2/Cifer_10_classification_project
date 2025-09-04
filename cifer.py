import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# -----------------------------------------------------------
# Load trained CNN model
# -----------------------------------------------------------
model = load_model("cifar10_cnn_model.h5")

# CIFAR-10 Classes
class_names = ['✈ Airplane', '🚗 Automobile', '🐦 Bird', '🐱 Cat', '🦌 Deer', 
               '🐶 Dog', '🐸 Frog', '🚢 Ship', '🚚 Truck', '🐠 Fish']

# -----------------------------------------------------------
# Streamlit App UI
# -----------------------------------------------------------
st.set_page_config(page_title="CIFAR-10 Classifier", page_icon="🤖", layout="wide")

st.title("📸 CIFAR-10 Image Classification")
st.write("Upload an image (32x32 or bigger) and the model will classify it into one of the CIFAR-10 categories.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = img.resize((32, 32))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.subheader("🔎 Prediction Result")
    st.success(f"*Class:* {class_names[class_index]} \n\n *Confidence:* {confidence:.2f}%")

    # Show probability chart
    st.subheader("📊 Prediction Probabilities")
    st.bar_chart(prediction[0])