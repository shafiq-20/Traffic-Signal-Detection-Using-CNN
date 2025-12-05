import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import pyttsx3
import threading

# Load trained model
model = load_model("traffic_model.h5")

# Class labels must match training generator order
# Replace this list with actual output from train_generator.class_indices
# Example: {'green': 0, 'red': 1, 'yellow': 2}
class_names = ['green', 'red', 'yellow']

# Voice assistant
def speak(text):
    def _speak():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_speak).start()

# Preprocess input image
def preprocess_image(image):
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# Predict function
def predict_image(image):
    input_img = preprocess_image(image)
    prediction = model.predict(input_img)
    class_idx = np.argmax(prediction)
    label = class_names[class_idx]
    confidence = np.max(prediction) * 100
    return label, confidence

# Streamlit app layout
st.set_page_config(page_title="🚦 Traffic Signal Detector", page_icon="🚦")
st.title("🚦 Smart Traffic Signal Detector")
st.write("Upload an image of a traffic light to identify its color.")

uploaded_file = st.file_uploader("📁 Upload a traffic signal image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🚥 Predict Signal"):
        try:
            label, confidence = predict_image(image)

            if label == "red":
                st.error(f"🟥 RED signal detected! Please STOP. ({confidence:.2f}%)")
                speak("Red signal detected. Please stop.")
            elif label == "yellow":
                st.warning(f"🟨 YELLOW signal detected! Be READY. ({confidence:.2f}%)")
                speak("Yellow signal detected. Be ready.")
            elif label == "green":
                st.success(f"🟩 GREEN signal detected! You may GO. ({confidence:.2f}%)")
                speak("Green signal detected. You may go.")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
