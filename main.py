import streamlit as st
import numpy as np
from PIL import Image
import joblib
from tensorflow.keras.models import load_model

model = load_model("model.h5", compile=False)
CLASS_NAMES = ['Early Blight','Late Blight','Healthy']

st.title("🖼️ Image Classification App")

st.write("Upload an image to make a prediction using your trained model.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_array = np.array(image)

    st.write("✅ Image converted to NumPy array with shape:", img_array.shape)

    if st.button("Predict"):
        try:
            # Ensure input is properly shaped for your model
            X_input = np.expand_dims(img_array, axis=0)

            # Predict using model
            y_pred = model.predict(X_input)
            predicted_index = np.argmax(y_pred[0])

            st.success(f"🎯 Model Prediction: {CLASS_NAMES[predicted_index]}")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

