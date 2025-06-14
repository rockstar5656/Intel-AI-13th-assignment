import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------
# Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="🛠️ Anomaly Classifier", layout="centered")

st.markdown(
    "<h1 style='text-align: center; color: #2196F3;'>🔍 Surface Defect Detector</h1>",
    unsafe_allow_html=True
)

st.markdown("Upload or capture an image to classify it into one of the following:")
st.markdown("**🟢 good** (Normal) or **🔴 cut**, **fold**, **glue**, **poke**, **color** (Anomalies)")

# ---------------------------
# Class labels (must match model training order)
# ---------------------------
class_names = ["color", "cut", "fold", "glue", "good", "poke"]

# ---------------------------
# Image Upload / Capture
# ---------------------------
st.markdown("### 📷 Capture from webcam")
image_input = st.camera_input("Take a picture")

st.markdown("Or 📁 upload an image instead:")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# ---------------------------
# Load and Display Image
# ---------------------------
image = None
if image_input:
    image = Image.open(image_input).convert("RGB")
elif uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

if image:
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)
    st.markdown("---")

    with st.spinner("🔍 Analyzing image..."):
        # Preprocess Image
        image = image.resize((224, 224))
        image_array = np.array(image).astype(np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        try:
            # Load TFLite Model
            interpreter = tf.lite.Interpreter(model_path="model/model.tflite")
            interpreter.allocate_tensors()

            input_index = interpreter.get_input_details()[0]['index']
            output_index = interpreter.get_output_details()[0]['index']

            # Run Inference
            interpreter.set_tensor(input_index, image_array)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_index)[0]

            # Get Results
            predicted_index = int(np.argmax(predictions))
            predicted_class = class_names[predicted_index]
            confidence = predictions[predicted_index]

            # Display Prediction
            st.markdown("### 🧪 Prediction Result")
            if predicted_class == "good":
                st.success(f"🟢 **{predicted_class.upper()}** ({confidence:.2%} confidence)")
            else:
                st.error(f"🔴 **{predicted_class.upper()}** detected! ({confidence:.2%} confidence)")

            # Show All Class Scores
            st.markdown("### 📊 Full Confidence Scores:")
            for i, prob in enumerate(predictions):
                bar_color = "green" if class_names[i] == "good" else "red"
                st.write(f"**{class_names[i].capitalize()}**: {prob:.2%}")
                st.progress(min(int(prob * 100), 100))

        except Exception as e:
            st.error(f"❌ Error loading model: {e}")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Made with ❤️ using TensorFlow Lite & Streamlit</p>", unsafe_allow_html=True)
