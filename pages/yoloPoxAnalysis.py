import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

def app():
    st.subheader("Monkeypox Detection using YOLO Model")
    model_path = 'models/last.pt'
    model = YOLO(model_path)  # Load the YOLO model

    image_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    if image_file is not None:
        st.image(Image.open(image_file), caption="Uploaded Image", use_column_width=True)

        # Save the image temporarily
        with open("temp_image.jpg", "wb") as f:
            f.write(image_file.getbuffer())

        # Run the YOLO model
        results = model("temp_image.jpg")
        names_dict = results[0].names  # Class labels
        probs = results[0].probs.data.tolist()  # Probabilities

        if probs:
            st.subheader("Detection Results")
            for idx, prob in enumerate(probs):
                st.text(f"{names_dict[idx]}: {prob:.2f}")

            max_class = names_dict[np.argmax(probs)]
            st.success(f"Predicted Class: {max_class}")
        else:
            st.warning("No detections made.")
