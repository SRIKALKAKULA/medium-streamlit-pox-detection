import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
from sidebar import show_sidebar

# Load the pre-trained YOLO model
model = YOLO('weights\yolov8_monkeypox.pt')

# Streamlit App Title
st.title("Monkey Pox Disease Detection")
st.write("Pox (Affected Or Not): To detect chances of MonkeyPox, Measles, and ChickenPox.")
st.write("Test whether an area is affected by pox.")

# Upload image
st.write("Upload Images")
uploaded_file = st.file_uploader("Drag and drop file here", type=["png", "jpg", "jpeg", "jfif"])

# Check if a file is uploaded
if uploaded_file is not None:
    # Convert the file to an image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the image to a format that YOLO model can process
    image = np.array(image)

    # Perform predictions
    results = model(image)
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()

    # Get the highest probability class
    detected_class = names_dict[np.argmax(probs)]

    # Display the results
    st.subheader("Results")
    
    if detected_class == 'Monkeypox':
        st.write(f"**Pox type detected:** Monkey Pox")
    else:
        st.write(f"**Not a case of Monkey Pox**")
    
    # Display pox types arranged by probability
    st.write("Pox types arranged in order of probability (highest first):")
    for idx, (pox_type, prob) in enumerate(sorted(zip(names_dict.values(), probs), key=lambda x: x[1], reverse=True)):
        st.write(f"{idx+1}. {pox_type} ({prob*100:.2f}%)")
