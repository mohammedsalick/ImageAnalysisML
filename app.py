import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' for better accuracy

st.title("🔍 YOLOv8 Object Detection")
st.write("Upload an image to detect objects!")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Convert uploaded file to OpenCV format
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Run YOLO detection
    results = model(image)

    # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = f"{model.names[class_id]}: {confidence:.2f}"

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display output
    st.image(image, caption="Detected Objects", use_column_width=True)
