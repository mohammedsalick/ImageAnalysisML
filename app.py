import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import time

# Install dependencies
os.system("pip install opencv-python-headless")

# Streamlit Page Config
st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide", page_icon="🔍")

st.title("🔍 YOLOv8 Object Detection & Model Comparison")
st.write("Upload an **image** or use the **webcam** to detect objects!")

# Sidebar Settings
st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
iou_threshold = st.sidebar.slider("IOU Threshold", 0.1, 1.0, 0.5, 0.05)

# Model Selection
models_dict = {
    "YOLOv8n (Nano)": "yolov8n.pt",
    "YOLOv8s (Small)": "yolov8s.pt",
    "YOLOv8m (Medium)": "yolov8m.pt",
    "YOLOv8l (Large)": "yolov8l.pt",
}

mode = st.sidebar.radio("Select Mode", ["Single Model", "Compare Models"])
if mode == "Single Model":
    selected_model = st.sidebar.selectbox("Choose a YOLOv8 Model", list(models_dict.keys()))
else:
    selected_models = st.sidebar.multiselect("Select Models for Comparison", list(models_dict.keys()), default=["YOLOv8n (Nano)", "YOLOv8s (Small)"])

# Choose Input Type
input_mode = st.sidebar.radio("Select Input Type", ["Image", "Webcam"])

# Load Selected Models
def load_model(model_name):
    return YOLO(models_dict[model_name])

# Process Image with YOLO
def process_image(image, model):
    start_time = time.time()
    image_np = np.array(image)
    results = model(image_np, conf=confidence_threshold, iou=iou_threshold)
    total_time = time.time() - start_time

    # Draw bounding boxes
    detected_classes = {}
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = f"{model.names[class_id]}: {confidence:.2f}"

            # Draw bounding box
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Track class count
            class_name = model.names[class_id]
            detected_classes[class_name] = detected_classes.get(class_name, 0) + 1

    return image_np, detected_classes, total_time

# Upload or Capture Image
uploaded_image = None
if input_mode == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
elif input_mode == "Webcam":
    uploaded_image = st.camera_input("Take a picture")

if uploaded_image:
    image = Image.open(uploaded_image)

    # Single Model Mode
    if mode == "Single Model":
        model = load_model(selected_model)
        processed_image, detected_classes, time_taken = process_image(image, model)
        
        # Display Image
        st.image(processed_image, caption=f"Processed with {selected_model}", use_column_width=True)
        
        # Accuracy Report
        total_objects = sum(detected_classes.values())
        avg_confidence = round((sum(detected_classes.values()) / total_objects) * 100, 2) if total_objects else 0
        st.write(f"### 📊 Accuracy Report for {selected_model}")
        st.write(f"🔹 **Total Objects Detected:** {total_objects}")
        st.write(f"🔹 **Average Confidence:** {avg_confidence}%")
        st.write(f"🔹 **Processing Time:** {time_taken:.2f} sec")
        st.write("🔹 **Detected Classes:**")
        st.write(detected_classes)

        # Download Report
        report_text = f"""
        YOLO Model: {selected_model}
        Total Objects Detected: {total_objects}
        Average Confidence: {avg_confidence}%
        Processing Time: {time_taken:.2f} sec
        Detected Classes: {detected_classes}
        """
        st.sidebar.download_button("Download Report", data=report_text, file_name="detection_report.txt", mime="text/plain")

    # Compare Models Mode
    elif mode == "Compare Models":
        st.write("### 📈 Model Comparison")
        comparison_results = {}
        
        for model_name in selected_models:
            model = load_model(model_name)
            processed_image, detected_classes, time_taken = process_image(image, model)
            comparison_results[model_name] = {
                "Image": processed_image,
                "Detected Classes": detected_classes,
                "Total Objects": sum(detected_classes.values()),
                "Processing Time": time_taken,
            }

        # Display Comparison Results
        cols = st.columns(len(selected_models))
        for i, model_name in enumerate(selected_models):
            with cols[i]:
                st.write(f"#### {model_name}")
                st.image(comparison_results[model_name]["Image"], caption=model_name, use_column_width=True)
                st.write(f"📊 **Total Objects:** {comparison_results[model_name]['Total Objects']}")
                st.write(f"⏳ **Processing Time:** {comparison_results[model_name]['Processing Time']:.2f} sec")
                st.write(f"🔍 **Detected Classes:**")
                st.write(comparison_results[model_name]["Detected Classes"])

st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Developed by Mohammed Salick** | Powered by [YOLOv8](https://ultralytics.com/) & [Streamlit](https://streamlit.io/)")
