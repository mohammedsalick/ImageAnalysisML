import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    page_icon="🔍",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 20px;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
    }
    .result-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for app configuration
with st.sidebar:
    st.markdown("## Model Configuration")
    
    model_type = st.selectbox(
        "Select YOLO Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        index=0,
        help="Larger models are more accurate but slower"
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence score for detection"
    )
    
    st.markdown("## Visual Settings")
    
    box_color = st.color_picker(
        "Bounding Box Color",
        value="#00FF00",
        help="Color of detection boxes"
    )
    
    box_thickness = st.slider(
        "Box Thickness",
        min_value=1,
        max_value=5,
        value=2
    )
    
    show_labels = st.checkbox("Show Labels", value=True)
    
    st.markdown("## App Settings")
    
    enable_history = st.checkbox(
        "Save Detection History",
        value=True,
        help="Save detection results for comparison"
    )

# Initialize session state for history
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

if 'model' not in st.session_state or st.session_state.model_type != model_type:
    with st.spinner(f"Loading {model_type} model..."):
        # Load YOLOv8 model
        try:
            st.session_state.model = YOLO(model_type)
            st.session_state.model_type = model_type
            st.sidebar.success(f"Model {model_type} loaded successfully!", icon="✅")
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {e}", icon="❌")

# Main content
st.markdown("<h1 class='main-header'>🔍 Advanced YOLOv8 Object Detection</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Upload multiple images to detect objects with customizable settings</p>", unsafe_allow_html=True)

# Function to process images
def process_image(image, file_name=""):
    start_time = time.time()
    
    # Run YOLO detection
    results = st.session_state.model(image, conf=confidence_threshold)
    
    # Process results
    output_image = image.copy()
    detections = []
    
    # Get RGB values from hex color
    hex_color = box_color.lstrip('#')
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    # Convert to BGR for OpenCV
    bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = st.session_state.model.names[class_id]
            
            # Draw bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), bgr_color, box_thickness)
            
            # Add label if enabled
            if show_labels:
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(
                    output_image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    bgr_color,
                    2
                )
            
            # Store detection info
            detections.append({
                "class": class_name,
                "confidence": round(confidence, 2),
                "bbox": [x1, y1, x2, y2]
            })
    
    processing_time = time.time() - start_time
    
    # Store in history if enabled
    if enable_history:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.detection_history.append({
            "timestamp": timestamp,
            "filename": file_name,
            "detections": detections,
            "processing_time": processing_time,
            "model": model_type
        })
    
    return output_image, detections, processing_time

# Tab system for different functionalities
tab1, tab2, tab3, tab4 = st.tabs([
    "📸 Image Upload", 
    "📊 Detection Analytics", 
    "📜 History",
    "ℹ️ About"
])

# Tab 1: Image Upload
with tab1:
    # Multi-file uploader
    uploaded_files = st.file_uploader(
        "Upload one or more images",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Process each uploaded image
        for uploaded_file in uploaded_files:
            col1, col2 = st.columns(2)
            
            with col1:
                # Original image
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                st.image(image_np, caption=f"Original: {uploaded_file.name}", use_column_width=True)
                
            with col2:
                # Process image
                output_image, detections, processing_time = process_image(image_np, uploaded_file.name)
                st.image(output_image, caption=f"Detected Objects: {uploaded_file.name}", use_column_width=True)
                st.write(f"Processing time: {processing_time:.2f} seconds")
            
            # Show detection results
            if detections:
                st.markdown(f"### Detections in {uploaded_file.name}")
                
                # Convert to DataFrame for better display
                df = pd.DataFrame(detections)
                st.dataframe(df, use_container_width=True)
                
                # Generate detection statistics
                if len(detections) > 0:
                    class_counts = pd.DataFrame(df['class'].value_counts()).reset_index()
                    class_counts.columns = ['Object', 'Count']
                    
                    # Plot the counts
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.bar(class_counts['Object'], class_counts['Count'], color='skyblue')
                    ax.set_ylabel('Count')
                    ax.set_title('Objects Detected')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Download options
                output_image_pil = Image.fromarray(output_image)
                buf = io.BytesIO()
                output_image_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                filename = os.path.splitext(uploaded_file.name)[0]
                st.download_button(
                    label="Download Processed Image",
                    data=byte_im,
                    file_name=f"{filename}_detected.png",
                    mime="image/png"
                )
                
                # Export detection data
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Export Detections as CSV",
                    data=csv,
                    file_name=f"{filename}_detections.csv",
                    mime="text/csv"
                )
            else:
                st.info("No objects detected in this image.")
            
            st.markdown("---")
    else:
        st.info("Please upload one or more images to begin detection.")
        
        # Sample images option
        if st.button("Try Sample Images"):
            # Here you would include some sample images that come with the app
            st.info("Sample images feature would load pre-packaged images for demonstration.")

# Tab 2: Detection Analytics
with tab2:
    st.markdown("### Object Detection Analytics")
    
    if st.session_state.detection_history:
        # Aggregate all detections
        all_objects = []
        for entry in st.session_state.detection_history:
            for detection in entry["detections"]:
                all_objects.append({
                    "class": detection["class"],
                    "confidence": detection["confidence"],
                    "image": entry["filename"]
                })
        
        if all_objects:
            df_all = pd.DataFrame(all_objects)
            
            # Overall statistics
            st.subheader("Session Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Images Processed", len(st.session_state.detection_history))
            with col2:
                st.metric("Objects Detected", len(all_objects))
            with col3:
                st.metric("Unique Object Types", len(df_all['class'].unique()))
            
            # Class distribution
            st.subheader("Object Class Distribution")
            class_counts = df_all['class'].value_counts().reset_index()
            class_counts.columns = ['Object', 'Count']
            
            # Bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(class_counts['Object'], class_counts['Count'], color='skyblue')
            ax.set_ylabel('Count')
            ax.set_title('Objects Detected Across All Images')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Confidence distribution
            st.subheader("Confidence Score Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df_all['confidence'], bins=10, alpha=0.7, color='green')
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Confidence Score Distribution')
            st.pyplot(fig)
            
            # Export option
            csv = df_all.to_csv(index=False)
            st.download_button(
                label="Export All Detection Data as CSV",
                data=csv,
                file_name="all_detections.csv",
                mime="text/csv"
            )
        else:
            st.info("No detection data available for analysis.")
    else:
        st.info("Process some images to view analytics.")

# Tab 3: History
with tab3:
    st.markdown("### Detection History")
    
    if st.session_state.detection_history:
        # Show history in reverse (newest first)
        for i, entry in enumerate(reversed(st.session_state.detection_history)):
            with st.expander(f"{entry['filename']} - {entry['timestamp']}"):
                st.write(f"Model: {entry['model']}")
                st.write(f"Processing time: {entry['processing_time']:.2f} seconds")
                
                if entry['detections']:
                    df = pd.DataFrame(entry['detections'])
                    st.dataframe(df)
                else:
                    st.write("No objects detected in this image.")
        
        if st.button("Clear History"):
            st.session_state.detection_history = []
            st.experimental_rerun()
    else:
        st.info("No detection history available.")

# Tab 4: About
with tab4:
    st.markdown("### About YOLOv8 Object Detection App")
    st.write("""
    This application uses YOLOv8 (You Only Look Once) to detect objects in images.
    
    **Features:**
    - Upload and process multiple images
    - Customizable detection settings
    - Visual analytics of detection results
    - Save and review detection history
    - Export detection data and processed images
    
    **Model Information:**
    - YOLOv8 is a state-of-the-art object detection model
    - Capable of detecting 80 different object classes
    - Various model sizes available (from Nano to XLarge)
    
    **How to use:**
    1. Upload one or more images
    2. Adjust detection settings in the sidebar if needed
    3. View detection results and analytics
    4. Download processed images or detection data
    """)
    
    st.markdown("---")
    st.write("Developed with Streamlit and YOLOv8")