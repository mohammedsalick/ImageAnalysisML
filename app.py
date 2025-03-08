import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import io

# Page configuration
st.set_page_config(
    page_title="Advanced Object Detection",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .detection-stats {
        background-color: #e6f3ff;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for history and model caching
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

if 'model' not in st.session_state:
    st.session_state.model = None

# Main title
st.title("🔍 Advanced YOLOv8 Object Detection")
st.write("Upload an image or use your camera to detect objects with advanced features")

# Sidebar for settings
with st.sidebar:
    st.title("🛠️ Settings")
    
    # Model selection
    model_size = st.radio(
        "YOLOv8 Model Size",
        ["nano", "small"],
        index=0,
        help="Nano is fastest, Small is more accurate but slower"
    )
    
    # Detection settings
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence score for detections"
    )
    
    # Display settings
    show_labels = st.checkbox("Show Labels", value=True)
    show_confidence = st.checkbox("Show Confidence Scores", value=True)
    
    # Box color
    box_color = st.color_picker("Bounding Box Color", "#00FF00")
    box_thickness = st.slider("Box Thickness", 1, 5, 2)
    
    # Advanced options
    with st.expander("Advanced Options"):
        use_dynamic_colors = st.checkbox("Color by Class", value=True, 
                               help="Use different colors for different object classes")
        
        max_detections = st.slider(
            "Max Detections", 
            min_value=1, 
            max_value=100, 
            value=50,
            help="Maximum number of detections to show"
        )
        
        nms_threshold = st.slider(
            "NMS Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.45, 
            step=0.05,
            help="Non-maximum suppression threshold"
        )
    
    # Class filtering
    with st.expander("Class Filtering"):
        filter_by_class = st.checkbox("Filter by Class", value=False)
        if filter_by_class:
            # We'll populate this once model is loaded
            if st.session_state.model is not None:
                class_names = list(st.session_state.model.names.values())
                selected_classes = st.multiselect(
                    "Select Classes to Detect",
                    options=class_names,
                    default=["person", "car", "dog", "cat"] if all(c in class_names for c in ["person", "car", "dog", "cat"]) else class_names[:5]
                )
            else:
                st.info("Class selection will be available once model is loaded")
                selected_classes = []
        else:
            selected_classes = []

# Function to load model
@st.cache_resource
def load_yolo_model(model_size):
    model_path = f"yolov8{model_size[0]}.pt"  # n for nano, s for small
    model = YOLO(model_path)
    return model

# Function to process image and run detection
def process_image(image, confidence, nms_iou, max_detections, selected_classes):
    # Convert PIL image to numpy array if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image
    
    # Make a copy for drawing
    output_image = image_np.copy()
    
    # Run YOLO detection
    results = st.session_state.model(
        image_np, 
        conf=confidence, 
        iou=nms_iou,
        max_det=max_detections,
        verbose=False
    )
    
    # Process results
    detected_objects = {}
    detections_data = []
    
    # Process each result (usually just one for a single image)
    for result in results:
        for i, box in enumerate(result.boxes):
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get confidence score
            confidence = float(box.conf[0])
            
            # Get class ID and name
            class_id = int(box.cls[0])
            class_name = st.session_state.model.names[class_id]
            
            # Skip if class filtering is active and class not selected
            if filter_by_class and selected_classes and class_name not in selected_classes:
                continue
            
            # Count objects by class
            if class_name in detected_objects:
                detected_objects[class_name] += 1
            else:
                detected_objects[class_name] = 1
            
            # Create label
            label = class_name
            if show_confidence:
                label += f": {confidence:.2f}"
            
            # Determine box color
            if use_dynamic_colors:
                # Generate a deterministic color based on class name
                color_hash = sum(ord(c) for c in class_name) % 255
                color = (color_hash, 255 - color_hash, 180)
            else:
                # Convert hex color to RGB
                color = tuple(int(box_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                # Reverse for BGR (OpenCV)
                color = color[::-1]
            
            # Draw bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, box_thickness)
            
            # Draw label if enabled
            if show_labels:
                # Calculate text size for background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Draw background for text
                cv2.rectangle(
                    output_image,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    output_image,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )
            
            # Store detection data for analysis
            detections_data.append({
                "id": i,
                "class": class_name,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "width": x2 - x1,
                "height": y2 - y1,
                "area": (x2 - x1) * (y2 - y1)
            })
    
    return output_image, detected_objects, detections_data

# Create two columns for layout
col1, col2 = st.columns([1, 1])

# Input column
with col1:
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "png", "jpeg"],
        help="Supported formats: JPG, PNG, JPEG"
    )
    
    # Camera input
    camera_input = st.camera_input("Or take a photo")
    
    # Initialize input source
    input_image = None
    source_text = ""
    
    # Process uploaded file
    if uploaded_file:
        input_image = Image.open(uploaded_file)
        source_text = "Uploaded Image"
    # Process camera input if no file uploaded
    elif camera_input:
        input_image = Image.open(camera_input)
        source_text = "Camera Image"
    
    # Display the input image
    if input_image:
        st.image(input_image, caption=source_text, use_column_width=True)
        
        # Run detection button
        detect_button = st.button("Detect Objects", type="primary")

# Output column
with col2:
    if input_image and detect_button:
        # Load model if not already loaded or if model size changed
        model_key = f"yolov8{model_size[0]}"
        if st.session_state.model is None or model_key not in str(st.session_state.model):
            with st.spinner(f"Loading YOLOv8 {model_size} model..."):
                st.session_state.model = load_yolo_model(model_size)
        
        # Run detection
        with st.spinner("Detecting objects..."):
            start_time = time.time()
            
            try:
                output_image, detected_objects, detections_data = process_image(
                    input_image,
                    confidence_threshold,
                    nms_threshold,
                    max_detections,
                    selected_classes
                )
                
                detection_time = time.time() - start_time
                
                # Save to detection history
                history_entry = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model": f"YOLOv8-{model_size}",
                    "objects": detected_objects,
                    "detection_time": detection_time,
                    "total_objects": sum(detected_objects.values())
                }
                st.session_state.detection_history.append(history_entry)
                
                # Display output image
                st.image(output_image, caption="Detection Results", use_column_width=True)
                
                # Detection statistics
                st.markdown('<div class="detection-stats">', unsafe_allow_html=True)
                st.subheader("📊 Detection Summary")
                
                stat_cols = st.columns(4)
                
                total_objects = sum(detected_objects.values())
                with stat_cols[0]:
                    st.metric("Objects Detected", total_objects)
                
                with stat_cols[1]:
                    st.metric("Detection Time", f"{detection_time:.3f} sec")
                
                with stat_cols[2]:
                    st.metric("Unique Classes", len(detected_objects))
                
                with stat_cols[3]:
                    st.metric("Model", f"YOLOv8-{model_size}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Create DataFrame for detections
                if detections_data:
                    df = pd.DataFrame(detections_data)
                    
                    # Show first few detections in a table
                    with st.expander("Detailed Detections", expanded=False):
                        st.dataframe(
                            df[["id", "class", "confidence", "width", "height", "area"]].sort_values(
                                by="confidence", ascending=False
                            ),
                            use_container_width=True
                        )
                
                # Class distribution chart
                if detected_objects:
                    st.subheader("Detected Object Classes")
                    
                    # Create DataFrame for plotting
                    plot_df = pd.DataFrame({
                        "Class": list(detected_objects.keys()),
                        "Count": list(detected_objects.values())
                    }).sort_values(by="Count", ascending=False)
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, 4))
                    bars = ax.bar(plot_df["Class"], plot_df["Count"], color='skyblue')
                    ax.set_xlabel("Object Class")
                    ax.set_ylabel("Count")
                    ax.set_title("Detected Object Classes")
                    plt.xticks(rotation=45, ha='right')
                    
                    # Add count labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                int(height), ha='center', va='bottom')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Download buttons
                download_cols = st.columns(2)
                
                with download_cols[0]:
                    # Convert numpy image to PIL then to bytes
                    pil_img = Image.fromarray(output_image)
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    
                    st.download_button(
                        "Download Result Image",
                        data=buf.getvalue(),
                        file_name=f"detection_result_{time.strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png"
                    )
                
                with download_cols[1]:
                    if detections_data:
                        # Create CSV for download
                        csv = pd.DataFrame(detected_objects.items(), columns=["Class", "Count"]).to_csv(index=False)
                        
                        st.download_button(
                            "Download Results as CSV",
                            data=csv,
                            file_name=f"detection_data_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            
            except Exception as e:
                st.error(f"Error during detection: {str(e)}")
                st.write("Please try again with a different image or check if the model was loaded correctly.")

# Detection History
if st.session_state.detection_history:
    st.markdown("---")
    with st.expander("📜 Detection History", expanded=False):
        for i, entry in enumerate(reversed(st.session_state.detection_history)):
            st.write(f"**Detection #{len(st.session_state.detection_history) - i}** - {entry['timestamp']}")
            st.write(f"Model: {entry['model']} | Time: {entry['detection_time']:.3f}s | Objects: {entry['total_objects']}")
            
            # Show objects as a horizontal pill format
            html_pills = ""
            for obj, count in entry['objects'].items():
                html_pills += f'<span style="background-color: #e0e0e0; padding: 3px 8px; border-radius: 12px; margin-right: 6px; font-size: 0.8em;">{obj} ({count})</span>'
            
            st.markdown(f"Objects: {html_pills}", unsafe_allow_html=True)
            st.divider()
        
        # Clear history button
        if st.button("Clear History"):
            st.session_state.detection_history = []
            st.experimental_rerun()

# Tips and Help Section
with st.expander("ℹ️ Tips & Help"):
    st.markdown("""
    ### Tips for Better Detection
    
    - **Image Quality:** Higher resolution images generally produce better results, but may be slower to process.
    - **Lighting:** Well-lit images tend to yield more accurate detections.
    - **Adjust Confidence:** Lower the confidence threshold to detect more objects (may include false positives).
    - **Model Selection:** Use YOLOv8n for speed, YOLOv8s for more accuracy.
    - **Class Filtering:** For busy scenes, filter by class to focus on objects of interest.
    
    ### Troubleshooting
    
    - If the app seems slow, try using the nano model instead of small.
    - If detection is missing objects, try lowering the confidence threshold.
    - For overlapping detections, adjust the NMS threshold.
    """)

# Footer
st.markdown("---")
st.markdown("💡 **YOLOv8 Object Detection** | Built with Streamlit & Ultralytics YOLOv8")