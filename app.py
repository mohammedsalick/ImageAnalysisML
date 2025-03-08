import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import time
import io
import base64

# Install required packages at runtime
import subprocess
import sys

# Page configuration
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    page_icon="🔍",
    layout="wide"
)

# Function to install packages
def install_packages():
    with st.spinner("Setting up dependencies (this might take a minute)..."):
        # Install packages
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless", "--quiet"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "--quiet"])
        
        # Now import the packages after installation
        global cv2, YOLO
        import cv2
        from ultralytics import YOLO
        
        st.success("Setup complete!")
        return True
    
# Try importing, install if not available
try:
    import cv2
    from ultralytics import YOLO
    packages_installed = True
except ImportError:
    packages_installed = False

# If packages are not installed, install them
if not packages_installed:
    packages_installed = install_packages()

# Only proceed if packages are installed
if packages_installed:
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
        .sidebar .stRadio > div {
            flex-direction: column;
        }
    </style>
    """, unsafe_allow_html=True)

    # Main title
    st.title("🔍 Advanced YOLOv8 Object Detection")
    st.write("Upload an image to detect objects with customizable settings")

    # Sidebar
    with st.sidebar:
        st.title("🛠️ Settings")
        
        # Model selection
        model_type = st.radio(
            "Select YOLO Model",
            ["YOLOv8n (Fast)", "YOLOv8s (Balanced)"]
        )
        
        model_mapping = {
            "YOLOv8n (Fast)": "yolov8n.pt",
            "YOLOv8s (Balanced)": "yolov8s.pt"
        }
        
        selected_model = model_mapping[model_type]
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        # Detection options
        show_labels = st.checkbox("Show Labels", value=True)
        show_confidence = st.checkbox("Show Confidence", value=True)
        
        # Color options
        box_color = st.color_picker("Box Color", "#00FF00")
        # Convert hex to RGB
        box_color_rgb = tuple(int(box_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            box_thickness = st.slider("Box Thickness", 1, 10, 2)
            label_size = st.slider("Label Size", 0.3, 2.0, 0.6, 0.1)
        
        # About section
        with st.expander("About"):
            st.write("""
            This app uses YOLOv8 for real-time object detection. 
            Upload an image to detect objects from 80 different classes.
            
            Built with Streamlit and Ultralytics YOLOv8.
            """)

    # Use columns for layout
    col1, col2 = st.columns([1, 1])

    # Initialize session state for history
    if 'detection_history' not in st.session_state:
        st.session_state.detection_history = []

    with col1:
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload an image", 
            type=["jpg", "png", "jpeg"],
            help="Supported formats: JPG, PNG, JPEG"
        )
        
        # Camera input option
        camera_input = st.camera_input("Or take a photo")
        
        input_image = None
        source_text = ""
        
        if uploaded_file:
            input_image = Image.open(uploaded_file)
            source_text = "Uploaded Image"
        elif camera_input:
            input_image = Image.open(camera_input)
            source_text = "Camera Image"
        
        # Action buttons
        if input_image:
            st.image(input_image, caption=source_text, use_column_width=True)
            
            detect_button = st.button("Detect Objects", type="primary")
        
    with col2:
        if input_image and detect_button:
            with st.spinner("Detecting objects..."):
                try:
                    # Convert image for processing
                    image_np = np.array(input_image)
                    
                    # Download and load model
                    with st.spinner(f"Loading {model_type} model (first run may take longer)..."):
                        model = YOLO(selected_model)
                    
                    # Time the detection process
                    start_time = time.time()
                    results = model(image_np, conf=confidence_threshold)
                    detection_time = time.time() - start_time
                    
                    # Process results
                    output_img = image_np.copy()
                    detected_objects = {}
                    
                    # Process all result objects
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            confidence = float(box.conf[0])
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            
                            # Count objects
                            if class_name in detected_objects:
                                detected_objects[class_name] += 1
                            else:
                                detected_objects[class_name] = 1
                            
                            # Construct label
                            label = ""
                            if show_labels:
                                label = f"{class_name}"
                                if show_confidence:
                                    label += f": {confidence:.2f}"
                            
                            # Draw bounding box
                            cv2.rectangle(output_img, (x1, y1), (x2, y2), box_color_rgb, box_thickness)
                            
                            # Draw label background and text
                            if label:
                                (label_width, label_height), _ = cv2.getTextSize(
                                    label, cv2.FONT_HERSHEY_SIMPLEX, label_size, 2
                                )
                                cv2.rectangle(
                                    output_img, 
                                    (x1, y1 - int(label_height * 1.5)), 
                                    (x1 + label_width, y1), 
                                    box_color_rgb, 
                                    -1
                                )
                                cv2.putText(
                                    output_img, 
                                    label, 
                                    (x1, y1 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    label_size, 
                                    (0, 0, 0), 
                                    2
                                )
                    
                    # Save to detection history
                    history_entry = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "model": model_type,
                        "objects": detected_objects,
                        "detection_time": detection_time
                    }
                    st.session_state.detection_history.append(history_entry)
                    
                    # Display output
                    st.image(output_img, caption="Detection Results", use_column_width=True)
                    
                    # Detection statistics
                    st.markdown('<div class="detection-stats">', unsafe_allow_html=True)
                    st.subheader("📊 Detection Summary")
                    
                    col_stats1, col_stats2 = st.columns(2)
                    
                    with col_stats1:
                        total_objects = sum(detected_objects.values())
                        st.metric("Objects Detected", total_objects)
                        st.metric("Detection Time", f"{detection_time:.3f} sec")
                    
                    with col_stats2:
                        st.metric("Unique Classes", len(detected_objects))
                        st.metric("Model Used", model_type)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Objects table
                    if detected_objects:
                        st.subheader("Detected Objects")
                        df = pd.DataFrame({
                            "Class": list(detected_objects.keys()),
                            "Count": list(detected_objects.values())
                        }).sort_values(by="Count", ascending=False)
                        
                        # Plot objects bar chart
                        fig, ax = plt.subplots(figsize=(10, 4))
                        bars = ax.bar(df["Class"], df["Count"], color='skyblue')
                        ax.set_xlabel("Object Class")
                        ax.set_ylabel("Count")
                        ax.set_title("Detected Object Classes")
                        plt.xticks(rotation=45, ha='right')
                        
                        # Add count values on top of the bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                    int(height), ha='center', va='bottom')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Download buttons
                        download_col1, download_col2 = st.columns(2)
                        
                        with download_col1:
                            # Create PIL image for download
                            img_pil = Image.fromarray(output_img)
                            buffer = io.BytesIO()
                            img_pil.save(buffer, format="PNG")
                            img_bytes = buffer.getvalue()
                            
                            st.download_button(
                                label="Download Result Image",
                                data=img_bytes,
                                file_name=f"detection_result_{time.strftime('%Y%m%d_%H%M%S')}.png",
                                mime="image/png"
                            )
                        
                        with download_col2:
                            # Create CSV for download
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name=f"detection_data_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                    
                except Exception as e:
                    st.error(f"Error during detection: {str(e)}")
                    st.write("Detailed error information:")
                    st.code(str(e))

    # Detection history tab
    if st.session_state.detection_history:
        with st.expander("Detection History", expanded=False):
            for i, entry in enumerate(reversed(st.session_state.detection_history)):
                st.write(f"**Detection #{len(st.session_state.detection_history) - i}** - {entry['timestamp']}")
                st.write(f"Model: {entry['model']} | Time: {entry['detection_time']:.3f}s")
                
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

    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tip:** Try different models and confidence thresholds to optimize detection quality.",
        help="YOLOv8n is fastest but less accurate. YOLOv8s is more accurate but slower."
    )
else:
    st.error("Unable to set up required dependencies. Please try again or contact the administrator.")