import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import time
import io
import requests
from io import BytesIO
import base64

# Page configuration
st.set_page_config(
    page_title="Object Detection App",
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
    .sidebar .stRadio > div {
        flex-direction: column;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.title("🔍 Object Detection App")
st.write("Upload an image to detect objects with customizable settings")

# Sidebar
with st.sidebar:
    st.title("🛠️ Settings")
    
    # Framework selection (major change: added YOLOv8 option)
    model_framework = st.radio(
        "Select Framework",
        ["Roboflow API", "YOLOv8 (Local)"]
    )
    
    # Conditional settings based on framework selection
    if model_framework == "Roboflow API":
        # Model selection
        model_type = st.radio(
            "Select Model",
            ["General Object Detection", "Face Detection"]
        )
        
        # API key input (only for Roboflow)
        if 'api_key' not in st.session_state:
            api_key = st.text_input("Enter your Roboflow API key", type="password", help="Get a free API key from Roboflow.com")
            if api_key:
                st.session_state.api_key = api_key
                st.success("API Key saved!")
    
    else:  # YOLOv8 settings
        model_size = st.radio(
            "YOLOv8 Model Size",
            ["nano", "small", "medium", "large", "extra-large"],
            index=1,  # Default to small
            help="Larger models are more accurate but slower"
        )
        
        # First run installation notice
        if 'yolo_installed' not in st.session_state:
            st.warning("First use of YOLOv8 will install dependencies. This may take a moment.")
    
    # Common settings for both frameworks
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
    
    # About section
    with st.expander("About"):
        st.write("""
        This app lets you detect objects in images using either:
        
        1. **Roboflow API** - Cloud-based detection (requires API key)
        2. **YOLOv8** - Runs locally in your browser (no API key needed)
        
        Upload an image to get started!
        """)

# Use columns for layout
col1, col2 = st.columns([1, 1])

# Initialize session state for history
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []

# Function to detect objects using Roboflow API
def detect_objects_roboflow(image, model_type, confidence):
    api_key = st.session_state.api_key
    
    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Choose model ID based on selection
    if model_type == "General Object Detection":
        model_id = "coco-128/1"  # General COCO model
    else:
        model_id = "face-detection/1"  # Face detection model
    
    # Call Roboflow API
    upload_url = f"https://detect.roboflow.com/{model_id}?api_key={api_key}&confidence={confidence}"
    
    try:
        response = requests.post(upload_url, 
                                data=img_byte_arr,
                                headers={
                                   "Content-Type": "application/x-www-form-urlencoded"
                                })
        
        # Check if request was successful
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None

# Function to detect objects using YOLOv8 (new function)
def detect_objects_yolo(image, model_size, confidence):
    # Install YOLOv8 if not already installed
    try:
        import ultralytics
    except ImportError:
        with st.spinner("Installing YOLOv8 dependencies (first run only)..."):
            import subprocess
            subprocess.run(["pip", "install", "ultralytics"], check=True)
            st.session_state.yolo_installed = True
            # Import after installation
            import ultralytics
    
    from ultralytics import YOLO
    
    # Load model - this will download the model on first run
    with st.spinner(f"Loading YOLOv8 {model_size} model..."):
        model = YOLO(f"yolov8{model_size[0]}.pt")  # Uses first letter of size (n, s, m, l, x)
    
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Run detection
    results = model(img_array, conf=confidence)
    
    # Convert YOLO results to Roboflow-like format for compatibility with existing code
    predictions_list = []
    
    # Extract result for the first image (we only send one)
    result = results[0]
    
    # Access the boxes, confidence scores, and class IDs
    for box, score, cls in zip(result.boxes.xyxy.tolist(), 
                              result.boxes.conf.tolist(),
                              result.boxes.cls.tolist()):
        x1, y1, x2, y2 = box
        cls_id = int(cls)
        class_name = model.names[cls_id]
        
        # Calculate center point and width/height (to match Roboflow format)
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        
        predictions_list.append({
            "x": x_center,
            "y": y_center,
            "width": width,
            "height": height,
            "class": class_name,
            "confidence": score
        })
    
    # Return in Roboflow-compatible format
    return {"predictions": predictions_list}

# Function to draw bounding boxes on the image
def draw_boxes(image, predictions, show_labels, show_confidence, box_color_hex):
    # Convert hex color to RGB
    box_color = tuple(int(box_color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    
    # Create a draw object
    draw = Image.open(image)
    draw_img = draw.copy()
    img_width, img_height = draw_img.size
    
    detected_objects = {}
    
    # Create a numpy array to draw on
    draw_np = np.array(draw_img)
    
    # Import cv2 only inside this function to avoid dependency issues
    try:
        import cv2
        for prediction in predictions['predictions']:
            # Get box coordinates
            x = prediction['x'] 
            y = prediction['y']
            width = prediction['width'] 
            height = prediction['height']
            
            # Calculate box coordinates
            x1 = int(x - width/2)
            y1 = int(y - height/2)
            x2 = int(x + width/2)
            y2 = int(y + height/2)
            
            # Make sure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)
            
            # Get class and confidence
            class_name = prediction['class']
            confidence = prediction['confidence']
            
            # Count objects
            if class_name in detected_objects:
                detected_objects[class_name] += 1
            else:
                detected_objects[class_name] = 1
            
            # Draw rectangle
            cv2.rectangle(draw_np, (x1, y1), (x2, y2), box_color, 2)
            
            # Add label if requested
            if show_labels:
                label = class_name
                if show_confidence:
                    label += f": {confidence:.2f}"
                
                # Draw label background
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    draw_np, 
                    (x1, y1 - int(label_height * 1.5)), 
                    (x1 + label_width, y1), 
                    box_color, 
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    draw_np, 
                    label, 
                    (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0, 0, 0), 
                    2
                )
        
        output_img = Image.fromarray(draw_np)
        return output_img, detected_objects
    except ImportError:
        # Fallback to PIL if cv2 is not available
        from PIL import ImageDraw, ImageFont
        
        draw_img = draw.copy()
        drawing = ImageDraw.Draw(draw_img)
        
        for prediction in predictions['predictions']:
            # Get box coordinates
            x = prediction['x'] 
            y = prediction['y']
            width = prediction['width'] 
            height = prediction['height']
            
            # Calculate box coordinates
            x1 = int(x - width/2)
            y1 = int(y - height/2)
            x2 = int(x + width/2)
            y2 = int(y + height/2)
            
            # Make sure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)
            
            # Get class and confidence
            class_name = prediction['class']
            confidence = prediction['confidence']
            
            # Count objects
            if class_name in detected_objects:
                detected_objects[class_name] += 1
            else:
                detected_objects[class_name] = 1
            
            # Draw rectangle
            drawing.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
            
            # Add label if requested
            if show_labels:
                label = class_name
                if show_confidence:
                    label += f": {confidence:.2f}"
                
                # Draw label background
                try:
                    font = ImageFont.truetype("arial.ttf", 14)
                except IOError:
                    font = ImageFont.load_default()
                
                text_width, text_height = drawing.textsize(label, font=font)
                drawing.rectangle(
                    [(x1, y1 - text_height - 4), (x1 + text_width, y1)],
                    fill=box_color
                )
                
                # Draw label text
                drawing.text(
                    (x1, y1 - text_height - 2),
                    label,
                    fill=(0, 0, 0),
                    font=font
                )
        
        return draw_img, detected_objects

# Main interface
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
        
        # Different button text based on selected framework
        button_text = "Detect Objects"
        if model_framework == "YOLOv8 (Local)":
            button_text = "Detect with YOLOv8"
        else:
            button_text = "Detect with Roboflow"
            
        detect_button = st.button(button_text, type="primary")
    
with col2:
    if input_image and detect_button:
        with st.spinner("Detecting objects..."):
            try:
                # Time the detection process
                start_time = time.time()
                
                # Different detection method based on selected framework
                if model_framework == "Roboflow API":
                    # Check if API key exists
                    if 'api_key' not in st.session_state:
                        st.error("Please enter your Roboflow API key in the sidebar first.")
                    else:
                        # Call Roboflow API for detection
                        predictions = detect_objects_roboflow(
                            input_image, 
                            model_type, 
                            confidence_threshold
                        )
                else:  # YOLOv8
                    # Call YOLOv8 for detection
                    predictions = detect_objects_yolo(
                        input_image,
                        model_size,
                        confidence_threshold
                    )
                
                detection_time = time.time() - start_time
                
                if predictions and 'predictions' in predictions:
                    # Draw boxes on image
                    output_img, detected_objects = draw_boxes(
                        input_image, 
                        predictions, 
                        show_labels, 
                        show_confidence, 
                        box_color
                    )
                    
                    # Get model name for history
                    if model_framework == "Roboflow API":
                        model_name = model_type
                    else:
                        model_name = f"YOLOv8-{model_size}"
                    
                    # Save to detection history
                    history_entry = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "model": model_name,
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
                        total_objects = len(predictions['predictions'])
                        st.metric("Objects Detected", total_objects)
                        st.metric("Detection Time", f"{detection_time:.3f} sec")
                    
                    with col_stats2:
                        st.metric("Unique Classes", len(detected_objects))
                        st.metric("Model Used", model_name)
                    
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
                            # Convert image to bytes for download
                            img_bytes = io.BytesIO()
                            output_img.save(img_bytes, format='PNG')
                            st.download_button(
                                label="Download Result Image",
                                data=img_bytes.getvalue(),
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
                else:
                    if model_framework == "Roboflow API":
                        st.error("No predictions returned from the API.")
                    else:
                        st.error("No objects detected by YOLOv8.")
            
            except Exception as e:
                st.error(f"Error during detection: {str(e)}")
                if model_framework == "Roboflow API":
                    st.write("Please check your API key and try again.")
                else:
                    st.write("There was an issue with YOLOv8. Try refreshing the page.")

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
    "💡 **Tip:** YOLOv8 runs locally in your browser - no API key needed. For more specialized models, try Roboflow API.",
    help="YOLOv8 is pre-trained on the COCO dataset with 80 common object classes."
)