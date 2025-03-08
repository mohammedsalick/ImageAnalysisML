# Install required packages
!pip install ultralytics opencv-python-headless flask-ngrok flask pyngrok

# Import libraries
from flask import Flask, request, jsonify, render_template
from flask_ngrok import run_with_ngrok
import cv2
import numpy as np
from PIL import Image
import io
import base64
from ultralytics import YOLO
import os

# Create Flask app
app = Flask(__name__)
run_with_ngrok(app)  # Start ngrok when app is run

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # You can change to 'yolov8s.pt' for better accuracy

# Create templates directory
os.makedirs('templates', exist_ok=True)

# Create HTML template
with open('templates/index.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Object Detection with YOLOv8</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-color: #f8f9fa;
            padding-top: 20px;
        }
        .card {
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        #result-image {
            max-width: 100%;
            height: auto;
        }
        .detection-list {
            max-height: 300px;
            overflow-y: auto;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .spinner-border {
            width: 3rem;
            height: 3rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-4">🔍 Object Detection with YOLOv8</h1>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Upload Image</h5>
                    </div>
                    <div class="card-body">
                        <form id="upload-form">
                            <div class="mb-3">
                                <input class="form-control" type="file" id="formFile" accept="image/*">
                            </div>
                            <button type="submit" class="btn btn-primary">Detect Objects</button>
                        </form>
                        
                        <div class="loading">
                            <div class="spinner-border text-primary" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                            <p>Processing image...</p>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header">
                        <h5>Detected Objects</h5>
                    </div>
                    <div class="card-body">
                        <ul id="detections" class="list-group detection-list">
                            <li class="list-group-item text-center text-muted">No detections yet</li>
                        </ul>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Results</h5>
                    </div>
                    <div class="card-body text-center">
                        <img id="result-image" src="" class="d-none">
                        <p id="no-image" class="text-muted">Upload an image to see detection results</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('upload-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('formFile');
            if (!fileInput.files[0]) {
                alert('Please select an image first');
                return;
            }
            
            const formData = new FormData();
            formData.append('image', fileInput.files[0]);
            
            // Show loading indicator
            document.querySelector('.loading').style.display = 'block';
            
            try {
                const response = await fetch('/detect', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error('Server error');
                }
                
                const data = await response.json();
                
                // Update the image
                const resultImage = document.getElementById('result-image');
                resultImage.src = data.image;
                resultImage.classList.remove('d-none');
                document.getElementById('no-image').classList.add('d-none');
                
                // Update detections list
                const detectionsList = document.getElementById('detections');
                if (data.detections && data.detections.length > 0) {
                    detectionsList.innerHTML = '';
                    data.detections.forEach(det => {
                        const li = document.createElement('li');
                        li.className = 'list-group-item';
                        li.innerHTML = `<strong>${det.class}</strong>: ${(det.confidence * 100).toFixed(2)}% confidence`;
                        detectionsList.appendChild(li);
                    });
                } else {
                    detectionsList.innerHTML = '<li class="list-group-item text-center">No objects detected</li>';
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Error processing image. Please try again.');
            } finally {
                // Hide loading indicator
                document.querySelector('.loading').style.display = 'none';
            }
        });
    </script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
    ''')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    
    # Read image
    img_bytes = file.read()
    
    # Convert to PIL Image
    image = Image.open(io.BytesIO(img_bytes))
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Run detection
    results = model(img_array)
    
    # Process results
    detections = []
    for result in results:
        result_img = img_array.copy()
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            confidence = float(box.conf[0])  # Confidence
            class_id = int(box.cls[0])  # Class ID
            label = f"{model.names[class_id]}: {confidence:.2f}"
            
            # Draw bounding box
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(result_img, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add to detections list
            detections.append({
                'class': model.names[class_id],
                'confidence': float(confidence),
                'bbox': [int(x1), int(y1), int(x2), int(y2)]
            })
    
    # Convert processed image to base64 for display
    if len(img_array.shape) == 2:  # Grayscale
        result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        result_img = cv2.cvtColor(result_img, cv2.COLOR_RGBA2RGB)
        
    result_pil = Image.fromarray(result_img)
    buffered = io.BytesIO()
    result_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return jsonify({
        'image': f'data:image/jpeg;base64,{img_str}',
        'detections': detections
    })

# Start the Flask app
app.run()