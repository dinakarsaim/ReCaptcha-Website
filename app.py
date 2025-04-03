from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)

# Load the trained YOLOv8 model
model = YOLO("best.pt")  # Replace with your trained model weights

# Ensure uploads folder exists
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded!"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file!"})

    # Save the uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    # Run inference
    results = model(image_path)
    
    # Draw bounding boxes
    result_image_path = os.path.join(RESULT_FOLDER, "result_" + file.filename)
    for result in results:
        im_array = result.plot()  # Get image with bounding boxes
        cv2.imwrite(result_image_path, im_array)

    return jsonify({"image_path": result_image_path})

if __name__ == "__main__":
    app.run(debug=True)