from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import cv2
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO("best.pt") 

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    results = {}
    selected_object = request.form.get("selected_object", "all")  

    for i in range(1, 5):
        file_key = f"file{i}"
        image_path = None
        if file_key in request.files:
            file = request.files[file_key]
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)
        
            yolo_results = model(image_path)

            for result in yolo_results:
                filtered_results = []                

                for box in result.boxes:
                    class_id = int(box.cls[0])  
                    class_name = result.names[class_id]  
                    
                    if selected_object == "all" or class_name == selected_object:
                        filtered_results.append(box)

                result.boxes = filtered_results  
                im_array = result.plot()  
                
                result_image_path = os.path.join(RESULT_FOLDER, f"result_{file.filename}")
                cv2.imwrite(result_image_path, im_array)

                results[f"image_path{i}"] = f"/results/result_{file.filename}"

    return jsonify(results)

@app.route("/results/<filename>")
def get_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True) 
