Door & Window Detection in Blueprints (YOLOv11 OBB)

This project detects doors and windows in architectural blueprint images using a custom-trained YOLOv8-OBB model (Oriented Bounding Box). It includes a FastAPI-based inference API deployed on Render.

🚀 Features

    🔍 Detects door and window symbols in construction blueprints

    📦 API accepts PNG/JPG images and returns bounding boxes + confidences

    🧠 Trained with manually labeled data using LabelImg

    📤 Deployed as a public API via Render

📁 Project Structure

visionai/
├── app.py                 # FastAPI app
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── images/                # Labeled blueprint images
├── labels/                # YOLO format label .txt files
├── classes.txt            # door, window


📦 Setup (Local)
1. Clone the repo

git clone https://github.com/Himanshukurrey/visionai.git
cd visionai

2. Create virtual environment (optional)

python -m venv venv
venv\Scripts\activate    # Windows

3. Install dependencies

pip install -r requirements.txt

4. Run the API

uvicorn app:app --reload


Visit:

http://127.0.0.1:8000/detect

Use the interactive /detect endpoint to upload an image and view results.
curl Example

curl -X POST http://127.0.0.1:8000/detect -F "file=@test.jpg"

Response format:

{
  "detections": [
    {"label": "door", "confidence": 0.91, "bbox": [34, 55, 78, 102]},
    {"label": "window", "confidence": 0.84, "bbox": [140, 88, 60, 90]}
  ]
}
