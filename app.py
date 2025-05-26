from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io

app = FastAPI()


model = YOLO(r"E:\AI_Vision\obb\train\weights\best.pt")  


CLASS_NAMES = ["Door", "Window"]

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
   
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_array = np.array(image)

   
    results = model.predict(image_array, conf=0.25, imgsz=640)[0]

    detections = []

   
    if hasattr(results, "obb"):
        boxes = results.obb.xyxy.numpy() 
        confs = results.obb.conf.numpy()
        clss = results.obb.cls.numpy().astype(int)
    else:
        boxes = results.boxes.xyxy.numpy()
        confs = results.boxes.conf.numpy()
        clss = results.boxes.cls.numpy().astype(int)

    
    for box, conf, cls_id in zip(boxes, confs, clss):
        x1, y1, x2, y2 = box
        bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
        detections.append({
            "label": CLASS_NAMES[cls_id],
            "confidence": round(float(conf), 2),
            "bbox": bbox
        })

    return JSONResponse(content={"detections": detections})
