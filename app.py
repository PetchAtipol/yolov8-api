import sys
from dotenv import load_dotenv
import os
import urllib.parse
import requests
import firebase_admin
from firebase_admin import credentials, storage
from fastapi import FastAPI
from io import BytesIO
from datetime import timedelta
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
from ultralytics import YOLO

app = FastAPI()
load_dotenv()

# ✅ Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ✅ Firebase Configuration
FIREBASE_CREDENTIALS = os.getenv('FIREBASE_CREDENTIALS')
cred = credentials.Certificate(FIREBASE_CREDENTIALS)
FIREBASE_BUCKET_NAME = os.getenv('FIREBASE_BUCKET_NAME')

if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_CREDENTIALS)
    firebase_admin.initialize_app(cred, {"storageBucket": FIREBASE_BUCKET_NAME})

# ✅ Load YOLOv8 Model
MODEL_PATH = "models/best50epoch.pt"
model = YOLO(MODEL_PATH)  # Automatically uses CPU or CUDA if available
model.to('cpu')

# ✅ Load class names from training config
import yaml
with open("models/data.yaml", "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
class_names = data["names"]



# ✅ Helper: Get latest image URL from Firebase
def get_latest_image_url():
    try:
        bucket = storage.bucket()
        blobs = list(bucket.list_blobs(prefix="ingredients/"))

        if not blobs:
            return None, "No images found in Firebase Storage."

        latest_blob = max(blobs, key=lambda blob: blob.time_created)

        latest_url = latest_blob.generate_signed_url(
            version="v4",
            expiration=timedelta(hours=1),
            method="GET"
        )
        return latest_url, None

    except Exception as e:
        return None, f"Error generating signed URL: {e}"

@app.get("/")
def root():
    return {"message": "✅ Chefsense API is running!"}

# ✅ FastAPI Route for Detection
@app.get("/detect/latest")
async def detect_latest():
    try:
        latest_url, error = get_latest_image_url()
        if error:
            return {"error": error}

        # Download image from URL
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(latest_url, headers=headers)

        if response.status_code != 200:
            return {"error": f"Failed to download image, HTTP {response.status_code}"}

        try:
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            return {"error": f"Invalid image format: {e}"}

        # Convert PIL image to numpy
        img_array = np.array(image)

        # Run YOLOv8 detection
        results = model(img_array, conf=0.25)

        # Parse results
        detected_items = []
        for result in results:
            boxes = result.boxes
            for cls_id, conf in zip(boxes.cls, boxes.conf):
                label_idx = int(cls_id.item())
                conf_score = float(conf.item())
                if label_idx < len(class_names):
                    name = class_names[label_idx]
                    detected_items.append(f"{name} ({conf_score:.2f})")

        # Format response
        if detected_items:
            detected_text = ", ".join(detected_items)
            response_text = f"ช่วยคิดเมนูที่สามารถทำได้ด้วยวัตถุดิบเหล่านี้หน่อย  {detected_text}"
        else:
            response_text = "ไม่ตรวจพบวัตถุดิบในภาพ ช่วยคิดเมนูด้วยวัตถุดิบง่ายขึ้นมาหน่อย"

        return {
            "latest_image_url": latest_url,
            "detections": response_text
        }

    except Exception as e:
        return {"error": str(e)}
