import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO
import requests
from datetime import datetime

# Configuration
# ==============================
UPLOAD_FOLDER = "uploads"
MODEL_PATH = "models/yolov8n.pt"

BOT_TOKEN = "7771739775:AAGKOHZrXpewsKXWPP2XO38KIQfiMr12-bA"
CHAT_ID = 5157363214


CONFIDENCE_THRESHOLD = 0.5  # adjust as needed


# Setup
# ==============================
app = Flask(__name__)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("[INFO] Loading YOLO model...")
model = YOLO(MODEL_PATH)


# Telegram Send Function
# ==============================
def send_telegram_alert(image_path, detections):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    message = f"⚠️ Sharp object detected!\nObjects: {', '.join(detections)}"
    with open(image_path, "rb") as img:
        requests.post(url, data={"chat_id": CHAT_ID, "caption": message}, files={"photo": img})


# Flask Route
# ==============================
@app.route("/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = os.path.join(UPLOAD_FOLDER, f"{timestamp}.jpg")
    file.save(img_path)

    # Run YOLO detection
    results = model(img_path)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            if conf > CONFIDENCE_THRESHOLD:
                detections.append(label)

    # Check for "knife" or similar
    detected_objects = [obj for obj in detections if "knife" in obj.lower() or "scissors" in obj.lower()]

    if detected_objects:
        send_telegram_alert(img_path, detected_objects)
        return jsonify({"status": "alert", "detections": detected_objects}), 200
    else:
        return jsonify({"status": "safe", "detections": detections}), 200
    # if detected_objects:
    #     print(f"[ALERT] Sharp object detected: {detected_objects} in {img_path}")
    #     return jsonify({"status": "alert", "detections": detected_objects}), 200
    # else:
    #     print(f"[OK] No sharp object. Detections: {detections}")
    #     return jsonify({"status": "safe", "detections": detections}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
