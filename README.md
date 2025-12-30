# Autonomous Robot: Cloud-Processed Threat Detection

A high-performance backend system designed for autonomous robots (ESP32/Arduino) to perform real-time image recognition. The system leverages **YOLOv8** for object detection and uses **Telegram** for instant security notifications.

## 🚀 Features
- **REST API:** Flask-based endpoint for receiving image data from IoT devices.
- **Deep Learning Inference:** Real-time object detection using YOLOv8 (Ultralytics).
- **Intelligent Filtering:** Specifically configured to detect "sharp objects" (knives, scissors) and trigger alerts only when confidence exceeds 50%.
- **Instant Alerts:** Automated Telegram notifications with the captured image attached for immediate verification.

## 🏗️ Architecture
1. **Edge Device:** Captures image and sends a POST request to the cloud server.
2. **Flask Backend:** Receives the file, saves it locally, and passes it to the AI model.
3. **YOLOv8 Model:** Analyzes the image for specific classes.
4. **Notification Service:** If a threat is detected, the Telegram Bot API sends a photo alert to the user.

## 🛠️ Tech Stack
- **Language:** Python
- **Framework:** Flask
- **AI/ML:** YOLOv8 (Ultralytics), OpenCV, NumPy
- **API:** Telegram Bot API
- **Deployment:** (e.g., AWS EC2, Heroku, or Localhost)

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/robot-threat-detection.git](https://github.com/yourusername/robot-threat-detection.git)
   cd robot-threat-detection