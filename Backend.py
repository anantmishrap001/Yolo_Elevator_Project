from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import csv
import os
from datetime import datetime

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8n.pt")

# CSV log setup
LOG_FILE = "people_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "People_Count", "Door_Status"])

def log_data(people_count, door_status):
    with open(LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), people_count, door_status])

camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            results = model(frame)
            person_count = 0

            # Only detect class 'person' (class ID 0 in COCO dataset)
            for box in results[0].boxes:
                cls = int(box.cls[0])
                if cls == 0:  # Person class only
                    person_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Person", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Door and LED logic
            if person_count < 4:
                door_status = "Closed"
                led_color = "red"
            else:
                door_status = "Open"
                led_color = "green"

            # Log the data
            log_data(person_count, door_status)

            # Add overlay info
            cv2.putText(frame, f"People: {person_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
            cv2.putText(frame, f"Door: {door_status}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/get_data')
def get_data():
    data = []
    with open(LOG_FILE, mode='r') as file:
        reader = csv.DictReader(file)
        data = list(reader)
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
