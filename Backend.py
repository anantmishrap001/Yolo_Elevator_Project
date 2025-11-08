from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import time
import os

app = Flask(__name__)

# --- CONFIGURATION ---
MAX_PEOPLE_CHANGE_PER_FRAME = 2
model_path = YOLO('yolov8n.pt')

if not os.path.exists(model_path):
    raise FileNotFoundError("⚠️ Model file not found! Please place 'yolov8n.pt' in this directory.")

model = YOLO(model_path)
camera = cv2.VideoCapture(0)

# --- STATE VARIABLES ---
last_people_count = 0
security_alert_active_until = 0
door_status = "Door Closed"
people_count = 0

def generate_frames():
    global last_people_count, security_alert_active_until, door_status, people_count

    while True:
        success, frame = camera.read()
        if not success:
            break

        results = model(frame, classes=[0], verbose=False)
        annotated_frame = results[0].plot()
        people_count = len(results[0].boxes)

        change = abs(people_count - last_people_count)
        if change > MAX_PEOPLE_CHANGE_PER_FRAME:
            security_alert_active_until = time.time() + 5

        last_people_count = people_count

        # --- Door Logic ---
        if people_count >= 4:
            door_status = "Door Open"
        else:
            door_status = "Door Closed"

        # --- Display Overlay ---
        if time.time() < security_alert_active_until:
            cv2.putText(annotated_frame, "SECURITY ALERT!", (50, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            text_info = f"People: {people_count} | {door_status}"
            cv2.putText(annotated_frame, text_info, (50, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_data = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def get_status():
    """Live door and people info for frontend"""
    return jsonify({
        "door_status": door_status,
        "people_count": people_count
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
