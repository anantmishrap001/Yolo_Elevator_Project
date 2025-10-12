from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import hashlib
import os
import time # <--- IMPORT THE TIME LIBRARY

# --- 1. SECURITY CONFIGURATION ---
KNOWN_MODEL_HASH = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"  # SHA256 hash of the known good model file
MAX_PEOPLE_CHANGE_PER_FRAME = 2

# --- 2. INITIALIZE APP AND MODEL ---
app = Flask(__name__)
model = YOLO('yolov8n.pt')
camera = cv2.VideoCapture(0)

# --- 3. SECURITY CHECKS & STATE VARIABLES ---
def check_model_integrity(file_path):
    """Verifies the SHA256 hash of the model file."""
    if not os.path.exists(file_path):
        return False, "Model file not found!"
    with open(file_path, "rb") as f:
        current_hash = hashlib.sha256(f.read()).hexdigest()
    if current_hash == KNOWN_MODEL_HASH:
        return True, "Model OK"
    else:
        return False, "TAMPERED MODEL"

is_model_safe, model_status = check_model_integrity('yolov8n.pt')
if not is_model_safe:
    print(f"CRITICAL SECURITY ALERT: {model_status}")

# --- STATE VARIABLES FOR SECURITY ALERT ---
last_people_count = 0
# This will store the timestamp when the alert should automatically turn off
security_alert_active_until = 0

def generate_frames():
    """Captures frames, runs detection, and applies security logic."""
    global last_people_count, security_alert_active_until

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            if not is_model_safe:
                # Handle tampered model display
                cv2.putText(frame, "SECURITY ALERT: MODEL TAMPERED", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                ret, buffer = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                continue

            results = model(frame, classes=[0], verbose=False)
            annotated_frame = results[0].plot()
            people_count = len(results[0].boxes)
            
            # --- Anomaly Detection and Alert Latching Logic ---
            change = abs(people_count - last_people_count)
            
            # *** THE FIX IS HERE (Part 1) ***
            # If a new anomaly is detected, set the alert timer for 5 seconds into the future.
            if change > MAX_PEOPLE_CHANGE_PER_FRAME:
                security_alert_active_until = time.time() + 5 # 5-second alert duration
            
            # Update the count for the next frame's comparison
            last_people_count = people_count

            # --- System Logic ---
            if people_count >= 4:
                light_color = "green"
                door_status = "Door Open"
            else:
                light_color = "red"
                door_status = "Door Closed"

            # --- Display Information ---
            # *** THE FIX IS HERE (Part 2) ***
            # Check if the alert timer is still active.
            if time.time() < security_alert_active_until:
                # If the alert is active, show the security warning.
                alert_text = "Potential Video Spoofing Detected!"
                cv2.putText(annotated_frame, "SECURITY ALERT:", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(annotated_frame, alert_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                # If the alert is not active, show the normal operational status.
                cv2.circle(annotated_frame, (50, 50), 30, (0, 255, 0) if light_color == "green" else (0, 0, 255), -1)
                text_info = f"People: {people_count} | Status: {door_status}"
                cv2.putText(annotated_frame, text_info, (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)