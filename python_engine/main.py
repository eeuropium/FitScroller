import cv2
import mediapipe as mp
import pyautogui
from flask import Flask, jsonify
from flask_cors import CORS
import threading
import time

# --- FLASK SERVER SETUP ---
app = Flask(__name__)
CORS(app) # Allows the Chrome Extension to talk to this script
stats = {"reps": 0, "time": "00:00"}

@app.route('/stats')
def get_stats():
    return jsonify(stats)

# --- AI DETECTION LOGIC ---
def run_detection():
    global stats
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.7)
    cap = cv2.VideoCapture(0)
    counter = 0
    stage = None
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            s_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            e_y = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y

            if s_y > (e_y - 0.05): stage = "down"
            if s_y < (e_y - 0.15) and stage == "down":
                stage = "up"
                counter += 1
                pyautogui.press('down')
                stats["reps"] = counter

        elapsed = int(time.time() - start_time)
        stats["time"] = f"{elapsed // 60:02d}:{elapsed % 60:02d}"

        cv2.imshow('AI Engine (Minimize Me)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

# Start AI in a background thread
threading.Thread(target=run_detection, daemon=True).start()

if __name__ == '__main__':
    app.run(port=5001)
