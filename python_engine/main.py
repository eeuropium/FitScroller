import cv2
import mediapipe as mp
import pyautogui
from flask import Flask, jsonify
from flask_cors import CORS
import threading
import time
import logging

# --- SILENCE FLASK LOGS ---
# log = logging.getLogger('werkzeug')
# log.setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)
stats = {"reps": 0, "time": "00:00"}

@app.route('/stats')
def get_stats():
    return jsonify(stats)

# Function to run the server in the background
def run_server():
    # Use port 5001 to avoid AirPlay conflict
    app.run(host='127.0.0.1', port=5001, debug=False, use_reloader=False)

def main():
    global stats
    print("Starting AI Engine...")

    # Start Flask in a background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print("Server started on http://127.0.0.1:5001")

    # Initialize AI (On Main Thread)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.7)
    cap = cv2.VideoCapture(0)

    counter = 0
    stage = None
    start_time = time.time()

    while cap.isOpened():
        print("RNING")
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            s_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
            e_y = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y

            # Detection Logic
            if s_y > (e_y - 0.05):
                stage = "down"
            if s_y < (e_y - 0.15) and stage == "down":
                stage = "up"
                counter += 1
                pyautogui.press('down')
                stats["reps"] = counter
                print(f"Rep: {counter}")

        # Update Timer
        elapsed = int(time.time() - start_time)
        stats["time"] = f"{elapsed // 60:02d}:{elapsed % 60:02d}"

        # Visual Feedback
        cv2.putText(frame, f"Reps: {counter}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('AI Engine (Keep Open)', frame)

        # Break on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
