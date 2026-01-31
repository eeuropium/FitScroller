import cv2
import mediapipe as mp
import pyautogui
from flask import Flask, jsonify
from flask_cors import CORS
import threading
import time
import logging
import numpy as np
import webbrowser

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

def calculate_angle(a, b, c):
    """Calculates the angle between three points (Shoulder, Elbow, Wrist)"""
    a = np.array([a.x, a.y]) # Shoulder
    b = np.array([b.x, b.y]) # Elbow
    c = np.array([c.x, c.y]) # Wrist

    # Calculate the radians and convert to degrees
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def main():
    global stats

    # Start Flask in a background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print("Server started on http://127.0.0.1:5001")


    print("Launching Instagram Reels...")
    webbrowser.open("https://www.instagram.com/reels/")

    # Give the browser a second to load before starting the thread
    time.sleep(2)

    # Initialize AI (On Main Thread)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.7)
    cap = cv2.VideoCapture(0)

    counter = 0
    stage = None
    start_time = time.time()

    while cap.isOpened():
        # print("RNING")
        ret, frame = cap.read()
        if not ret:
            break

        # Get frame dimensions for pixel mapping
        h, w, _ = frame.shape

        # Process frame
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            # --- START DRAWING SECTION ---
            # 1. Draw the full skeleton connections (thin lines)
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
            )

            landmarks = results.pose_landmarks.landmark

            ''' Method 2 - Angle '''
            # # Get coordinates
            # shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            # elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
            # wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
            #
            # # Calculate Angle
            # angle = calculate_angle(shoulder, elbow, wrist)
            #
            # # Draw angle on screen
            # e_pos = (int(elbow.x * w), int(elbow.y * h))
            # cv2.putText(frame, str(int(angle)), e_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            #
            # # Detection Logic
            # # 90 degrees is a good "down" position, 160 is "up"
            # if angle < 90:
            #     stage = "down"
            # if angle > 160 and stage == "down":
            #     stage = "up"
            #     counter += 1
            #     pyautogui.press('down')
            #     stats["reps"] = counter

            ''' Method 1 - Shoulder and Elbow '''
            # 2. Get pixel coordinates for visual dots
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]

            s_pos = (int(left_shoulder.x * w), int(left_shoulder.y * h))
            e_pos = (int(left_elbow.x * w), int(left_elbow.y * h))

            # 3. Highlight the specific points used for logic (Shoulder=Blue, Elbow=Red)
            cv2.circle(frame, s_pos, 8, (255, 0, 0), -1)
            cv2.circle(frame, e_pos, 8, (0, 0, 255), -1)
            # --- END DRAWING SECTION ---

            s_y = left_shoulder.y
            e_y = left_elbow.y

            # Detection Logic (Existing)
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
