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
from collections import deque

# --- SILENCE FLASK LOGS ---
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)
stats = {"reps": 0, "time": "00:00", "camera_angle": "perfect"}

# import like image for finding like coordinates

import ctypes
import platform

def get_scale_factor():
    """Detects the DPI scaling factor of the monitor."""
    # 1. Get Logical Width (what the OS says)
    logic_w, _ = pyautogui.size()

    # 2. Get Physical Width (actual hardware pixels)
    if platform.system() == "Windows":
        # Force Windows to be DPI aware to get actual hardware metrics
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        phys_w = ctypes.windll.user32.GetSystemMetrics(0)
    elif platform.system() == "Darwin": # macOS
        # On Mac, locateOnScreen usually returns 2x coordinates on Retina
        # We can test this by checking a screenshot's size
        phys_w = pyautogui.screenshot().size[0]
    else:
        phys_w = logic_w # Default for Linux/others

    return phys_w / logic_w

# --- How to use it in your code ---
SCALE = get_scale_factor()
print(f"Detected Scale Factor: {SCALE}")


@app.route('/stats')
def get_stats():
    return jsonify(stats)

def sgn(x):
    return (x > 0) - (x < 0)

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

# for blinking detection
def get_ear(landmarks, eye_indices):
    """Calculates Normalized Eye Aspect Ratio (EAR)"""
    # Vertical distance (Top to Bottom)
    p1 = landmarks[eye_indices[0]] # Landmark 159
    p2 = landmarks[eye_indices[1]] # Landmark 145
    vertical_dist = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    # Horizontal distance (Left corner to Right corner of eye)
    # Using landmarks 33 and 133 for the left eye
    h_left = landmarks[33]
    h_right = landmarks[133]
    horizontal_dist = np.sqrt((h_left.x - h_right.x)**2 + (h_left.y - h_right.y)**2)

    # The Ratio: If this is < ~0.2, it's a blink, regardless of distance
    ear = vertical_dist / horizontal_dist
    return ear

def find_like():
    like_list = list(pyautogui.locateAllOnScreen('../assets/like.png', grayscale=True, confidence=0.5))
    like_list = [box for box in like_list if (box.left / SCALE) > pyautogui.size()[0] // 2]

    if len(like_list) > 0:
        like_box = like_list[0]
        like_loc = ((like_box.left + like_box.width // 2) // SCALE, (like_box.top + like_box.height // 2) // SCALE)
    else:
        like_loc = None

    return like_loc

def get_camera_angle_feedback(left_shoulder_y, right_shoulder_y, face_visible):
    """
    Determines if user needs to tilt camera based on shoulder position.
    Returns: "tilt_back", "tilt_forward", or "perfect"

    Logic:
    - If shoulders are too high in frame (y < 0.25), user should tilt camera back
    - If shoulders are too low in frame (y > 0.55), user should tilt camera forward
    - Otherwise, angle is perfect
    """
    if not face_visible:
        return "face_missing"

    # Average shoulder Y position (0 = top of frame, 1 = bottom of frame)
    avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2

    # Thresholds for camera angle
    if avg_shoulder_y < 0.10:
        return "tilt_back"
    elif avg_shoulder_y > 0.70:
        return "tilt_forward"
    else:
        return "perfect"

def main():
    global stats

    # Start Flask in a background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print("Server started on http://127.0.0.1:5001")


    print("Launching Instagram Reels...")
    webbrowser.open("https://www.instagram.com/reels/")

    # Give the browser a second to load before starting the thread
    time.sleep(5)

    # find heart locationt o like reels later
    like_location = pyautogui.locateCenterOnScreen('../assets/like.png', confidence=0.3)
    can_like = bool(like_location) # if canLike = False, we will have no liking feature


    # Initialize AI (On Main Thread)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.7)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7)

    cap = cv2.VideoCapture(0)

    counter = 0
    stage = None
    blink_active = False # Debounce for blinking
    start_time = time.time()

    # Face landmarks for left eye (Top: 159, Bottom: 145)
    LEFT_EYE_TOP_BOTTOM = [159, 145]

    # find heart location
    like_loc = find_like()

    print(like_loc)

    BUFFER_LEN = 30
    INIT_VAL = 0.5
    buffer = deque(maxlen = BUFFER_LEN) # start empty

    while cap.isOpened():
        if like_loc is None: # try again
            like_loc = find_like()
        # print("RNING")

        ret, frame = cap.read()
        if not ret:
            break

        # Get frame dimensions for pixel mapping
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        face_results = face_mesh.process(rgb_frame) # for blink
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # for pushup


        face_is_visible = bool(face_results.multi_face_landmarks)

        # 1. BLINK DETECTION (LIKE REEL)
        if like_loc is not None and face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                ear = get_ear(face_landmarks.landmark, LEFT_EYE_TOP_BOTTOM)

                buffer.append(ear)

                # Threshold for a blink (0.015 - 0.02 is typical)

                if len(buffer) == BUFFER_LEN:
                    diffs = [buffer[i] - buffer[i-1] for i in range(1, len(buffer))]

                    avg_diff = np.mean(diffs)
                    std_diff = np.std(diffs)


                    current_diff = buffer[-1] - buffer[-2]

                    deviation = 2 * std_diff

                    # print(abs(current_diff - avg_diff))

                    if (current_diff < avg_diff and current_diff < avg_diff - deviation) or (current_diff > avg_diff and current_diff > avg_diff + deviation):
                        if not blink_active:
                            print("Blink Detected! Liking...")
                            # Double click center of screen to like
                            pyautogui.click(like_loc)
                            blink_active = True
                    else:
                        blink_active = False

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
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

            s_pos = (int(left_shoulder.x * w), int(left_shoulder.y * h))
            e_pos = (int(left_elbow.x * w), int(left_elbow.y * h))

            # 3. Highlight the specific points used for logic (Shoulder=Blue, Elbow=Red)
            cv2.circle(frame, s_pos, 8, (255, 0, 0), -1)
            cv2.circle(frame, e_pos, 8, (0, 0, 255), -1)
            # --- END DRAWING SECTION ---

            s_y = left_shoulder.y
            e_y = left_elbow.y

            # Camera Angle Detection
            camera_feedback = get_camera_angle_feedback(left_shoulder.y, right_shoulder.y, face_is_visible)
            stats["camera_angle"] = camera_feedback

            # Detection Logic (Existing)
            if s_y > (e_y - 0.05):
                stage = "down"
            if s_y < (e_y - 0.15) and stage == "down":
                stage = "up"
                counter += 1
                pyautogui.press('down')
                stats["reps"] = counter
                print(f"Rep: {counter}")

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # This draws the dots on the eyes and face
                mp.solutions.drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                        color=(255, 255, 255), thickness=1, circle_radius=1
                    )
                )

                # OPTIONAL: Specifically highlight the two points used for EAR
                # This helps you see if the logic is matching your actual eyes
                for idx in LEFT_EYE_TOP_BOTTOM:
                    eye_pt = face_landmarks.landmark[idx]
                    pos = (int(eye_pt.x * w), int(eye_pt.y * h))
                    cv2.circle(frame, pos, 3, (0, 255, 255), -1)

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
