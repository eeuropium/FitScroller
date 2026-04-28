# Fit Scroller

A Chrome extension that makes you earn every Instagram Reel. Do a push-up, scroll to the next reel.

---

https://github.com/user-attachments/assets/b71c9694-7ba1-4c69-928e-272402527aa8


## 🎯 How It Works

While you watch Instagram Reels, a HUD (heads-up display) sits in the corner of your screen showing your **live rep count** and **session timer**.

Under the hood:
1. Your **webcam** captures your movement
2. **MediaPipe + OpenCV** track your body pose and count push-up reps
3. A **Flask server** exposes this data locally
4. The **Chrome extension** polls the API and updates the HUD in real time

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Chrome Extension (JS / CSS), Manifest V3 |
| Communication | Flask (local API bridge) |
| Pose Tracking | MediaPipe, OpenCV |
| Input Simulation | PyAutoGUI |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- Google Chrome

### 1. Install Python dependencies

```bash
pip install flask opencv-python mediapipe pyautogui
```

### 2. Run the backend

```bash
python backend/server.py
```

### 3. Load the Chrome extension

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable **Developer mode** (top right toggle)
3. Click **Load unpacked** and select the `chrome_extension/` folder

### 4. Start scrolling (and repping)

Navigate to any Instagram Reel at `instagram.com/reels/`. The HUD will appear in the top-right corner of your screen.

---

## ⚠️ Notes
- The backend must be running **before** you open Instagram, otherwise the HUD will display `"Start Python Script..."`
- Make sure your webcam is accessible and unobstructed
- 
