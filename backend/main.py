from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import cv2
import math
import numpy as np
import mediapipe as mp

app = FastAPI()

# Allow React frontend to communicate with FastAPI backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "FastAPI is running!"}

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

mp_pose = mp.solutions.pose

def extract_pose_landmarks(video_path: str):
    cap = cv2.VideoCapture(video_path)
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    all_landmarks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Extract landmarks with visibility confidence
            landmarks = [
                (lm.x, lm.y, lm.z, lm.visibility) for lm in results.pose_landmarks.landmark
            ]
            all_landmarks.append(landmarks)
        else:
            all_landmarks.append(None)

    cap.release()
    pose.close()
    return all_landmarks

def calculate_angle_3d(a, b, c):
    a = np.array(a[:3])
    b = np.array(b[:3])
    c = np.array(c[:3])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return math.degrees(angle)

def knee_score(angle):
    if angle <= 90:
        return 1.0
    elif 90 < angle <= 120:
        return (120 - angle) / 30
    else:
        return 0.0

def torso_score(angle):
    # Ideal torso angle roughly between 70° and 110°
    if 70 <= angle <= 110:
        return 1.0
    elif angle < 70:
        return max(0, (angle - 50) / 20)
    else:
        return max(0, (130 - angle) / 20)

def analyze_landmarks(landmarks: list):
    if not landmarks or all(l is None for l in landmarks):
        return 0, ["No landmarks extracted. Please try re-recording."]

    mp_indices = mp_pose.PoseLandmark

    # Choose side with more valid landmarks overall
    left_valid = sum(
        1
        for f in landmarks
        if f
        and f[mp_indices.LEFT_KNEE.value][3] > 0.5  # visibility threshold
        and f[mp_indices.LEFT_KNEE.value][0] > 0
    )
    right_valid = sum(
        1
        for f in landmarks
        if f
        and f[mp_indices.RIGHT_KNEE.value][3] > 0.5
        and f[mp_indices.RIGHT_KNEE.value][0] > 0
    )
    preferred_side = "left" if left_valid >= right_valid else "right"

    # Helper to get relevant landmarks for angles
    def get_side_landmarks(frame, side="left"):
        try:
            if side == "left":
                hip = frame[mp_indices.LEFT_HIP.value]
                knee = frame[mp_indices.LEFT_KNEE.value]
                ankle = frame[mp_indices.LEFT_ANKLE.value]
                shoulder = frame[mp_indices.LEFT_SHOULDER.value]
            else:
                hip = frame[mp_indices.RIGHT_HIP.value]
                knee = frame[mp_indices.RIGHT_KNEE.value]
                ankle = frame[mp_indices.RIGHT_ANKLE.value]
                shoulder = frame[mp_indices.RIGHT_SHOULDER.value]

            # Check visibility for all
            if (
                hip[3] < 0.5
                or knee[3] < 0.5
                or ankle[3] < 0.5
                or shoulder[3] < 0.5
            ):
                return None, None, None, None
            return hip, knee, ankle, shoulder
        except:
            return None, None, None, None

    # Extract knee angles and torso angles per frame (only frames with good landmarks)
    knee_angles = []
    torso_angles = []

    for frame in landmarks:
        if frame is None:
            continue
        hip, knee, ankle, shoulder = get_side_landmarks(frame, preferred_side)
        if None in [hip, knee, ankle, shoulder]:
            continue

        # Calculate angles
        knee_ang = calculate_angle_3d(hip, knee, ankle)  # knee angle
        torso_ang = calculate_angle_3d(shoulder, hip, knee)  # hip/torso angle
        knee_angles.append(knee_ang)
        torso_angles.append(torso_ang)

    if len(knee_angles) == 0:
        return 0, ["No valid frames with full landmarks found. Try re-recording."]

    # Detect squat reps by finding local minima in knee angle (bottom of squat)
    # Simple approach: a frame is local min if knee_angle < previous and next frame
    rep_bottoms = []
    for i in range(1, len(knee_angles) - 1):
        if knee_angles[i] < knee_angles[i - 1] and knee_angles[i] < knee_angles[i + 1]:
            rep_bottoms.append(i)

    if len(rep_bottoms) == 0:
        # If no clear minima found, fallback: take min knee angle frame
        min_idx = np.argmin(knee_angles)
        rep_bottoms.append(min_idx)

    # Score each rep bottom frame
    rep_scores = []
    for idx in rep_bottoms:
        ks = knee_score(knee_angles[idx])
        ts = torso_score(torso_angles[idx])
        rep_score = (ks * 0.6) + (ts * 0.4)
        rep_scores.append(rep_score)

    final_score = round(np.mean(rep_scores) * 100)

    suggestions = []
    avg_knee_score = np.mean([knee_score(a) for a in knee_angles])
    avg_torso_score = np.mean([torso_score(a) for a in torso_angles])

    if avg_knee_score < 0.7:
        suggestions.append("Try to squat deeper (bend your knees more).")
    else:
        suggestions.append("Good depth in your squat!")

    if avg_torso_score < 0.7:
        suggestions.append("Keep your torso more upright during the squat.")
    else:
        suggestions.append("Nice upright posture!")

    return final_score, suggestions

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Only video files are allowed.")

    file_location = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"File saved to: {file_location}")

        landmarks = extract_pose_landmarks(file_location)
        score, suggestions = analyze_landmarks(landmarks)

        os.remove(file_location)
        print(f"File {file.filename} removed after processing.")

        return {
            "filename": file.filename,
            "message": "Upload and analysis complete",
            "score": score,
            "suggestions": suggestions
        }

    except Exception as e:
        print(f"Error during upload or processing: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")
