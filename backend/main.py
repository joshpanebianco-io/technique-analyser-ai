from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import cv2
import math
import numpy as np
import mediapipe as mp

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

mp_pose = mp.solutions.pose

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
    elif 90 < angle <= 140:
        return (140 - angle) / 50
    else:
        return 0.0

def torso_score(angle):
    if 70 <= angle <= 110:
        return 1.0
    elif angle < 70:
        return max(0, (angle - 50) / 20)
    else:
        return max(0, (130 - angle) / 20)

def depth_feedback(angle):
    if angle <= 90:
        return "Excellent depth!"
    elif angle <= 110:
        return "Good depth, try to go a bit lower."
    elif angle <= 130:
        return "Shallow squat, aim for deeper bend."
    else:
        return "Very shallow squat, bend knees more."

def posture_feedback(angle, side="left"):
    if 70 <= angle <= 110:
        return "Great upright torso posture."

    if (side == "left" and angle < 70) or (side != "left" and angle > 110):
        return "Torso leaning too far forward, try to stay more upright."

    return "Torso leaning forward, keep chest up."



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
            landmarks = [(lm.x, lm.y, lm.z, lm.visibility) for lm in results.pose_landmarks.landmark]
            all_landmarks.append(landmarks)
        else:
            all_landmarks.append(None)

    cap.release()
    pose.close()
    return all_landmarks

def analyze_landmarks(landmarks: list):
    if not landmarks or all(l is None for l in landmarks):
        return {
            "rep_count": 0,
            "avg_score": 0,
            "rep_feedback": [],
            "suggestions": ["No landmarks extracted. Please try re-recording."]
        }

    mp_indices = mp_pose.PoseLandmark

    # Decide left or right based on visibility
    left_valid = sum(
        1 for f in landmarks
        if f and f[mp_indices.LEFT_KNEE.value][3] > 0.5
    )
    right_valid = sum(
        1 for f in landmarks
        if f and f[mp_indices.RIGHT_KNEE.value][3] > 0.5
    )
    preferred_side = "left" if left_valid >= right_valid else "right"

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

            if hip[3] < 0.5 or knee[3] < 0.5 or ankle[3] < 0.5 or shoulder[3] < 0.5:
                return None, None, None, None
            return hip, knee, ankle, shoulder
        except:
            return None, None, None, None

    rep_count = 0
    rep_feedback = []

    rep_in_progress = False
    bottom_reached = False
    min_knee_angle = 180
    rep_min_knee_angles = []
    rep_torso_angles = []

    DEPTH_TRIGGER = 140
    STAND_THRESHOLD = 160
    MIN_REP_RANGE = 20

    for frame in landmarks:
        if frame is None:
            continue

        hip, knee, ankle, shoulder = get_side_landmarks(frame, preferred_side)
        if None in [hip, knee, ankle, shoulder]:
            continue

        knee_angle = calculate_angle_3d(hip, knee, ankle)
        torso_angle = calculate_angle_3d(shoulder, hip, knee)

        if not rep_in_progress:
            if knee_angle < DEPTH_TRIGGER:
                rep_in_progress = True
                bottom_reached = False
                min_knee_angle = knee_angle
                rep_min_knee_angles = [knee_angle]
                rep_torso_angles = [torso_angle]
        else:
            rep_min_knee_angles.append(knee_angle)
            rep_torso_angles.append(torso_angle)
            if knee_angle < min_knee_angle:
                min_knee_angle = knee_angle

            if not bottom_reached and knee_angle > min_knee_angle + 2:
                bottom_reached = True

            if bottom_reached and knee_angle > STAND_THRESHOLD:
                if (STAND_THRESHOLD - min_knee_angle) >= MIN_REP_RANGE:
                    ks = knee_score(min_knee_angle)
                    ts = np.mean([torso_score(a) for a in rep_torso_angles])
                    rep_score = (ks * 0.6 + ts * 0.4) * 100
                    rep_count += 1

                    rep_feedback.append({
                        "rep_number": rep_count,
                        "score": round(rep_score),
                        "min_knee_angle": round(min_knee_angle, 1),
                        "avg_torso_angle": round(np.mean(rep_torso_angles), 1),
                        "depth_feedback": depth_feedback(min_knee_angle),
                        "posture_feedback": posture_feedback(np.mean(rep_torso_angles), preferred_side)
                    })

                rep_in_progress = False
                bottom_reached = False
                min_knee_angle = 180
                rep_min_knee_angles = []
                rep_torso_angles = []

    overall_avg_score = round(np.mean([r["score"] for r in rep_feedback])) if rep_feedback else 0
    avg_knee = np.mean([r["min_knee_angle"] for r in rep_feedback]) if rep_feedback else 180
    avg_torso = np.mean([r["avg_torso_angle"] for r in rep_feedback]) if rep_feedback else 90

    suggestions = []
    if overall_avg_score == 0:
        suggestions.append("No valid reps detected. Try recording a clearer squat video.")
    else:
        if np.mean([knee_score(r["min_knee_angle"]) for r in rep_feedback]) < 0.7:
            suggestions.append("Try to squat deeper (bend your knees more).")
        else:
            suggestions.append("Good depth in your squat!")

        if np.mean([torso_score(r["avg_torso_angle"]) for r in rep_feedback]) < 0.7:
            suggestions.append("Keep your torso more upright during the squat.")
        else:
            suggestions.append("Nice upright posture!")

    return {
        "rep_count": rep_count,
        "avg_score": overall_avg_score,
        "rep_feedback": rep_feedback,
        "suggestions": suggestions,
    }

@app.get("/")
def read_root():
    return {"message": "FastAPI is running!"}

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Only video files are allowed.")

    file_location = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        landmarks = extract_pose_landmarks(file_location)
        analysis_result = analyze_landmarks(landmarks)

        os.remove(file_location)

        return {
            "filename": file.filename,
            "message": "Upload and analysis complete",
            **analysis_result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")
