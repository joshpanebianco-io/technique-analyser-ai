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
    if angle <= 95:  # Was 90 — expanded by 5 degrees
        return 1.0  # Excellent depth
    elif 95 < angle <= 110:
        return 0.7  # Good depth
    elif 110 < angle <= 140:
        return max(0.3, (140 - angle) / 25 * 0.7)  # Gradual decline
    else:
        return 0.0  # Too shallow


def torso_score(angle):
    if 70 <= angle <= 110:  # Was 75–105 — expanded by 5° on each end
        return 1.0  # Ideal upright posture
    elif 65 <= angle < 70 or 110 < angle <= 115:
        return 0.8  # Slight lean, still acceptable
    elif angle < 65:
        return max(0.3, (angle - 60) / 5 * 0.5)  # Forward lean penalty
    else:
        return 0.0  # Severe backward lean


def depth_feedback(angle):
    if angle <= 95:  # Matches revised "Excellent" threshold
        return "Excellent squat depth!"
    elif angle <= 110:
        return "Decent depth — just a little more and it’s perfect."
    elif angle <= 115:
        return "Need more depth — keep working toward deeper range."
    elif angle <= 130:
        return "Shallow — need to go much deeper."
    else:
        return "Too shallow — try to sit lower into the squat."


def posture_feedback(angle, side="left"):
    if 70 <= angle <= 110:  # Matches revised "Excellent" range
        return "Excellent upright torso posture."
    elif 65 <= angle < 70 or 110 < angle <= 115:
        return "Good posture — just a slight adjustment needed."
    elif angle < 65:
        return "Torso leaning too far forward — keep your chest up."
    else:
        return "It's supposed to be a squat, not a good morning."





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
            "set_feedback": ["No landmarks extracted. Please try re-recording."]
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

    set_feedback = []
    if overall_avg_score == 0:
        set_feedback.append("❌ No valid squats detected — please ensure you're visible and your form is clear in the video.")
    else:
        avg_knee_score = np.mean([knee_score(r["min_knee_angle"]) for r in rep_feedback])
        avg_torso_score = np.mean([torso_score(r["avg_torso_angle"]) for r in rep_feedback])

        if avg_knee_score >= 0.9:
            set_feedback.append("✅ Excellent squat depth across the set — great job hitting depth!")
        elif avg_knee_score >= 0.7:
            set_feedback.append("👍 Good depth overall, but aim to consistently reach parallel or slightly below.")
        elif avg_knee_score >= 0.4:
            set_feedback.append("⚠️ Some reps were too shallow — try to bend your knees more to reach proper depth.")
        else:
            set_feedback.append("❗ Most reps lacked depth. Focus on sitting back and lowering yourself more into the squat.")

        if avg_torso_score >= 0.9:
            set_feedback.append("✅ Fantastic torso posture — you maintained an upright position throughout.")
        elif avg_torso_score >= 0.7:
            set_feedback.append("👍 Decent posture overall, but keep working on staying more upright during the descent.")
        elif avg_torso_score >= 0.4:
            set_feedback.append("⚠️ There was noticeable forward lean — keep your chest up and engage your core.")
        else:
            set_feedback.append("❗ Excessive torso lean in most reps — work on mobility and torso control to avoid tipping forward.")


    return {
        "rep_count": rep_count,
        "avg_score": overall_avg_score,
        "rep_feedback": rep_feedback,
        "set_feedback": set_feedback,
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
