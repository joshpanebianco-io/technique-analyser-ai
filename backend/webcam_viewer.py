import cv2
import math
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle_3d(a, b, c):
    a, b, c = np.array(a[:3]), np.array(b[:3]), np.array(c[:3])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return math.degrees(angle)

def knee_score(angle):
    if angle <= 90:
        return 1.0  # Excellent depth
    elif angle <= 105:
        return 0.8  # Good
    elif angle <= 120:
        return 0.5  # Fair
    elif angle <= 135:
        return 0.2  # Poor
    else:
        return 0.0  # Too shallow


def torso_score(angle):
    if angle < 25:
        return 1.0  # Excellent posture
    elif angle < 40:
        return 0.8  # Good posture
    elif angle < 55:
        return 0.5  # Noticeable lean
    elif angle < 70:
        return 0.2  # Excessive lean
    else:
        return 0.0  # Likely bad posture or missing torso detection


def depth_feedback(angle):
    if angle <= 90:
        return "Excellent squat depth!"
    elif angle <= 105:
        return "Good depth — just a little deeper for perfection."
    elif angle <= 120:
        return "Fair depth — work on reaching parallel."
    elif angle <= 135:
        return "Shallow — aim to lower yourself more."
    else:
        return "Too shallow — bend your knees further and sit deeper."


def posture_feedback(angle, side="left"):
    if angle < 25:
        return "Excellent upright torso posture."
    elif angle < 40:
        return "Good posture — aim to stay tall and balanced."
    elif angle < 55:
        return "Torso leaning forward — lift your chest and brace your core."
    elif angle < 70:
        return "Excessive forward lean — focus on staying upright throughout the squat."
    else:
        return "Severe torso angle — posture likely needs significant correction or visibility may be poor."
    

def main():
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        rep_in_progress = False
        bottom_reached = False
        rep_count = 0
        min_knee_angle = 180
        rep_scores = []
        rep_feedback = []

        DEPTH_TRIGGER = 130
        STAND_THRESHOLD = 160
        MIN_REP_RANGE = 30
        feedback = ""

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                )

                landmarks = results.pose_landmarks.landmark
                visibility_left = landmarks[mp_pose.PoseLandmark.LEFT_HIP].visibility
                visibility_right = landmarks[mp_pose.PoseLandmark.RIGHT_HIP].visibility
                side = "LEFT" if visibility_left > visibility_right else "RIGHT"

                idx = mp_pose.PoseLandmark
                hip = landmarks[idx.LEFT_HIP] if side == "LEFT" else landmarks[idx.RIGHT_HIP]
                knee = landmarks[idx.LEFT_KNEE] if side == "LEFT" else landmarks[idx.RIGHT_KNEE]
                ankle = landmarks[idx.LEFT_ANKLE] if side == "LEFT" else landmarks[idx.RIGHT_ANKLE]
                shoulder = landmarks[idx.LEFT_SHOULDER] if side == "LEFT" else landmarks[idx.RIGHT_SHOULDER]

                def lm_coords(lm): return [lm.x, lm.y, lm.z, lm.visibility]
                hip, knee, ankle, shoulder = map(lm_coords, [hip, knee, ankle, shoulder])
                
                def calculate_torso_angle(shoulder, hip):
                    # Vector from hip to shoulder
                    torso_vector = np.array(shoulder[:3]) - np.array(hip[:3])

                    # Reference vertical vector (pointing straight up in Y direction)
                    vertical_vector = np.array([0, -1, 0])  # -Y because in image coordinates, down is positive

                    # Calculate angle between torso and vertical
                    cosine_angle = np.dot(torso_vector, vertical_vector) / (np.linalg.norm(torso_vector) + 1e-8)
                    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
                    return math.degrees(angle)

                if all(lm[3] > 0.5 for lm in [hip, knee, ankle, shoulder]):
                    knee_angle = calculate_angle_3d(hip, knee, ankle)
                    torso_angle = calculate_torso_angle(shoulder, hip)

                    if knee_angle < min_knee_angle:
                        min_knee_angle = knee_angle

                    if not rep_in_progress and knee_angle < DEPTH_TRIGGER:
                        rep_in_progress = True
                        bottom_reached = False
                        min_knee_angle = knee_angle
                        feedback = "Going down..."

                    if rep_in_progress and not bottom_reached and knee_angle > min_knee_angle + 5:
                        bottom_reached = True
                        feedback = "Bottom reached"

                    if rep_in_progress and bottom_reached and knee_angle > STAND_THRESHOLD:
                        angle_range = STAND_THRESHOLD - min_knee_angle
                        if angle_range >= MIN_REP_RANGE:
                            ks = knee_score(min_knee_angle)
                            ts = torso_score(torso_angle)
                            rep_score = (ks * 0.6 + ts * 0.4) * 100
                            rep_scores.append(rep_score)
                            rep_count += 1

                            df = depth_feedback(min_knee_angle)
                            pf = posture_feedback(torso_angle, side.lower())
                            feedback = f"Rep {rep_count} done! {df} {pf}"
                            rep_feedback.append((rep_count, rep_score, df, pf))
                        else:
                            feedback = "Movement too small, rep ignored."

                        rep_in_progress = False
                        bottom_reached = False
                        min_knee_angle = 180

                    cv2.putText(frame, f"Knee: {knee_angle:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame, f"Torso: {torso_angle:.1f}", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(frame, f"Side: {side}", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.putText(frame, f"Reps: {rep_count}", (10, height - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.putText(frame, feedback, (10, height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if rep_scores:
                avg_score = int(sum(rep_scores) / len(rep_scores))
                cv2.putText(frame, f"Avg Score: {avg_score}%", (width - 200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Squat Analyzer", frame)

            key = cv2.waitKey(5) & 0xFF
            if key == 27 or key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        print("\n--- Squat Set Summary ---")
        print(f"Total reps: {rep_count}")
        if rep_scores:
            print(f"Average score: {int(sum(rep_scores) / len(rep_scores))}%")
            print("\nRep-by-rep breakdown:")
            for count, score, df, pf in rep_feedback:
                print(f"Rep {count}: {score:.1f}% — {df} | {pf}")
        else:
            print("No valid reps scored.")

if __name__ == "__main__":
    main()
