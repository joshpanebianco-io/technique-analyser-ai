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
        return 1.0
    elif 90 < angle <= 140:
        return (140 - angle) / 50
    else:
        return 0.0

def torso_score(angle):
    if 75 <= angle <= 105:
        return 1.0
    elif angle < 75:
        return max(0, (angle - 50) / 25)
    else:
        return max(0, (130 - angle) / 25)

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
    if 75 <= angle <= 105:
        return "Great upright torso posture."
    if (side == "left" and angle < 75) or (side != "left" and angle > 105):
        return "Torso leaning too far forward, try to stay more upright."
    return "Torso leaning forward, keep chest up."

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

                if all(lm[3] > 0.5 for lm in [hip, knee, ankle, shoulder]):
                    knee_angle = calculate_angle_3d(hip, knee, ankle)
                    torso_angle = calculate_angle_3d(shoulder, hip, knee)

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
