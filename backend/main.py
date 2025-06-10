from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import cv2
import mediapipe as mp # Still imported but not used in the dummy scenario

app = FastAPI()

# Allow React frontend to communicate with FastAPI backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple test route
@app.get("/")
def read_root():
    return {"message": "FastAPI is running!"}

# Upload directory setup
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# # Uncomment and use these functions when you're ready to integrate real analysis
# mp_pose = mp.solutions.pose
#
# def extract_pose_landmarks(video_path: str):
#     """
#     (Dummy version or commented out if not used)
#     Extracts pose landmarks from a given video file using MediaPipe.
#     """
#     print("DUMMY: Skipping actual landmark extraction.")
#     # In a real scenario, this would contain your MediaPipe logic
#     return []
#
# def analyze_landmarks(landmarks: list):
#     """
#     (Dummy version or commented out if not used)
#     A placeholder function to analyze extracted pose landmarks.
#     """
#     print("DUMMY: Skipping actual landmark analysis.")
#     return 0, ["DUMMY: No real suggestions provided yet."]

# Video upload endpoint
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Only video files are allowed.")

    file_location = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # Save the uploaded file
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"File saved to: {file_location}")

        # --- DUMMY DATA GENERATION ---
        # Instead of calling real analysis, we'll return dummy data
        dummy_score = 75 # Example dummy score
        dummy_suggestions = [
            "Maintain a steady pace throughout the exercise.",
            "Ensure full range of motion in your movements.",
            "Keep your core engaged for better stability.",
            "Your form looks promising, keep practicing!",
            "Consider a slight adjustment in your foot placement."
        ]
        # --- END DUMMY DATA ---

        # Clean up the uploaded file after processing (optional, but good for storage)
        os.remove(file_location)
        print(f"File {file.filename} removed after processing.")

        return {
            "filename": file.filename,
            "message": "Upload successful and dummy analysis complete",
            "score": dummy_score,
            "suggestions": dummy_suggestions
        }
    except Exception as e:
        # Catch any errors during file handling
        print(f"Error during upload or processing: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

