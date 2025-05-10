import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from backend.pose_detection.mediapipe_model import PoseDetector

# ... rest of your script ...


# Setup
input_dir = "workout_videos"
output_dir = "pose_references"
model = PoseDetector()

# Process all *_correct.mp4 videos
for file in os.listdir(input_dir):
    if file.endswith("_correct.mp4"):
        name = file.replace("_correct.mp4", "")
        video_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, f"{name}_reference.npy")

        # ✅ Skip if already exists
        if os.path.exists(output_path):
            print(f"🟡 Skipping {name} — already exists.")
            continue

        print(f"⏳ Processing {file}...")

        cap = cv2.VideoCapture(video_path)
        pose_vectors = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.detect_pose(frame)
            landmarks = model.get_landmarks(results)
            if landmarks:
                vector = np.array([lm[:3] for lm in landmarks]).flatten()
                pose_vectors.append(vector)
        cap.release()

        if pose_vectors:
            avg_vector = np.mean(pose_vectors, axis=0)
            np.save(output_path, avg_vector)
            print(f"✅ Saved: {output_path}")
        else:
            print(f"❌ No valid poses found in {file}")
