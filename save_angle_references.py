import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from backend.pose_detection.mediapipe_model import PoseDetector
from backend.feedback_engine.angles import calculate_angle_from_landmarks

# Define joints of interest for each workout
angle_joints = {
    "pushup": {
        "elbow": (11, 13, 15),
        "back": (11, 23, 24)
    },
    "plank": {
        "shoulder_hip_knee": (11, 23, 25),
        "back": (11, 23, 24)
    },
    "pullup": {
        "elbow_shoulder_hip": (13, 11, 23),
        "back": (11, 23, 24)
    }
}

input_dir = "workout_videos"
output_dir = "pose_references"
model = PoseDetector()

for file in os.listdir(input_dir):
    if file.endswith("_correct.mp4"):
        name = file.replace("_correct.mp4", "")
        if name not in angle_joints:
            print(f"Skipping {name} — no angle config defined.")
            continue

        print(f"Processing angles for: {name}")
        cap = cv2.VideoCapture(os.path.join(input_dir, file))
        angle_records = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.detect_pose(frame)
            landmarks = model.get_landmarks(results)
            if not landmarks:
                continue

            angles = {}
            for label, (a, b, c) in angle_joints[name].items():
                angle = calculate_angle_from_landmarks(landmarks, a, b, c)
                angles[label] = angle

            angle_records.append(angles)

        cap.release()

        if angle_records:
            output_path = os.path.join(output_dir, f"{name}_angles_reference.npy")
            np.save(output_path, angle_records)
            print(f"✅ Saved: {output_path}")
        else:
            print(f"❌ No valid angles found for {name}")
