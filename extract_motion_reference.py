import cv2
import os
import numpy as np
import mediapipe as mp

# 📂 Your Video
VIDEO_PATH = "squats.mp4"  # Adjust if saved elsewhere
OUTPUT_PATH = "motion_references/squat_motion.npz"

# 🧠 Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# 🎥 Open Video
cap = cv2.VideoCapture(VIDEO_PATH)

frame_count = 0
landmark_sequence = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Only process every 5th frame (adjustable)
    if frame_count % 5 != 0:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        frame_landmarks = []
        for lm in results.pose_landmarks.landmark:
            frame_landmarks.append([lm.x, lm.y, lm.z, lm.visibility])

        landmark_sequence.append(frame_landmarks)

cap.release()

# ✅ Save as npz
if landmark_sequence:
    np.savez_compressed(OUTPUT_PATH, landmarks=np.array(landmark_sequence))
    print(f"✅ Saved motion sequence to: {OUTPUT_PATH}")
else:
    print("⚠️ No landmarks detected!")

pose.close()
