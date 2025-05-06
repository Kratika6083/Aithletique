# save_motion_reference.py
import cv2
import numpy as np
import os
from backend.pose_detection.mediapipe_model import PoseDetector

video_folder = "pose_videos"  # 📂 Make sure your video is here
save_folder = "motion_references"
os.makedirs(save_folder, exist_ok=True)

detector = PoseDetector()

print("\n🎥 Available Videos:")
videos = [v for v in os.listdir(video_folder) if v.endswith(('.mp4', '.avi', '.mov'))]
for idx, vid in enumerate(videos):
    print(f"{idx}: {vid}")

video_idx = int(input("\n👉 Enter video number to process: "))
video_path = os.path.join(video_folder, videos[video_idx])

cap = cv2.VideoCapture(video_path)
frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    results = detector.detect_pose(frame)
    landmarks = detector.get_landmarks(results)

    if landmarks:
        frames.append(landmarks)

cap.release()

if frames:
    pose_name = input("💾 Enter workout name (e.g., squat): ").strip().lower()
    np.savez(os.path.join(save_folder, f"{pose_name}_motion.npz"), landmarks=frames)
    print(f"✅ Saved motion reference for '{pose_name}' with {len(frames)} frames!")
else:
    print("⚠️ No poses detected!")

