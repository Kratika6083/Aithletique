# auto_save_yoga_reference.py
import cv2
import numpy as np
import os
from backend.pose_detection.mediapipe_model import PoseDetector

# 📂 Folders
video_folder = "pose_videos"
save_folder = "pose_references"
os.makedirs(save_folder, exist_ok=True)

# 🔥 Initialize pose detector
detector = PoseDetector()

# 📜 List available videos
print("\n🎥 Available Videos:")
videos = [v for v in os.listdir(video_folder) if v.endswith(('.mp4', '.avi', '.mov'))]
for idx, vid in enumerate(videos):
    print(f"{idx}: {vid}")

# 👉 Select video
video_idx = int(input("\n👉 Enter the number of the video you want to process: "))
video_path = os.path.join(video_folder, videos[video_idx])

cap = cv2.VideoCapture(video_path)

# 🧠 Settings
FRAME_SKIP = 5  # Take every 5th frame (you can adjust if needed)
max_frames = 30  # Max frames to average (to avoid too long videos)

landmark_list = []
frame_counter = 0

print("\n🧘‍♂️ Extracting landmarks automatically...")

while cap.isOpened() and len(landmark_list) < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    if frame_counter % FRAME_SKIP != 0:
        continue

    frame = cv2.flip(frame, 1)
    results = detector.detect_pose(frame)
    landmarks = detector.get_landmarks(results)

    if landmarks:
        landmark_list.append(landmarks)
        print(f"✅ Collected frame {len(landmark_list)} landmarks.")

cap.release()

if landmark_list:
    avg_landmarks = np.mean(np.array(landmark_list), axis=0)
    pose_name = input("\n💾 Enter pose name (e.g., tadasana): ").strip().lower()
    np.savez(os.path.join(save_folder, f"{pose_name}.npz"), landmarks=avg_landmarks)
    print(f"\n✅ Saved averaged yoga reference '{pose_name}' with {len(landmark_list)} frames!")
else:
    print("⚠️ No valid landmarks extracted!")

