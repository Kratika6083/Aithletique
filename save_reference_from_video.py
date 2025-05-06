import cv2
import os
import numpy as np
from backend.feedback_engine.pose_comparator import extract_important_angles, save_reference_angles
import mediapipe as mp

# CONFIGURATION
VIDEO_DIR = "pose videos"  # Folder where your videos are stored
REFERENCE_DIR = "pose_references"  # Folder to save references

if not os.path.exists(REFERENCE_DIR):
    os.makedirs(REFERENCE_DIR)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

# Step 1: List all videos
videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith((".mp4", ".mov", ".avi"))]

print("\n🎥 Available Videos:")
for idx, video_name in enumerate(videos):
    print(f"{idx}: {video_name}")

video_idx = int(input("\n👉 Enter the number of the video you want to process: "))
selected_video = videos[video_idx]
pose_name = input("\n🧘‍♂️ Enter the pose name to save (example: tadasana): ").strip().lower()

# Step 2: Load the selected video
video_path = os.path.join(VIDEO_DIR, selected_video)
cap = cv2.VideoCapture(video_path)

print("\n📸 Press 's' to save the current frame as reference.")
print("⏩ Press 'd' to move to next frame, 'a' to move back, 'q' to quit.\n")

frame_pos = 0
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
    ret, frame = cap.read()

    if not ret:
        print("\n⚠️ End of video reached.")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    show_frame = frame.copy()

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(show_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Select Best Frame", show_frame)

    key = cv2.waitKey(0) & 0xFF

    if key == ord('d'):
        frame_pos += 5  # Move 5 frames forward
        if frame_pos >= frame_count:
            frame_pos = frame_count - 1

    elif key == ord('a'):
        frame_pos -= 5  # Move 5 frames backward
        if frame_pos < 0:
            frame_pos = 0

    elif key == ord('s'):
        if results.pose_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            angles = extract_important_angles(landmarks)

            save_path = os.path.join(REFERENCE_DIR, f"{pose_name}.npz")
            save_reference_angles(angles, save_path)

            print(f"\n✅ Saved reference for {pose_name} at: {save_path}\n")
            break
        else:
            print("\n⚠️ No pose detected in this frame. Try a different frame.\n")

    elif key == ord('q'):
        print("\n❌ Exiting without saving.\n")
        break

cap.release()
cv2.destroyAllWindows()
