import cv2
import mediapipe as mp
import numpy as np
import os
from backend.feedback_engine.pose_comparator import extract_important_angles, save_reference_angles

# CONFIGURATION
REFERENCE_DIR = "pose_references"
POSE_NAME = "tadasana"  # Change this when creating other poses

if not os.path.exists(REFERENCE_DIR):
    os.makedirs(REFERENCE_DIR)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("\n📸 Press 's' to capture and save the reference pose.")
print("❌ Press 'q' to quit without saving.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Capture Reference Pose", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
        angles = extract_important_angles(landmarks)

        save_path = os.path.join(REFERENCE_DIR, f"{POSE_NAME}.npz")
        save_reference_angles(angles, save_path)

        print(f"\n✅ Reference pose saved successfully: {save_path}\n")
        break

    elif key == ord('q'):
        print("\n❌ Exiting without saving.\n")
        break

cap.release()
cv2.destroyAllWindows()
