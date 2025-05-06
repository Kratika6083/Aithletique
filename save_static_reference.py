# save_static_reference.py
import cv2
import numpy as np
import os
from backend.pose_detection.mediapipe_model import PoseDetector

save_folder = "pose_references"
os.makedirs(save_folder, exist_ok=True)

detector = PoseDetector()

cap = cv2.VideoCapture(0)
print("\n📸 Capturing from Webcam... Press 's' to Save Frame, 'q' to Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    results = detector.detect_pose(frame)
    frame = detector.draw_landmarks(frame, results)

    cv2.imshow('Pose Capture', frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        landmarks = detector.get_landmarks(results)
        if landmarks:
            pose_name = input("💾 Enter pose name (e.g., tadasana): ").strip().lower()
            np.savez(os.path.join(save_folder, f"{pose_name}.npz"), landmarks=landmarks)
            print(f"✅ Saved static pose for '{pose_name}'")
        else:
            print("⚠️ No landmarks detected. Try again.")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
