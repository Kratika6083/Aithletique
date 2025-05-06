import cv2
import numpy as np
import os
import mediapipe as mp

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

def extract_landmarks(results):
    """Extract (x, y, z, visibility) landmarks from pose results."""
    landmarks = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.append((lm.x, lm.y, lm.z, lm.visibility))
    return landmarks

def main():
    # 📁 Ask for folder containing pose images
    folder_path = input("Enter the folder path containing pose images: ").strip()
    pose_name = input("Enter the pose name to save (example: Vriksasana): ").strip()

    if not os.path.exists(folder_path):
        print("❌ Folder does not exist!")
        return

    save_path = "pose_references"
    os.makedirs(save_path, exist_ok=True)

    all_landmarks = []

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} images.")

    for img_name in image_files:
        img_path = os.path.join(folder_path, img_name)
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"⚠️ Skipping unreadable image: {img_name}")
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Draw landmarks
        annotated_frame = frame.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # Show image with skeleton
        cv2.imshow("Detected Pose - Press Y to Save, N to Skip", annotated_frame)
        key = cv2.waitKey(0)

        if key == ord('y') or key == ord('Y'):
            landmarks = extract_landmarks(results)
            if len(landmarks) > 0:
                all_landmarks.append(landmarks)
                print(f"✅ Accepted: {img_name}")
            else:
                print(f"⚠️ No landmarks found in {img_name}.")
        else:
            print(f"❌ Skipped: {img_name}")

    cv2.destroyAllWindows()

    if len(all_landmarks) == 0:
        print("❌ No valid landmarks collected. Nothing saved.")
        return

    # Save all accepted landmarks to one npz file
    save_file = os.path.join(save_path, f"{pose_name}.npz")
    np.savez_compressed(save_file, landmarks=all_landmarks)
    print(f"✅ Saved {len(all_landmarks)} pose landmarks to {save_file}.")

if __name__ == "__main__":
    main()
