import os
import cv2
import mediapipe as mp
import pandas as pd

# CONFIGURATION
DATASET_DIR = "Dataset"   # Folder where your Yoga Pose folders exist
OUTPUT_CSV = "pose_landmarks.csv"
IMG_SIZE = (480, 640)  # Match with your cleaned dataset size

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2)

all_landmarks = []

def extract_landmarks_from_folder(folder_path, label):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        img = cv2.imread(file_path)
        if img is None:
            print(f"⚠️ Skipping unreadable file: {file_path}")
            continue

        img = cv2.resize(img, IMG_SIZE)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)

        if result.pose_landmarks:
            landmarks = []
            for lm in result.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
            landmarks.append(label)
            all_landmarks.append(landmarks)
        else:
            print(f"⚠️ No pose detected: {file_path}")

if __name__ == "__main__":
    for pose_folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, pose_folder)
        if os.path.isdir(folder_path):
            print(f"\n🔍 Processing folder: {pose_folder}")
            extract_landmarks_from_folder(folder_path, pose_folder)

    if all_landmarks:
        num_landmarks = len(all_landmarks[0]) - 1
        columns = [f"x{i//4}_coord" if i % 4 == 0 else
                   f"y{i//4}_coord" if i % 4 == 1 else
                   f"z{i//4}_coord" if i % 4 == 2 else
                   f"v{i//4}" for i in range(num_landmarks)]
        columns.append("label")

        df = pd.DataFrame(all_landmarks, columns=columns)
        df.to_csv(OUTPUT_CSV, index=False)

        print(f"\n✅ Saved {len(df)} poses to: {OUTPUT_CSV}")
    else:
        print("❌ No landmarks extracted!")
