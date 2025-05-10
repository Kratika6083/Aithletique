import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def extract_average_pose(video_path, output_path):
    pose = mp_pose.Pose()
    cap = cv2.VideoCapture(video_path)
    landmarks_list = []
    frame_count = 0
    valid_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            valid_frames += 1
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            landmarks_list.append(landmarks)

        # Display the frame
        cv2.imshow('Processing Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    pose.close()
    cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames, valid pose frames: {valid_frames}")

    if valid_frames == 0:
        print("❌ No valid pose landmarks detected. Check your video quality and subject visibility.")
        return

    avg_landmarks = np.mean(np.array(landmarks_list), axis=0).flatten()
    np.save(output_path, avg_landmarks)
    print(f"✅ Saved reference pose to {output_path}")

# Example usage
extract_average_pose("workout_videos/Squat_correct.mp4", "pose_references/squat_reference.npy")
