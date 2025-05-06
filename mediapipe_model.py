# backend/pose_detection/mediapipe_model.py

import cv2
import mediapipe as mp

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def detect_pose(self, frame):
        """
        Processes the input frame and returns the detection results.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results

    def draw_landmarks(self, frame, results):
        """
        Draws the pose landmarks on the frame.
        """
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        return frame

    def get_landmarks(self, results):
        """
        Extracts landmark coordinates if available.
        Returns a list of (x, y, z, visibility) tuples.
        """
        if not results.pose_landmarks:
            return []
        
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append((lm.x, lm.y, lm.z, lm.visibility))
        return landmarks
