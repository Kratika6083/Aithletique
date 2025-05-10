# mediapipe_model.py

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

        self.landmark_map = {
            11: 'LEFT_SHOULDER',
            12: 'RIGHT_SHOULDER',
            23: 'LEFT_HIP',
            24: 'RIGHT_HIP',
            25: 'LEFT_KNEE',
            26: 'RIGHT_KNEE',
            27: 'LEFT_ANKLE',
            28: 'RIGHT_ANKLE',
        }

    def detect_pose(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose.process(rgb_frame)

    def draw_landmarks(self, frame, results):
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        return frame

    def get_named_landmarks(self, results):
        if not results.pose_landmarks:
            return {}

        named_landmarks = {}
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            if idx in self.landmark_map:
                named_landmarks[self.landmark_map[idx]] = (lm.x, lm.y)
        return named_landmarks

    def get_landmarks(self, results):
        """
        Returns raw list of (x, y, z, visibility) for all landmarks.
        """
        if not results.pose_landmarks:
            return []

        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.append((lm.x, lm.y, lm.z, lm.visibility))
        return landmarks
