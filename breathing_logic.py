# backend/breathing/breathing_logic.py

import numpy as np
import time

class BreathingMonitor:
    def __init__(self):
        self.prev_y = None
        self.prev_time = time.time()
        self.breath_state = "Unknown"
        self.no_breath_counter = 0

    def detect_breathing(self, landmarks, frame_height):
        """
        Detect breathing based on chest movement using y-coordinate.
        We track the vertical (y) movement of chest midpoint.
        """
        if not landmarks or len(landmarks) < 33:
            return "Pose not stable"

        # Chest reference point: midpoint between shoulders
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        chest_y = ((left_shoulder[1] + right_shoulder[1]) / 2) * frame_height

        if self.prev_y is None:
            self.prev_y = chest_y
            return "Analyzing breath..."

        movement = chest_y - self.prev_y
        time_diff = time.time() - self.prev_time

        self.prev_y = chest_y
        self.prev_time = time.time()

        if abs(movement) < 0.5:
            self.no_breath_counter += 1
        else:
            self.no_breath_counter = 0

        if self.no_breath_counter > 25:
            self.breath_state = "❌ Not breathing!"
        elif movement > 1:
            self.breath_state = "⬆️ Inhale"
        elif movement < -1:
            self.breath_state = "⬇️ Exhale"
        else:
            self.breath_state = "😮‍💨 Soft breathing"

        return self.breath_state
