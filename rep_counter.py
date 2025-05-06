# backend/feedback_engine/rep_counter.py

from backend.feedback_engine.angles import calculate_angle, get_point_coords

class RepCounter:
    def __init__(self):
        self.stage = None  # "up" or "down"
        self.reps = 0

    def count_squat(self, landmarks, frame_width, frame_height):
        """
        Count squats by tracking knee angle motion.
        """
        hip = get_point_coords(landmarks, 24, frame_width, frame_height)
        knee = get_point_coords(landmarks, 26, frame_width, frame_height)
        ankle = get_point_coords(landmarks, 28, frame_width, frame_height)

        if hip and knee and ankle:
            angle = calculate_angle(hip, knee, ankle)

            # Logic to detect motion stages
            if angle < 90:
                self.stage = "down"
            if angle > 160 and self.stage == "down":
                self.stage = "up"
                self.reps += 1

        return self.reps

    def count_pushup(self, landmarks, frame_width, frame_height):
        """
        Count pushups by tracking elbow angle motion.
        """
        shoulder = get_point_coords(landmarks, 12, frame_width, frame_height)
        elbow = get_point_coords(landmarks, 14, frame_width, frame_height)
        wrist = get_point_coords(landmarks, 16, frame_width, frame_height)

        if shoulder and elbow and wrist:
            angle = calculate_angle(shoulder, elbow, wrist)

            if angle < 90:
                self.stage = "down"
            if angle > 160 and self.stage == "down":
                self.stage = "up"
                self.reps += 1

        return self.reps
