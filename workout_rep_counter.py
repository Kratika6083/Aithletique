import numpy as np

class WorkoutRepCounter:
    def __init__(self, joint_name, threshold_down=70, threshold_up=160):
        self.joint_name = joint_name
        self.threshold_down = threshold_down
        self.threshold_up = threshold_up
        self.in_down_position = False

    def update(self, angle):
        if angle < self.threshold_down:
            self.in_down_position = True
        elif angle > self.threshold_up and self.in_down_position:
            self.in_down_position = False
            return True  # rep completed
        return False
