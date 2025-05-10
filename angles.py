# backend/feedback_engine/angles.py

import numpy as np

def calculate_angle(a, b, c):
    """
    Calculates the angle (in degrees) between three points.
    Each point is in (x, y) format.
    
    a = First point (e.g., shoulder)
    b = Mid point (e.g., elbow)
    c = End point (e.g., wrist)
    
    Returns:
        Angle in degrees
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Calculate the vectors
    ba = a - b
    bc = c - b

    # Cosine formula
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)


def get_point_coords(landmarks, index, frame_width, frame_height):
    """
    Convert MediaPipe's normalized coordinates to pixel values
    """
    if index >= len(landmarks):
        return None

    x = int(landmarks[index][0] * frame_width)
    y = int(landmarks[index][1] * frame_height)
    return (x, y)
def calculate_angle_from_landmarks(landmarks, a_index, b_index, c_index):
    a = np.array(landmarks[a_index])
    b = np.array(landmarks[b_index])
    c = np.array(landmarks[c_index])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle
