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
