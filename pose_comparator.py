import numpy as np
import os
import math

# Important points for angle calculation
IMPORTANT_ANGLE_PAIRS = [
    (11, 13, 15),  # Right Shoulder-Elbow-Wrist
    (12, 14, 16),  # Left Shoulder-Elbow-Wrist
    (23, 25, 27),  # Right Hip-Knee-Ankle
    (24, 26, 28),  # Left Hip-Knee-Ankle
    (11, 23, 25),  # Right Shoulder-Hip-Knee
    (12, 24, 26),  # Left Shoulder-Hip-Knee
    (23, 11, 13),  # Hip-Shoulder-Elbow Right
    (24, 12, 14)   # Hip-Shoulder-Elbow Left
]

def calculate_angle(a, b, c):
    """Calculate angle (in degrees) between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ab = a - b
    cb = c - b

    cosine_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def extract_important_angles_safe(landmarks_list):
    """Extract joint angles for important body parts."""
    angles = []
    for (a, b, c) in IMPORTANT_ANGLE_PAIRS:
        if a < len(landmarks_list) and b < len(landmarks_list) and c < len(landmarks_list):
            a_point = landmarks_list[a][:3]  # Take only (x, y, z)
            b_point = landmarks_list[b][:3]
            c_point = landmarks_list[c][:3]
            angle = calculate_angle(a_point, b_point, c_point)
            angles.append(angle)
        else:
            angles.append(0)  # fallback dummy
    return np.array(angles)


def load_single_reference_landmarks(filepath):
    """Load only a single .npz pose file."""
    data = np.load(filepath)
    return data['landmarks']

def flip_left_right_angles(angles_array):
    """Swap left and right body parts in angle array for mirror comparison."""
    flip_indices = [1, 0, 3, 2, 5, 4, 7, 6]  # swap right and left angles
    flipped = angles_array.copy()
    for i, j in enumerate(flip_indices):
        flipped[i] = angles_array[j]
    return flipped

def compute_pose_accuracy(live_landmarks, reference_landmarks_list):
    """Compare joint angles, allowing mirror flip matching."""
    live_angles = extract_important_angles_safe(live_landmarks)
    live_flipped = flip_left_right_angles(live_angles)
    best_score = 0

    for ref_landmarks in reference_landmarks_list:
        ref_angles = extract_important_angles_safe(ref_landmarks)

        if live_angles.shape != ref_angles.shape:
            continue  # skip mismatch

        # Normal comparison
        diff_normal = np.abs(live_angles - ref_angles)
        accuracy_normal = max(0, 100 - np.mean(diff_normal))

        # Flipped comparison
        diff_flipped = np.abs(live_flipped - ref_angles)
        accuracy_flipped = max(0, 100 - np.mean(diff_flipped))

        # Take better
        accuracy = max(accuracy_normal, accuracy_flipped)

        if accuracy > best_score:
            best_score = accuracy

    return best_score

def check_enough_landmarks(landmarks_list, required_ids=[11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32]):
    """Check if enough important joints have high visibility."""
    visible_count = 0
    for idx in required_ids:
        if idx < len(landmarks_list):
            visibility = landmarks_list[idx][3]
            if visibility > 0.5:
                visible_count += 1
    return visible_count >= int(0.75 * len(required_ids))
