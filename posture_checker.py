# backend/feedback_engine/posture_checker.py

from backend.feedback_engine.angles import calculate_angle, get_point_coords
from backend.feedback_engine.rules import check_joint_angle

def check_posture(pose_name, landmarks, frame_width, frame_height):
    """
    Analyze posture by checking angles for a specific pose/exercise.

    Returns:
        List of feedback messages
    """
    feedback_msgs = []

    if not landmarks or len(landmarks) < 33:
        return ["Pose not fully visible. Step back or adjust camera."]

    # Define common joints to check per pose
    joint_angles = {}

    # Squat
    if pose_name == "squat":
        hip = get_point_coords(landmarks, 24, frame_width, frame_height)
        knee = get_point_coords(landmarks, 26, frame_width, frame_height)
        ankle = get_point_coords(landmarks, 28, frame_width, frame_height)
        if hip and knee and ankle:
            joint_angles["knee"] = calculate_angle(hip, knee, ankle)

        shoulder = get_point_coords(landmarks, 12, frame_width, frame_height)
        if shoulder and hip and ankle:
            joint_angles["back"] = calculate_angle(shoulder, hip, ankle)

    # Pushup
    elif pose_name == "pushup":
        shoulder = get_point_coords(landmarks, 12, frame_width, frame_height)
        elbow = get_point_coords(landmarks, 14, frame_width, frame_height)
        wrist = get_point_coords(landmarks, 16, frame_width, frame_height)
        if shoulder and elbow and wrist:
            joint_angles["elbow"] = calculate_angle(shoulder, elbow, wrist)

        hip = get_point_coords(landmarks, 24, frame_width, frame_height)
        ankle = get_point_coords(landmarks, 28, frame_width, frame_height)
        if shoulder and hip and ankle:
            joint_angles["back"] = calculate_angle(shoulder, hip, ankle)

    # Meditation
    elif pose_name == "meditation":
        shoulder = get_point_coords(landmarks, 12, frame_width, frame_height)
        ear = get_point_coords(landmarks, 8, frame_width, frame_height)
        if shoulder and ear:
            joint_angles["neck"] = calculate_angle((ear[0], shoulder[1]), shoulder, ear)

        hip = get_point_coords(landmarks, 24, frame_width, frame_height)
        ankle = get_point_coords(landmarks, 28, frame_width, frame_height)
        if shoulder and hip and ankle:
            joint_angles["spine"] = calculate_angle(shoulder, hip, ankle)

    # ✅ Tree Pose (fixed)
    elif pose_name == "tree_pose":
        shoulder = get_point_coords(landmarks, 12, frame_width, frame_height)
        hip = get_point_coords(landmarks, 24, frame_width, frame_height)
        knee = get_point_coords(landmarks, 26, frame_width, frame_height)
        ankle = get_point_coords(landmarks, 28, frame_width, frame_height)

        if hip and knee and ankle:
            joint_angles["knee"] = calculate_angle(hip, knee, ankle)

        if shoulder and hip and knee:
            joint_angles["hip"] = calculate_angle(shoulder, hip, knee)

    # Compare angles against rules
    for joint, angle in joint_angles.items():
        valid, msg = check_joint_angle(pose_name, joint, angle)
        if not valid and msg:
            feedback_msgs.append(msg)

    if not feedback_msgs:
        feedback_msgs.append("✅ Good posture!")

    return feedback_msgs
