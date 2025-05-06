# backend/feedback_engine/rules.py

# Angle rules for different body parts during exercises
POSTURE_RULES = {
    "squat": {
        "knee": {"min": 70, "max": 100, "message": "Go deeper in your squat."},
        "back": {"min": 160, "max": 180, "message": "Keep your back straighter."}
    },
    "pushup": {
        "elbow": {"min": 70, "max": 110, "message": "Bend your elbows more."},
        "back": {"min": 160, "max": 180, "message": "Maintain a straight back."}
    },
    "tree_pose": {
        "knee": {"min": 160, "max": 190, "message": "Keep your knee straighter."},
        "hip": {"min": 160, "max": 190, "message": "Keep hips level."}
    },
    "meditation": {
        "neck": {"min": 85, "max": 95, "message": "Keep your head upright."},
        "spine": {"min": 170, "max": 190, "message": "Sit up straight."}
    }
}

def check_joint_angle(pose_name, joint_name, angle):
    """
    Checks if the given angle is within the defined threshold range.
    
    Returns:
        - (True, None) if angle is good
        - (False, message) if angle is wrong
    """
    try:
        rules = POSTURE_RULES[pose_name][joint_name]
        if rules["min"] <= angle <= rules["max"]:
            return True, None
        else:
            return False, rules["message"]
    except KeyError:
        return True, None  # No rule defined
