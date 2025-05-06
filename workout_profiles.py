# workout_profiles.py

# Each workout defines ideal angles for joints at the bottom position.

workout_profiles = {
    "squats": {
        "target_joints": {
            "left_knee": (85, 100),  # Ideal squat depth
            "right_knee": (85, 100),
            "left_hip": (80, 100),
            "right_hip": (80, 100),
        },
        "form_tips": [
            ("left_knee", "Push your knees outward!"),
            ("right_knee", "Push your knees outward!"),
            ("left_hip", "Lower your hips more!"),
            ("right_hip", "Lower your hips more!"),
        ],
        "motion_joint": "left_knee"  # To detect up/down motion
    },

    "pushups": {
        "target_joints": {
            "left_elbow": (70, 100),
            "right_elbow": (70, 100),
        },
        "form_tips": [
            ("left_elbow", "Bend your elbows properly!"),
            ("right_elbow", "Bend your elbows properly!"),
        ],
        "motion_joint": "left_elbow"
    },

    "lunges": {
        "target_joints": {
            "left_knee": (80, 100),
            "right_knee": (80, 100),
        },
        "form_tips": [
            ("left_knee", "Bend your front knee to 90 degrees!"),
            ("right_knee", "Drop your back knee lower!"),
        ],
        "motion_joint": "left_knee"
    },

    "situps": {
        "target_joints": {
            "left_hip": (60, 90),
            "right_hip": (60, 90),
        },
        "form_tips": [
            ("left_hip", "Engage your core more!"),
            ("right_hip", "Engage your core more!"),
        ],
        "motion_joint": "left_hip"
    },

    "plank": {
        "target_joints": {
            "left_shoulder": (80, 100),
            "right_shoulder": (80, 100),
            "left_hip": (160, 180),
            "right_hip": (160, 180),
        },
        "form_tips": [
            ("left_hip", "Keep your hips level!"),
            ("right_hip", "Keep your hips level!"),
        ],
        "motion_joint": None  # Static hold, no motion
    }
}

# Later you can easily add more like: wall sit, pullups, hip thrusts, deadlifts, etc.
