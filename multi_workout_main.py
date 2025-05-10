import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
import streamlit as st
import time
from backend.pose_detection.mediapipe_model import PoseDetector
from backend.feedback_engine.workout_feedback import WorkoutFeedback
from backend.feedback_engine.workout_rep_counter import WorkoutRepCounter
from backend.feedback_engine.pose_similarity_checker import compare_pose
from backend.feedback_engine.angles import calculate_angle_from_landmarks

def run_workout(exercise_name, joint_indices, thresholds):
    stframe = st.empty()
    rep_placeholder = st.empty()
    similarity_placeholder = st.empty()
    message_placeholder = st.empty()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access the camera.")
        return

    stop_button = st.button("🛑 Stop Workout")

    model = PoseDetector()
    feedback = WorkoutFeedback()
    rep_counter = WorkoutRepCounter(exercise_name, threshold_down=thresholds['down'], threshold_up=thresholds['up'])

    reference_pose_path = f"pose_references/{exercise_name}_reference.npy"
    reference_pose = np.load(reference_pose_path)

    angle_reference_path = f"pose_references/{exercise_name}_angles_reference.npy"
    angle_reference_data = np.load(angle_reference_path, allow_pickle=True)

    reps = 0
    similarity = 0.0
    smooth_similarity = []
    similarity_scores = []
    mistakes = set()
    last_feedback_time = 0
    cooldown = 2.5
    visibility_threshold = 0.6

    feedback_rules = {
        "pushup": {
            "elbow": (11, 13, 15),
            "back": (11, 23, 24),
            "shoulder": (13, 11, 23)
        },
        "plank": {
            "shoulder_hip_knee": (11, 23, 25),
            "back": (11, 23, 24),
            "hip_knee_ankle": (23, 25, 27)
        },
        "pullup": {
            "elbow_shoulder_hip": (13, 11, 23),
            "back": (11, 23, 24),
            "shoulder_alignment": (12, 11, 23)
        }
    }

    def check_angles(landmarks, ref_angles):
        for label, (a, b, c) in feedback_rules[exercise_name].items():
            current_angle = calculate_angle_from_landmarks(landmarks, a, b, c)
            reference_angle = ref_angles.get(label, None)
            if reference_angle is not None and abs(current_angle - reference_angle) > 15:
                if label == "elbow":
                    feedback.give_feedback("Straighten your arms more")
                elif label == "back":
                    feedback.give_feedback("Keep your back straight")
                elif label == "shoulder_hip_knee":
                    feedback.give_feedback("Keep your hips aligned")
                elif label == "elbow_shoulder_hip":
                    feedback.give_feedback("Lift your body higher")
                elif label == "shoulder":
                    feedback.give_feedback("Bring shoulders forward")
                elif label == "hip_knee_ankle":
                    feedback.give_feedback("Align your knees with hips and ankles")
                elif label == "shoulder_alignment":
                    feedback.give_feedback("Adjust your shoulder posture")

    frame_index = 0

    while cap.isOpened():
        if stop_button:
            break

        ret, frame = cap.read()
        if not ret:
            break

        results = model.detect_pose(frame)
        landmarks_full = model.get_landmarks(results)

        if not landmarks_full or any(lm[3] < visibility_threshold for lm in landmarks_full):
            similarity = 0.0
            if time.time() - last_feedback_time > cooldown:
                feedback.give_feedback("pose not fully visible")
                last_feedback_time = time.time()
            message_placeholder.markdown("<span style='color:red'><b>⚠️ Pose not fully visible</b></span>", unsafe_allow_html=True)
            stframe.image(frame, channels="BGR")
            rep_placeholder.markdown(f"### 🏋️ Repetitions: **{reps}**")
            similarity_placeholder.progress(0, text=f"🎯 Accuracy: --")
            continue

        landmarks = [lm[:3] for lm in landmarks_full]
        flat_landmarks = np.array(landmarks).flatten()

        # Multiple angles could be used here to support richer feedback
        angle = calculate_angle_from_landmarks(landmarks_full, *joint_indices)
        deep_position = angle < thresholds['down']
        correct_posture = thresholds['back'][0] < calculate_angle_from_landmarks(landmarks_full, *thresholds['back'][1]) < thresholds['back'][1]

        if deep_position:
            similarity_now, is_correct = compare_pose(flat_landmarks, reference_pose, threshold=0.92)
            smooth_similarity.append(similarity_now)
            if len(smooth_similarity) > 5:
                smooth_similarity.pop(0)
            similarity = np.mean(smooth_similarity)

            if time.time() - last_feedback_time > cooldown:
                if frame_index < len(angle_reference_data):
                    ref_angles = angle_reference_data[frame_index].item()
                    check_angles(landmarks_full, ref_angles)
                    last_feedback_time = time.time()
        else:
            similarity = 0.0
            smooth_similarity.clear()

        if rep_counter.update(angle):
            reps += 1
            if smooth_similarity:
                similarity_scores.append(np.mean(smooth_similarity))
            smooth_similarity.clear()

        frame = model.draw_landmarks(frame, results)
        stframe.image(frame, channels="BGR")
        rep_placeholder.markdown(f"### 🏋️ Repetitions: **{reps}**")
        similarity_placeholder.progress(int(similarity * 100), text=f"🎯 Accuracy: {similarity * 100:.1f}%")
        frame_index += 1

    cap.release()
    st.success("Workout session ended.")
    st.markdown("---")
    st.markdown(f"## 🧾 {exercise_name.capitalize()} Session Summary")
    st.markdown(f"**Total Repetitions:** {reps}")
    if similarity_scores:
        st.markdown(f"**Best Accuracy:** {max(similarity_scores) * 100:.1f}%")
        st.markdown(f"**Average Accuracy:** {np.mean(similarity_scores) * 100:.1f}%")
    else:
        st.markdown("**Accuracy Data:** No valid reps captured")

def start_pushup_workout():
    run_workout(
        exercise_name="pushup",
        joint_indices=(11, 13, 15),
        thresholds={"down": 70, "up": 160, "back": (160, 195, (11, 23, 24))}
    )

def start_plank_workout():
    run_workout(
        exercise_name="plank",
        joint_indices=(11, 23, 25),
        thresholds={"down": 160, "up": 170, "back": (160, 195, (11, 23, 24))}
    )

def start_pullup_workout():
    run_workout(
        exercise_name="pullup",
        joint_indices=(13, 11, 23),
        thresholds={"down": 80, "up": 150, "back": (160, 195, (11, 23, 24))}
    )
