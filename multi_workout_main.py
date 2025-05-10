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

    reps = 0
    similarity = 0.0
    smooth_similarity = []
    similarity_scores = []
    mistakes = set()
    last_feedback_time = 0
    cooldown = 2.5
    visibility_threshold = 0.6

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

        angle = calculate_angle_from_landmarks(landmarks_full, *joint_indices)
        deep_position = angle < thresholds['down']
        correct_posture = thresholds['back'][0] < calculate_angle_from_landmarks(landmarks_full, *thresholds['back'][1]) < thresholds['back'][1]

        if deep_position:
            similarity_now, is_correct = compare_pose(flat_landmarks, reference_pose, threshold=0.92)
            smooth_similarity.append(similarity_now)
            if len(smooth_similarity) > 5:
                smooth_similarity.pop(0)
            similarity = np.mean(smooth_similarity)
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
