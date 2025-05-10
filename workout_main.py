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

def start_squat_workout():
    stframe = st.empty()
    rep_placeholder = st.empty()
    similarity_placeholder = st.empty()
    message_placeholder = st.empty()

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access the camera.")
        return

    stop_button = st.button("🛑 Stop Workout")

    # Initialize modules
    model = PoseDetector()
    feedback = WorkoutFeedback()
    rep_counter = WorkoutRepCounter("knee", threshold_down=70, threshold_up=160)

    reference_pose_path = "pose_references/squat_reference.npy"
    reference_pose = np.load(reference_pose_path)

    reps = 0
    similarity = 0.0
    avg_similarity = 0.0
    smooth_similarity = []
    squat_position_held = False
    frame_feedback_given = False
    visibility_threshold = 0.6
    similarity_scores = []
    mistakes = set()

    last_feedback_time = 0
    cooldown = 2.5  # seconds between feedback
    in_squat = False

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
            similarity_placeholder.markdown("### 🎯 Accuracy: --")
            continue

        landmarks = [lm[:3] for lm in landmarks_full]
        flat_landmarks = np.array(landmarks).flatten()

        # Angles
        knee_angle = calculate_angle_from_landmarks(landmarks_full, 23, 25, 27)
        back_angle = calculate_angle_from_landmarks(landmarks_full, 11, 23, 24)
        hip_angle = calculate_angle_from_landmarks(landmarks_full, 11, 23, 25)
        leg_gap = abs(landmarks[23][0] - landmarks[24][0])

        deep_squat = knee_angle < 100
        upright_back = 165 < back_angle < 195
        leg_gap_ok = 0.2 < leg_gap < 0.5

        if deep_squat:
            similarity_now, is_correct = compare_pose(flat_landmarks, reference_pose, threshold=0.92)
            smooth_similarity.append(similarity_now)
            if len(smooth_similarity) > 5:
                smooth_similarity.pop(0)
            similarity = np.mean(smooth_similarity)
            similarity_scores.append(similarity)
            message_placeholder.empty()
            in_squat = True
        else:
            similarity = 0.0
            smooth_similarity.clear()
            in_squat = False


        if deep_squat:
            similarity_scores.append(similarity)
            message_placeholder.empty()
            in_squat = True
        else:
            in_squat = False

        if deep_squat and upright_back and leg_gap_ok:
            if not squat_position_held:
                squat_position_held = True
                if time.time() - last_feedback_time > cooldown:
                    feedback.give_feedback("correct")
                    last_feedback_time = time.time()
        else:
            squat_position_held = False
            if not frame_feedback_given:
                if not deep_squat:
                    mistakes.add("Not bending knees enough")
                    if time.time() - last_feedback_time > cooldown:
                        feedback.give_feedback("bend knees more")
                        last_feedback_time = time.time()
                elif not upright_back:
                    mistakes.add("Back posture not straight")
                    if time.time() - last_feedback_time > cooldown:
                        feedback.give_feedback("keep spine straight")
                        last_feedback_time = time.time()
                elif not leg_gap_ok:
                    mistakes.add("Legs not properly apart")
                    if time.time() - last_feedback_time > cooldown:
                        feedback.give_feedback("adjust leg position")
                        last_feedback_time = time.time()
                frame_feedback_given = True

        # Count rep only on return from deep squat
        if rep_counter.update(knee_angle):
            reps += 1
            if smooth_similarity:
                rep_accuracy = np.mean(smooth_similarity)
                similarity_scores.append(rep_accuracy)
            smooth_similarity.clear()


        frame_feedback_given = False
        frame = model.draw_landmarks(frame, results)

        stframe.image(frame, channels="BGR")
        rep_placeholder.markdown(f"### 🏋️ Repetitions: **{reps}**")
        similarity_placeholder.progress(int(similarity * 100), text=f"🎯 Accuracy: {similarity * 100:.1f}%")




    cap.release()
    st.success("Workout session ended.")

    st.markdown("---")
    st.markdown("## 🧾 Session Summary")
    st.markdown(f"**Total Repetitions:** {reps}")
    if similarity_scores:
        st.markdown(f"**Best Accuracy:** {max(similarity_scores) * 100:.1f}%")
        st.markdown(f"**Average Accuracy:** {np.mean(similarity_scores) * 100:.1f}%")

    else:
        st.markdown("**Accuracy Data:** No valid squat captured")

    if mistakes:
        st.markdown("**What to Improve:**")
        for m in mistakes:
            st.markdown(f"- ❌ {m}")
    else:
        st.markdown("**Great job! ✅ Your form was consistent.")
