import cv2 
import tempfile
import streamlit as st
import time
import numpy as np
import os

from backend.pose_detection.mediapipe_model import PoseDetector
from backend.feedback_engine.pose_comparator import (
    load_single_reference_landmarks, 
    compute_pose_accuracy, 
    check_enough_landmarks
)
from backend.feedback_engine.motion_tools import load_motion_reference, compute_motion_similarity
from backend.voice.tts_engine import VoiceCoach
from database.logger import init_db, log_session

def run_pose_detection(pose_name="tadasana", category="Yoga & Meditation"):
    init_db()

    if "running" not in st.session_state:
        st.session_state.running = False
    if "start_time" not in st.session_state:
        st.session_state.start_time = None
    if "feedback_collected" not in st.session_state:
        st.session_state.feedback_collected = set()
    if "stop" not in st.session_state:
        st.session_state.stop = False
    if "reps" not in st.session_state:
        st.session_state.reps = 0
    if "pose_held" not in st.session_state:
        st.session_state.pose_held = False

    detector = PoseDetector()
    coach = VoiceCoach()

    reference_landmarks = None
    motion_reference = None

    if category == "Yoga & Meditation":
        try:
            corrected_pose_name = pose_name.capitalize()
            reference_landmarks = load_single_reference_landmarks(f"pose_references/{corrected_pose_name}.npz")
        except Exception as e:
            st.error(f"❌ Could not load reference for {pose_name}.")
            return
    elif category == "Workout & Training":
        motion_reference = load_motion_reference(f"motion_references/{pose_name}_motion.npz")
        if motion_reference is None:
            st.error(f"❌ No motion reference found for {pose_name}.")
            return

    st.session_state.running = True
    st.session_state.start_time = time.time()
    st.session_state.feedback_collected = set()
    st.session_state.stop = False
    st.session_state.reps = 0
    st.session_state.pose_held = False
    st.success("✅ Session started. Your form will now be monitored...")

    # Stop button rendered ONCE
    stop_button_placeholder = st.empty()

    if stop_button_placeholder.button("🔚 Stop Session", key=f"stop_button_once_{pose_name}"):
        st.session_state.stop = True

    process_camera(pose_name, detector, coach, reference_landmarks, motion_reference, stop_button_placeholder)

def process_camera(pose_name, detector, coach, reference_landmarks, motion_reference, stop_button_placeholder):
    camera = cv2.VideoCapture(0)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    stframe = st.empty()
    feedback_placeholder = st.empty()
    accuracy_display = st.empty()
    reps_display = st.empty()

    last_accuracy = 0
    motion_buffer = []

    while camera.isOpened():
        if st.session_state.stop:
            break

        ret, frame = camera.read()
        if not ret:
            st.error("❌ Camera error.")
            break

        frame = cv2.flip(frame, 1)
        results = detector.detect_pose(frame)
        frame = detector.draw_landmarks(frame, results)
        landmarks = detector.get_landmarks(results)

        if landmarks:
            if not check_enough_landmarks(landmarks):
                feedback_placeholder.warning("⚠️ Full body not detected. Please adjust your position!")
                accuracy_display.metric("🎯 Accuracy", "0%")
            else:
                if reference_landmarks is not None:
                    accuracy = compute_pose_accuracy(landmarks, reference_landmarks)
                    last_accuracy = (0.7 * last_accuracy) + (0.3 * accuracy)
                    accuracy_display.metric("🎯 Accuracy", f"{last_accuracy:.2f}%")

                    if last_accuracy >= 90:
                        feedback_placeholder.success("✅ Excellent posture!")
                    elif last_accuracy >= 75:
                        feedback_placeholder.warning("⚠️ Minor Adjustments Needed!")
                    else:
                        feedback_placeholder.error("❌ Major correction needed.")
                elif motion_reference is not None:
                    motion_buffer.append(landmarks)
                    if len(motion_buffer) > 10:
                        motion_buffer = motion_buffer[-10:]

                    accuracy = compute_motion_similarity(motion_buffer, motion_reference)
                    last_accuracy = (0.7 * last_accuracy) + (0.3 * accuracy)
                    accuracy_display.metric("🎯 Accuracy", f"{last_accuracy:.2f}%")
                    reps_display.metric("✅ Reps", st.session_state.reps)

                    if accuracy > 80 and not st.session_state.pose_held:
                        st.session_state.reps += 1
                        st.session_state.pose_held = True
                        coach.speak("✅ Great rep!")

                    if accuracy < 40:
                        st.session_state.pose_held = False

                    if accuracy < 60:
                        feedback_placeholder.markdown("### ⚠️ Adjust your form!")
                        coach.speak("Adjust your form!")
                    else:
                        feedback_placeholder.markdown("### ✅ Looking good!")
        else:
            feedback_placeholder.warning("⚠️ No pose detected.")

        cv2.imwrite(temp_file.name, frame)
        stframe.image(temp_file.name, channels="BGR", use_container_width=True)

        time.sleep(0.5)

    camera.release()
    duration = round(time.time() - st.session_state.start_time, 2)

    log_session(
        pose=pose_name,
        reps=st.session_state.reps,
        feedback_list=list(st.session_state.feedback_collected),
        duration=duration
    )

    st.session_state.running = False
    st.session_state.stop = False
    st.session_state.pose_held = False

    fb = list(st.session_state.feedback_collected)
    stframe.empty()
    feedback_placeholder.empty()
    accuracy_display.empty()
    reps_display.empty()
    stop_button_placeholder.empty()

    summary = f"✅ Session saved! Duration: {duration} sec | Reps: {st.session_state.reps} | Feedbacks: {len(fb)} types."
    st.success(summary)
