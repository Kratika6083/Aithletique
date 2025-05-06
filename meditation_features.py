# meditation/meditation_features.py
import cv2
import os
import time
import numpy as np
import pyttsx3
import threading
import mediapipe as mp
import streamlit as st
from scipy.signal import savgol_filter
from backend.pose_detection.mediapipe_model import PoseDetector
from database.logger import init_db, log_session
import json
import uuid
import random
import string

# --- Initialization ---
engine = pyttsx3.init()
voice_lock = threading.Lock()

# Thresholds (tweak as needed)
SPINE_THRESH = 0.06      # allowed shoulder-to-nose vertical difference
EYE_THRESH = 0.2        # EAR threshold for closed eyes
STRESS_THRESH = 0.04    # brow distance threshold for stress
REMINDER_INTERVAL = 2   # seconds between repeated voice reminders

# Ensure DB is ready
init_db()

# Per-category last-reminder times
if "last_reminder" not in st.session_state:
    st.session_state["last_reminder"] = {}

# Generate a unique Streamlit widget key
def get_unique_key(prefix="key"):
    suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    return f"{prefix}_{suffix}"

# Threaded TTS with rate limiting
def speak_feedback(text, category):
    now = time.time()
    last = st.session_state["last_reminder"].get(category, 0)
    if now - last > REMINDER_INTERVAL:
        def _speak():
            with voice_lock:
                engine.say(text)
                engine.runAndWait()
        threading.Thread(target=_speak).start()
        st.session_state["last_reminder"][category] = now

# Main meditation loop
def run_meditation_session(duration_minutes):
    # Initialize session state for reminder flags if not already set
    if "last_reminder_flags" not in st.session_state:
        st.session_state.last_reminder_flags = {
            "eyes": False,
            "stress": False,
            "spine": False,
            "head": False,
            "breathing": False
        }

    stframe = st.empty()
    feedback_box = st.empty()
    chart_box = st.empty()
    stop_placeholder = st.empty()

    # Generate a unique key for the stop button and store it in session state
    if "stop_btn_key" not in st.session_state:
        st.session_state.stop_btn_key = f"stop_btn_{uuid.uuid4().hex}"

    stop_btn_key = st.session_state.stop_btn_key

    pose_detector = PoseDetector()
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

    cap = cv2.VideoCapture(0)

    start_time = time.time()
    chest_movements = []
    eyes_closed_frames = 0
    total_frames = 0
    stress_frames = 0
    spine_issues = 0
    head_alignment_issues = 0
    calm_frames = 0
    upright_frames = 0
    feedback_msgs = []
    improvement_tips = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detector.detect_pose(frame)
        landmarks = pose_detector.get_landmarks(results)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        face_results = face_mesh.process(rgb)
        total_frames += 1

        if face_results.multi_face_landmarks:
            face_lms = face_results.multi_face_landmarks[0].landmark

            # Eye Aspect Ratio (EAR)
            def get_ear(eye):
                A = np.linalg.norm(np.array([eye[1].x, eye[1].y]) - np.array([eye[5].x, eye[5].y]))
                B = np.linalg.norm(np.array([eye[2].x, eye[2].y]) - np.array([eye[4].x, eye[4].y]))
                C = np.linalg.norm(np.array([eye[0].x, eye[0].y]) - np.array([eye[3].x, eye[3].y]))
                return (A + B) / (2.0 * C)

            left_eye = [face_lms[i] for i in [362, 385, 387, 263, 373, 380]]
            right_eye = [face_lms[i] for i in [33, 160, 158, 133, 153, 144]]
            ear = (get_ear(left_eye) + get_ear(right_eye)) / 2.0

            if ear < 0.2:
                eyes_closed_frames += 1
                st.session_state.last_reminder_flags["eyes"] = False
            else:
                if not st.session_state.last_reminder_flags["eyes"]:
                    speak_feedback("Please keep your eyes closed.", "eyes")

            # Brow distance for stress
            brow_dist = np.linalg.norm(np.array([face_lms[70].x, face_lms[70].y]) - np.array([face_lms[300].x, face_lms[300].y]))
            if brow_dist < 0.04:
                stress_frames += 1
                if not st.session_state.last_reminder_flags["stress"]:
                    speak_feedback("Relax your face and mind.", "stress")
            else:
                st.session_state.last_reminder_flags["stress"] = False

        feedback = ""
        if landmarks and len(landmarks) > 24:
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            nose = landmarks[0]

            # Check spine alignment
            mid_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
            chest_y = (left_hip[1] + right_hip[1]) / 2
            chest_movements.append(chest_y)

            spine_ok = abs(mid_shoulder_y - nose[1]) > 0.03 and abs(left_shoulder[1] - right_shoulder[1]) < 0.04
            if spine_ok:
                upright_frames += 1
                st.session_state.last_reminder_flags["spine"] = False
            else:
                spine_issues += 1
                if not st.session_state.last_reminder_flags["spine"]:
                    speak_feedback("Straighten your spine.", "spine")

            # Check head alignment
            head_ok = abs(nose[1] - mid_shoulder_y) < 0.05
            if head_ok:
                st.session_state.last_reminder_flags["head"] = False
            else:
                head_alignment_issues += 1
                if not st.session_state.last_reminder_flags.get("head", False):
                    speak_feedback("Keep your head aligned with your spine.", "head")

            # Breathing analysis
            if len(chest_movements) > 20:
                smoothed = savgol_filter(chest_movements[-20:], 9, 3)
                motion_range = max(smoothed) - min(smoothed)
                chart_box.line_chart(smoothed, height=100)

                if motion_range < 0.004:
                    feedback = "🟥 Not breathing"
                    if not st.session_state.last_reminder_flags["breathing"]:
                        speak_feedback("You're not breathing. Gently inhale and exhale.", "breathing")
                elif motion_range < 0.012:
                    feedback = "🟩 Calm breathing"
                    calm_frames += 1
                    st.session_state.last_reminder_flags["breathing"] = False
                else:
                    feedback = "🟧 Harsh breathing"
                    if not st.session_state.last_reminder_flags["breathing"]:
                        speak_feedback("You're breathing harshly. Try to breathe calmly.", "breathing")

                if not spine_ok:
                    feedback += "; 🧍 Sit up straight"

        if feedback:
            feedback_box.markdown(f"### 💬 Breathing: {feedback}")

        stframe.image(frame, channels="BGR")

        # Add a unique key to the stop button
        if stop_placeholder.button("⏹️ Stop Meditation", key=stop_btn_key) or time.time() - start_time > duration_minutes * 60:
            break

    cap.release()

    # Compute session metrics
    eye_closure_ratio = (eyes_closed_frames / total_frames * 100) if total_frames else 0
    upright_ratio = (upright_frames / total_frames * 100) if total_frames else 0
    head_alignment_ratio = (total_frames - head_alignment_issues) / total_frames * 100 if total_frames else 0
    stress_level = (stress_frames / total_frames * 100) if total_frames else 0
    calmness = (calm_frames / total_frames * 100) if total_frames else 0
    breath_score = min(100, max(20, calmness))

    feedback_msgs.append(f"🫁 Calm Breathing Score: {int(breath_score)} / 100")
    feedback_msgs.append(f"👁️ Eyes Closed: {int(eye_closure_ratio)}% of time")
    feedback_msgs.append(f"🧠 Stress Level: {int(stress_level)}% tense")
    feedback_msgs.append(f"📏 Upright Posture: {int(upright_ratio)}% of time")
    feedback_msgs.append(f"🧍 Head Alignment: {int(head_alignment_ratio)}% of time")

    if eye_closure_ratio < 60:
        improvement_tips.append("Try to keep your eyes closed throughout.")
    if breath_score < 50:
        improvement_tips.append("Work on calming your breath.")
    if stress_level > 40:
        improvement_tips.append("Relax your face and mind.")
    if upright_ratio < 70:
        improvement_tips.append("Sit upright with your spine relaxed and straight.")
    if head_alignment_ratio < 70:
        improvement_tips.append("Keep your head aligned with your spine.")

    st.markdown("### 🧘 Session Summary")
    for msg in feedback_msgs + improvement_tips:
        st.markdown(f"- {msg}")

    summary = {
        "duration_seconds": round(time.time() - start_time, 2),
        "breath_score": int(breath_score),
        "eye_closure_percent": int(eye_closure_ratio),
        "stress_percent": int(stress_level),
        "upright_posture_percent": int(upright_ratio),
        "head_alignment_percent": int(head_alignment_ratio),
        "feedback": feedback_msgs + improvement_tips
    }

    st.download_button("⬇️ Download Session Report", json.dumps(summary, indent=2), file_name="meditation_summary.json")

    log_session(
        pose="meditation",
        reps=0,
        feedback_list=summary["feedback"],
        duration=summary["duration_seconds"]
    )
