# meditation_refined.py (final working version)
import cv2
import time
import numpy as np
import threading
import streamlit as st
import mediapipe as mp
import uuid
import json
import os
from scipy.signal import savgol_filter
from scipy.spatial.distance import cosine
from gtts import gTTS
from playsound import playsound
from backend.pose_detection.mediapipe_model import PoseDetector
from database.logger import log_session

class VoiceFeedbackManager:
    def __init__(self, interval=6):
        self.interval = interval
        self.last_spoken = {}
        self.audio_dir = "voice_feedback"
        os.makedirs(self.audio_dir, exist_ok=True)
        self.queue = threading.Lock()

    def speak(self, tag, message):
        now = time.time()
        if tag not in self.last_spoken or now - self.last_spoken[tag] > self.interval:
            self.last_spoken[tag] = now
            threading.Thread(target=self._speak_thread, args=(tag, message), daemon=True).start()

    def _speak_thread(self, tag, message):
        with self.queue:
            filename = os.path.join(self.audio_dir, f"{tag}.mp3")
            if not os.path.exists(filename):
                tts = gTTS(text=message)
                tts.save(filename)
            try:
                playsound(filename)
            except Exception as e:
                print(f"Error playing sound: {e}")

voice_manager = VoiceFeedbackManager()
try:
    reference_pose = np.load("reference_meditation_pose.npz")['mean_pose']
except:
    reference_pose = None
    print("❌ Reference meditation pose file not found. Run the extractor script first.")

SIMILARITY_THRESHOLD = 0.75

def run_meditation_session(duration_minutes):
    st.title("🧘 Meditation Session")
    st.markdown("Live feedback will be provided. Ensure good lighting.")

    stframe = st.empty()
    chart_box = st.empty()
    feedback_box = st.empty()
    stop_button = st.button("⏹️ Stop Meditation")
    metrics_placeholder = st.empty()

    if "meditation_running" not in st.session_state:
        st.session_state.meditation_running = True
    if "alert_shown" not in st.session_state:
        st.session_state.alert_shown = False

    if stop_button:
        st.session_state.meditation_running = False

    duration = duration_minutes * 60
    start_time = time.time()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not access webcam.")
        return

    detector = PoseDetector()
    mp_draw = mp.solutions.drawing_utils
    face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    mp_pose = mp.solutions.pose

    chest_movements = []
    total_frames = eyes_open = calm_frames = 0
    usable_frames = head_issues = shoulder_issues = incorrect_posture = 0
    last_feedback = {"breathing": 0, "eyes": 0, "posture": 0, "head": 0, "shoulders": 0}
    breathing_scores = []

    try:
        while time.time() - start_time < duration and st.session_state.meditation_running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from camera")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.detect_pose(frame)
            face_results = face_mesh.process(rgb)
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = detector.get_landmarks(results)
            total_frames += 1

            eyes_closed = False
            posture_correct = False
            head_aligned = True
            shoulders_level = True
            now = time.time()
            breathing_check_allowed = False

            if face_results.multi_face_landmarks:
                face_lms = face_results.multi_face_landmarks[0].landmark
                def get_ear(eye):
                    A = np.linalg.norm(np.array([eye[1].x, eye[1].y]) - np.array([eye[5].x, eye[5].y]))
                    B = np.linalg.norm(np.array([eye[2].x, eye[2].y]) - np.array([eye[4].x, eye[4].y]))
                    C = np.linalg.norm(np.array([eye[0].x, eye[0].y]) - np.array([eye[3].x, eye[3].y]))
                    return (A + B) / (2.0 * C)
                left_eye = [face_lms[i] for i in [362, 385, 387, 263, 373, 380]]
                right_eye = [face_lms[i] for i in [33, 160, 158, 133, 153, 144]]
                ear = (get_ear(left_eye) + get_ear(right_eye)) / 2.0
                if ear > 0.25:
                    eyes_open += 1
                    if now - last_feedback["eyes"] > 6:
                        voice_manager.speak("eyes", "Please close your eyes to begin meditation.")
                        last_feedback["eyes"] = now
                else:
                    eyes_closed = True

            if landmarks and reference_pose is not None:
                flat = np.array(landmarks).flatten()
                sim = 1 - cosine(flat, reference_pose)
                if sim < SIMILARITY_THRESHOLD:
                    posture_correct = False
                else:
                    posture_correct = True

                left_shoulder, right_shoulder = landmarks[11], landmarks[12]
                nose = landmarks[0]
                mid_shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2
                if abs(nose[0] - mid_shoulder_x) > 0.05:
                    head_aligned = False
                if abs(left_shoulder[1] - right_shoulder[1]) > 0.03:
                    shoulders_level = False

                if not posture_correct and now - last_feedback["posture"] > 6:
                    voice_manager.speak("posture", "Straighten your posture and relax your body.")
                    last_feedback["posture"] = now
                if not head_aligned and now - last_feedback["head"] > 6:
                    voice_manager.speak("head", "Keep your head aligned and straight.")
                    last_feedback["head"] = now
                if not shoulders_level and now - last_feedback["shoulders"] > 6:
                    voice_manager.speak("shoulders", "Keep your shoulders level and relaxed.")
                    last_feedback["shoulders"] = now

            if not eyes_closed or not posture_correct or not head_aligned or not shoulders_level:
                if not st.session_state.alert_shown:
                    feedback_box.markdown("### ⏳ Waiting for correct posture and eyes closed to begin.")
                    st.session_state.alert_shown = True
                stframe.image(frame, channels="BGR", use_container_width=True)
                time.sleep(0.05)
                continue

            if st.session_state.alert_shown:
                feedback_box.markdown("")
                st.session_state.alert_shown = False

            usable_frames += 1

            if not posture_correct:
                incorrect_posture += 1
            if not head_aligned:
                head_issues += 1
            if not shoulders_level:
                shoulder_issues += 1

            if landmarks and len(landmarks) > 24:
                left_shoulder, right_shoulder = landmarks[11], landmarks[12]
                chest_y = (left_shoulder[1] + right_shoulder[1]) / 2
                chest_movements.append(chest_y)

                if len(chest_movements) > 30:
                    smoothed = savgol_filter(chest_movements[-30:], 11, 3)
                    motion_range = max(smoothed) - min(smoothed)
                    chart_box.line_chart(smoothed, height=100)

                    feedback = ""
                    if motion_range < 0.001:
                        feedback = "🟥 Not Breathing"
                        breathing_scores.append(0)
                        if now - last_feedback["breathing"] > 6:
                            voice_manager.speak("breathing_none", "You're not breathing. Inhale and exhale gently.")
                            last_feedback["breathing"] = now
                    elif motion_range < 0.01:
                        calm_frames += 1
                        feedback = "🟩 Calm Breathing"
                        breathing_scores.append(1)
                    else:
                        feedback = "🟧 Harsh Breathing"
                        breathing_scores.append(0.5)
                        if now - last_feedback["breathing"] > 6:
                            voice_manager.speak("breathing_harsh", "You're breathing harshly. Try to breathe calmly.")
                            last_feedback["breathing"] = now

                    if feedback:
                        feedback_box.markdown(f"### 💬 {feedback}")

            if usable_frames > 0:
                pose_accuracy = 100 - (incorrect_posture / usable_frames * 100)
                head_ratio = 100 - (head_issues / usable_frames * 100)
                shoulder_ratio = 100 - (shoulder_issues / usable_frames * 100)
                breath_score = int((sum(breathing_scores) / len(breathing_scores)) * 100) if breathing_scores else 0
                eye_ratio = 100 - (eyes_open / total_frames * 100)
                metrics_placeholder.markdown(f"""
                #### 📊 Live Metrics
                - 👁️ Eyes Closed: {eye_ratio:.1f}%
                - 🧘 Posture Accuracy: {pose_accuracy:.1f}%
                - 🧍 Head Alignment: {head_ratio:.1f}%
                - 💪 Shoulder Balance: {shoulder_ratio:.1f}%
                - 🫁 Breathing Score: {breath_score}
                """)

            stframe.image(frame, channels="BGR", use_container_width=True)
            time.sleep(0.05)

    finally:
        cap.release()
        face_mesh.close()

        if usable_frames == 0:
            st.warning("Session ended without valid meditation posture. No summary to show.")
            return

        pose_accuracy = 100 - (incorrect_posture / usable_frames * 100)
        head_ratio = 100 - (head_issues / usable_frames * 100)
        shoulder_ratio = 100 - (shoulder_issues / usable_frames * 100)
        breath_score = int((sum(breathing_scores) / len(breathing_scores)) * 100) if breathing_scores else 0
        eye_ratio = 100 - (eyes_open / total_frames * 100)

        feedback_msgs = [
            f"🧘 Pose Match Accuracy: {int(pose_accuracy)}%",
            f"👁️ Eyes Closed: {int(eye_ratio)}% of time",
            f"🧍 Head Alignment: {int(head_ratio)}% of time",
            f"💪 Shoulder Balance: {int(shoulder_ratio)}% of time",
            f"🫁 Calm Breathing Score: {int(breath_score)} / 100"
        ]

        improvement_tips = []
        if pose_accuracy < 75:
            improvement_tips.append("Improve your posture alignment.")
        if eye_ratio < 70:
            improvement_tips.append("Try to keep your eyes closed more consistently.")
        if head_ratio < 80:
            improvement_tips.append("Keep your head upright and straight.")
        if shoulder_ratio < 80:
            improvement_tips.append("Keep your shoulders relaxed and at equal level.")
        if breath_score < 50:
            improvement_tips.append("Focus on slow, steady breathing.")

        st.markdown("### 🧘 Session Summary")
        for msg in feedback_msgs + improvement_tips:
            st.markdown(f"- {msg}")

        summary = {
            "duration_seconds": round(time.time() - start_time, 2),
            "pose_accuracy": int(pose_accuracy),
            "eye_closure_percent": int(eye_ratio),
            "head_alignment_percent": int(head_ratio),
            "shoulder_balance_percent": int(shoulder_ratio),
            "breath_score": int(breath_score),
            "feedback": feedback_msgs + improvement_tips
        }

        st.download_button("⬇️ Download Session Report", json.dumps(summary, indent=2), file_name="meditation_summary.json")
        log_session(pose="meditation", reps=0, feedback_list=summary["feedback"], duration=summary["duration_seconds"])
        st.session_state["meditation_summary"] = feedback_msgs + improvement_tips
        return feedback_msgs + improvement_tips