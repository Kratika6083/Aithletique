# Home.py – Clean main entry with meditation features modularized
import streamlit as st
import os
from components.webcam_feed import run_pose_detection
from meditation.meditation_features import run_meditation_session

# -------------------- Streamlit Setup --------------------
st.set_page_config(page_title="Aithletique", layout="centered")
st.title("🤸‍♂️ Aithletique – Real-Time Posture Coach")
st.markdown("📹 Make sure you're clearly visible in the camera frame with good lighting.")

# -------------------- Category Selection --------------------
category = st.selectbox("🧘‍♂️ Select Your Training Category:", ["Yoga & Meditation", "Workout & Training"])
pose_type = None

# -------------------- Pose Dropdown --------------------
if category == "Yoga & Meditation":
    pose_folder = "pose_references"
    yoga_poses = [file.replace(".npz", "").replace("_", " ").title() for file in os.listdir(pose_folder) if file.endswith(".npz")]
    yoga_poses.append("Meditation")
    pose_type = st.selectbox("🎯 Choose Your Activity:", sorted(yoga_poses))

elif category == "Workout & Training":
    pose_type = st.selectbox("🎯 Choose Your Activity:", ["Squats", "Pushups", "Lunges", "Plank", "Situps"])

# -------------------- Run Yoga or Meditation --------------------
if pose_type:
    st.markdown(f"📝 You'll receive live posture feedback for: **{pose_type}**")

    if pose_type.lower() == "meditation":
        duration = st.slider("⏳ Set Meditation Duration (minutes)", 1, 30, 5)

        if st.button("🧘 Start Meditation"):
            # Reset session flags for voice feedback
            st.session_state.feedback_flags = {}
            st.session_state.last_reminder_time = {}
            run_meditation_session(duration)

    else:
        run_pose_detection(pose_name=pose_type.lower(), category=category)
