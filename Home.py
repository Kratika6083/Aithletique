import streamlit as st
import os
from components.webcam_feed import run_pose_detection
from meditation.meditation_features import run_meditation_session

st.set_page_config(page_title="Aithletique", layout="centered")
st.title("🤸‍♂️ Aithletique – Real-Time Posture Coach")
st.markdown("📹 Make sure you're clearly visible in the camera frame with good lighting.")

category = st.selectbox("🧘‍♂️ Select Your Training Category:", ["Yoga & Meditation", "Workout & Training"])
pose_type = None

if category == "Yoga & Meditation":
    pose_folder = "pose_references"
    yoga_poses = [file.replace(".npz", "").replace("_", " ").title() for file in os.listdir(pose_folder) if file.endswith(".npz")]
    yoga_poses.append("Meditation")
    pose_type = st.selectbox("🎯 Choose Your Activity:", sorted(yoga_poses))
elif category == "Workout & Training":
    pose_type = st.selectbox("🎯 Choose Your Activity:", ["Squats", "Pushups", "Lunges", "Plank", "Situps"])

if pose_type:
    st.markdown(f"📝 You'll receive live posture feedback for: **{pose_type}**")

    if pose_type.lower() == "meditation":
        duration = st.slider("⏳ Set Meditation Duration (minutes)", 1, 30, 5)

        if "meditation_summary" not in st.session_state:
            st.session_state.meditation_summary = None

        if st.button("🧘 Start Meditation", key="start_meditation"):
            result = run_meditation_session(duration_minutes=duration)
            if result:
                st.session_state.meditation_summary = result

        if st.session_state.get("meditation_summary"):
            st.markdown("### 🧘 Last Session Summary")
            for msg in st.session_state["meditation_summary"]:
                st.markdown(f"- {msg}")

    else:
        run_pose_detection(pose_name=pose_type.lower(), category=category)