import streamlit as st
import os
from components.webcam_feed import run_pose_detection
from meditation.meditation_features import run_meditation_session
from components.workout_main import start_squat_workout
from components.multi_workout_main import (
    start_pushup_workout,
    start_plank_workout,
    start_pullup_workout
)

st.set_page_config(page_title="Aithletique", layout="centered")
st.title("🤸‍♂️ Aithletique – Real-Time Posture Coach")

# Session states
if "meditation_summary" not in st.session_state:
    st.session_state["meditation_summary"] = None
if "start_meditation" not in st.session_state:
    st.session_state["start_meditation"] = False
if "show_summary" not in st.session_state:
    st.session_state["show_summary"] = False
if "start_yoga_session" not in st.session_state:
    st.session_state["start_yoga_session"] = False

# Tabs for activity categories
tabs = st.tabs(["🧘 Yoga & Meditation", "🏋️ Workout & Training"])

# -------------------------
# Yoga & Meditation Tab
# -------------------------
with tabs[0]:
    pose_folder = "pose_references"
    npz_files = [f for f in os.listdir(pose_folder) if f.endswith(".npz")]
    yoga_poses = [os.path.splitext(f)[0].replace("_", " ").title() for f in npz_files]
    yoga_poses.append("Meditation")
    pose_type = st.selectbox("🎯 Choose Your Activity:", sorted(yoga_poses), key="yoga_select")

    st.markdown(f"📝 Live feedback for: **{pose_type}**")

    if pose_type.lower() == "meditation":
        duration = st.slider("⏳ Meditation Duration (minutes)", 1, 30, 5, key="med_dur")

        if st.button("🧘 Start Meditation"):
            st.session_state["start_meditation"] = True
            st.session_state["meditation_summary"] = None
            st.session_state["show_summary"] = False

        if st.session_state["start_meditation"]:
            result = run_meditation_session(duration_minutes=duration)
            st.session_state["meditation_summary"] = result
            st.session_state["show_summary"] = True
            st.session_state["start_meditation"] = False
            st.success("✅ Meditation session completed.")

        if st.session_state.get("show_summary") and st.session_state.get("meditation_summary"):
            st.markdown("### 🧘 Last Meditation Session Summary")
            for line in st.session_state["meditation_summary"]:
                st.markdown(f"- {line}")
        else:
            st.info("No meditation session summary available. Start a session to see results.")

    else:
        if st.button("🎥 Start Yoga Session", key="start_yoga"):
            run_pose_detection(pose_name=pose_type.lower(), category="Yoga & Meditation")

# -------------------------
# Workout & Training Tab
# -------------------------
with tabs[1]:
    pose_type = st.selectbox("🎯 Choose Your Activity:", ["Squat", "Pushup", "Plank", "Pull-up"], key="workout_select")

    st.markdown(f"📝 Live feedback for: **{pose_type}**")

    if st.button("🎥 Start Workout Session"):
        if pose_type.lower() == "squat":
            start_squat_workout()
        elif pose_type.lower() == "pushup":
            start_pushup_workout()
        elif pose_type.lower() == "plank":
            start_plank_workout()
        elif pose_type.lower() == "pull-up":
            start_pullup_workout()
        else:
            run_pose_detection(pose_name=pose_type.lower(), category="Workout & Training")
