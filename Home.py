
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

if "meditation_summary" not in st.session_state:
    st.session_state["meditation_summary"] = None
if "start_meditation" not in st.session_state:
    st.session_state["start_meditation"] = False
if "show_summary" not in st.session_state:
    st.session_state["show_summary"] = False

# Category selection
category = st.selectbox("🧘‍♂️ Select Your Training Category:", ["Yoga & Meditation", "Workout & Training"])
pose_type = None

# Pose options
if category == "Yoga & Meditation":
    pose_folder = "pose_references"
    npz_files = [f for f in os.listdir(pose_folder) if f.endswith(".npz")]
    yoga_poses = [os.path.splitext(f)[0].replace("_", " ").title() for f in npz_files]
    yoga_poses.append("Meditation")
    pose_type = st.selectbox("🎯 Choose Your Activity:", sorted(yoga_poses))
else:
    pose_type = st.selectbox("🎯 Choose Your Activity:", ["Squat", "Pushup", "Lunge", "Plank", "Pull-up"])

# Show feedback target
if pose_type:
    st.markdown(f"📝 Live feedback for: **{pose_type}**")

# Meditation session
if pose_type and pose_type.lower() == "meditation":
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

# Yoga/workout session
elif pose_type:
    if st.button("🎥 Start Session"):
        if category == "Workout & Training":
            if pose_type.lower() == "squat":
                start_squat_workout()
            elif pose_type.lower() == "pushup":
                start_pushup_workout()
            elif pose_type.lower() == "plank":
                start_plank_workout()
            elif pose_type.lower() == "pull-up":
                start_pullup_workout()
            else:
                run_pose_detection(pose_name=pose_type.lower(), category=category)
        else:
            run_pose_detection(pose_name=pose_type.lower(), category=category)

# Show summary after meditation
if pose_type and pose_type.lower() == "meditation":
    if st.session_state.get("show_summary") and st.session_state.get("meditation_summary"):
        st.markdown("### 🧘 Last Meditation Session Summary")
        for line in st.session_state["meditation_summary"]:
            st.markdown(f"- {line}")
    else:
        st.info("No meditation session summary available. Start a session to see results.")
