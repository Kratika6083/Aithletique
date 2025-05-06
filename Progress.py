import streamlit as st
st.set_page_config(page_title="📊 Aithletique – Progress Tracker", layout="centered")

# Now safe to import everything else
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.logger import get_all_sessions
import pandas as pd
import matplotlib.pyplot as plt
import calplot
import altair as alt
from datetime import timedelta



# ---- Streamlit Setup ----

st.title("📊 My Progress")
st.markdown("Track your session history, performance, and feedback.")

# ---- Load Data ----
data = get_all_sessions()

if not data:
    st.info("No sessions logged yet. Start a workout or meditation session first.")
else:
    df = pd.DataFrame(data, columns=["ID", "Pose", "Reps", "Feedback", "Duration (sec)", "Date"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date", ascending=False)

    # ---- Feedback Summary Generator ----
    def summarize_feedback(raw_text):
        if not raw_text or pd.isna(raw_text):
            return "❔ No feedback available."
        parts = [p.strip() for p in raw_text.split(";") if p.strip()]
        good = [p for p in parts if "✅" in p or "Good posture" in p or "Soft breathing" in p]
        issues = [p for p in parts if "❌" in p or "Not breathing" in p or "Go deeper" in p or "Keep your head" in p]
        tips = [p for p in parts if "Try" in p or "Hold" in p or "Slow" in p]

        summary = ""
        if good:
            summary += f"✅ Good: {', '.join(good[:2])}. "
        if issues:
            summary += f"⚠️ Needs work: {', '.join(issues[:2])}. "
        if tips:
            summary += f"🔁 Tip: {', '.join(tips[:1])}."
        return summary or "😐 Neutral session."

    df["Feedback Summary"] = df["Feedback"].apply(summarize_feedback)

    # ---- Filters ----
    pose_options = df["Pose"].unique().tolist()
    selected_pose = st.selectbox("🧘 Filter by Pose:", ["All"] + pose_options)

    filtered_df = df.copy()
    if selected_pose != "All":
        filtered_df = df[df["Pose"] == selected_pose]

    recent_days = st.slider("📅 Show sessions from last X days:", 1, 30, 7)
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=recent_days)
    filtered_df = filtered_df[filtered_df["Date"] >= cutoff]

    # ---- Table View ----
    st.markdown("### 📋 Session Log")
    display_df = filtered_df[["Pose", "Reps", "Feedback Summary"]]
    st.dataframe(display_df, use_container_width=True)

    # ---- Metrics ----
    st.markdown("---")
    st.subheader("📈 Session Summary")

    total_sessions = len(filtered_df)
    total_duration = filtered_df["Duration (sec)"].sum()
    most_common_pose = filtered_df["Pose"].mode()[0] if total_sessions > 0 else "N/A"

    st.metric("Total Sessions", total_sessions)
    st.metric("Total Time (mins)", round(total_duration / 60, 1))
    st.metric("Most Frequent Pose", most_common_pose)

    # ---- CSV Export ----
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("📁 Download Report as CSV", data=csv, file_name="aithletique_progress.csv", mime='text/csv')

    # ---- Weekly Goal Tracker ----
    st.markdown("---")
    st.subheader("🎯 Weekly Goal Progress")

    weekly_goal = st.slider("Set your weekly session goal:", 1, 14, 5)
    last_7_days = pd.Timestamp.now() - pd.Timedelta(days=7)
    weekly_df = df[df["Date"] >= last_7_days]
    weekly_sessions = len(weekly_df)
    goal_percent = int((weekly_sessions / weekly_goal) * 100)
    goal_percent = min(goal_percent, 100)

    st.progress(goal_percent)
    st.write(f"✅ You've completed **{weekly_sessions}/{weekly_goal}** sessions this week!")

    if weekly_sessions >= weekly_goal:
        st.success("🏆 Goal achieved! Great work!")
    elif weekly_sessions == 0:
        st.warning("🚀 Let's get moving! Start a session today.")
    else:
        st.info("⏳ Keep going – you're getting closer!")

    # ---- Streak Tracker ----
    st.markdown("---")
    st.subheader("🔥 Streak Tracker")

    session_dates = pd.to_datetime(filtered_df["Date"].dt.date).sort_values().unique()

    current_streak = 0
    longest_streak = 0
    today = pd.Timestamp.today().normalize()
    yesterday = today - timedelta(days=1)

    previous_date = None
    for date in reversed(session_dates):
        if previous_date is None:
            if date in [today, yesterday]:
                current_streak = 1
            previous_date = date
        else:
            if (previous_date - date).days == 1:
                current_streak += 1
                previous_date = date
            elif (previous_date - date).days > 1:
                break

    streak = 1
    for i in range(1, len(session_dates)):
        if (session_dates[i] - session_dates[i - 1]).days == 1:
            streak += 1
        else:
            longest_streak = max(longest_streak, streak)
            streak = 1
    longest_streak = max(longest_streak, streak)

    st.metric("🔥 Current Streak (days)", current_streak)
    st.metric("🏅 Longest Streak", longest_streak)

    # ---- Calendar Heatmap ----
    st.markdown("---")
    st.subheader("📅 Calendar View")

    calendar_data = filtered_df.copy()
    calendar_data["date"] = pd.to_datetime(calendar_data["Date"].dt.date)
    calendar_counts = calendar_data.groupby("date").size()

    if not calendar_counts.empty:
        fig, ax = calplot.calplot(calendar_counts, cmap='YlGn', colorbar=True, suptitle='Your Activity Calendar')
        st.pyplot(fig)
    else:
        st.info("No session data available to show calendar heatmap.")

    # ---- Visual Charts ----
    st.markdown("---")
    st.subheader("📊 Visual Progress")

    chart_metric = st.radio("📈 Chart Metric:", ["Reps", "Duration"], horizontal=True)

    if len(filtered_df) >= 2:
        chart_data = filtered_df[["Date", "Pose", "Reps", "Duration (sec)"]].copy().sort_values("Date")

        y_field = "Reps" if chart_metric == "Reps" else "Duration (sec)"
        line_chart = alt.Chart(chart_data).mark_line(point=True).encode(
            x=alt.X("Date:T", title="Session Date"),
            y=alt.Y(f"{y_field}:Q", title=chart_metric),
            color="Pose:N",
            tooltip=["Date", "Pose", y_field]
        ).properties(height=300)

        st.altair_chart(line_chart, use_container_width=True)

        pose_counts = filtered_df["Pose"].value_counts().reset_index()
        pose_counts.columns = ["Pose", "Sessions"]

        st.markdown("### 🏋️ Session Count by Pose")
        bar_chart = alt.Chart(pose_counts).mark_bar().encode(
            x=alt.X("Pose:N"),
            y=alt.Y("Sessions:Q"),
            color="Pose:N"
        ).properties(height=250)

        st.altair_chart(bar_chart, use_container_width=True)
    else:
        st.info("📉 Not enough data to plot charts. Try logging more sessions.")

    # ---- Expandable Session Details ----
    st.markdown("---")
    st.subheader("📌 Session Details")

    for index, row in filtered_df.iterrows():
        with st.expander(f"📅 {row['Date'].strftime('%b %d, %Y')} – {row['Pose'].title()}"):
            st.markdown(f"**Reps:** {row['Reps']}")
            st.markdown(f"**Duration:** {round(row['Duration (sec)'], 1)} sec")
            st.markdown(f"**Feedback:**\n```\n{row['Feedback']}\n```")
