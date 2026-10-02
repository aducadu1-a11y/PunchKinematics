# app.py
# Streamlit app: upload a boxing video, track the arms, and review punch technique.
# Run with: streamlit run app.py

import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import pose_utils

LEFT_COLOR = (255, 128, 0)   # blue-ish (OpenCV uses BGR order)
RIGHT_COLOR = (0, 128, 255)  # orange


@st.cache_resource
def get_model():
    # Load the YOLO model once and reuse it
    return pose_utils.load_model()


@st.cache_data(show_spinner=False)
def process_video(video_bytes):
    # Reads the video frame by frame, gets the angles, and makes a video with the arms drawn on.
    # Cached so moving the sliders does not redo all of this work.
    model = get_model()

    # OpenCV needs a file on disk, so save the upload to a temporary file
    input_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_file.write(video_bytes)
    input_file.close()
    output_path = input_file.name.replace(".mp4", "_annotated.webm")

    video = cv2.VideoCapture(input_file.name)
    frames_per_second = video.get(cv2.CAP_PROP_FPS) or 30
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Write the output as webm (VP8) because browsers can play it
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"VP80"), frames_per_second, (width, height))

    # One list entry per frame where a person was found
    times = []
    left_elbow, left_shoulder = [], []
    right_elbow, right_shoulder = [], []

    frame_number = 0
    while True:
        frame_was_read, frame = video.read()
        if not frame_was_read:
            break  # end of video

        points, confidence = pose_utils.get_keypoints(model, frame)

        # Skip the measurements if nobody was detected in this frame
        if points is not None:
            left_elbow_angle, left_shoulder_angle = pose_utils.get_arm_angles(points, confidence, pose_utils.LEFT_ARM)
            right_elbow_angle, right_shoulder_angle = pose_utils.get_arm_angles(points, confidence, pose_utils.RIGHT_ARM)

            times.append(frame_number / frames_per_second)
            left_elbow.append(left_elbow_angle)
            left_shoulder.append(left_shoulder_angle)
            right_elbow.append(right_elbow_angle)
            right_shoulder.append(right_shoulder_angle)

            pose_utils.draw_arm(frame, points, confidence, pose_utils.LEFT_ARM, LEFT_COLOR)
            pose_utils.draw_arm(frame, points, confidence, pose_utils.RIGHT_ARM, RIGHT_COLOR)

        writer.write(frame)
        frame_number += 1

    video.release()
    writer.release()

    # Read the finished video back in so Streamlit can show it, then clean up
    annotated_video_bytes = b""
    if os.path.exists(output_path):
        with open(output_path, "rb") as f:
            annotated_video_bytes = f.read()
        os.remove(output_path)
    os.remove(input_file.name)

    angles = {
        "Left elbow": left_elbow,
        "Left shoulder": left_shoulder,
        "Right elbow": right_elbow,
        "Right shoulder": right_shoulder,
    }
    return times, angles, annotated_video_bytes, frames_per_second


# ---------- Page layout ----------

st.title("PunchKinematics")
st.write("Upload a boxing video to track arm angles, count punches, and check elbow extension.")

hyperextension_threshold = st.sidebar.slider("Hyperextension threshold (degrees)", 160, 180, 175)
min_extension_angle = st.sidebar.slider("Minimum elbow angle to count as a punch (degrees)", 100, 170, 140)

uploaded_video = st.file_uploader("Boxing video (mp4)", type=["mp4"])

if uploaded_video is None:
    st.info("Upload a video to get started. It works best with one person, filmed from the side.")
    st.stop()

with st.spinner("Analyzing video, this can take a minute..."):
    times, angles, annotated_video_bytes, frames_per_second = process_video(uploaded_video.getvalue())

if len(times) == 0:
    st.error("No person was detected in this video. Try a clearer clip with the whole upper body in frame.")
    st.stop()

# Smooth each angle over time so the chart and peaks are not jumpy
smoothed = {name: pose_utils.smooth_angles(values) for name, values in angles.items()}

# ---------- Video ----------

st.subheader("Video with keypoints")
st.video(annotated_video_bytes, format="video/webm")
st.caption("Left arm is blue, right arm is orange.")

# ---------- Angle chart ----------

st.subheader("Joint angles over time")
chart_data = pd.DataFrame(smoothed, index=pd.Index(times, name="Time (s)"))
st.line_chart(chart_data, y_label="Angle (degrees)")

# ---------- Punch count and max extension ----------

st.subheader("Punches")
left_column, right_column = st.columns(2)

for column, side in [(left_column, "Left"), (right_column, "Right")]:
    elbow_angles = smoothed[side + " elbow"]

    # This arm was never visible in the video
    if np.isnan(elbow_angles).all():
        column.metric(side + " punches", 0)
        column.write("Arm not visible.")
        continue

    punch_positions = pose_utils.find_punches(elbow_angles, frames_per_second, min_extension_angle)
    max_extension = float(np.max(elbow_angles))

    column.metric(side + " punches", len(punch_positions))
    column.metric(side + " max elbow extension", f"{max_extension:.0f}°")

    # Technique flag: elbow going past the threshold at full extension
    if max_extension > hyperextension_threshold:
        column.warning(
            f"{side} elbow went above {hyperextension_threshold}°. "
            "The elbow may be hyperextending, which is an injury risk."
        )
    else:
        column.success("No hyperextension flagged.")

total_frames_note = f"Person detected in {len(times)} frames."
st.caption(total_frames_note)
