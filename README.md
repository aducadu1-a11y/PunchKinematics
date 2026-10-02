# PunchKinematics

A small Streamlit app that takes a boxing video and measures what the arms are doing. It tracks the shoulder, elbow, and wrist in every frame, calculates the joint angles, counts punches, and flags when an elbow might be hyperextending.

## Why I built it

I'm the president of my university's boxing club. When we review technique we usually just watch phone footage and guess. I wanted a simple way to put numbers on it, especially for elbow extension, since snapping the elbow all the way straight on a punch is a common way people get hurt. This project is my attempt at a basic tool for reviewing technique and injury risk from a regular video.

## How it works

1. You upload an mp4 in the Streamlit app.
2. OpenCV reads the video one frame at a time.
3. YOLOv8-Pose (the `yolov8n-pose.pt` model from the ultralytics package) finds the person and returns body keypoints. I use the shoulder, elbow, wrist, and hip on each side.
4. NumPy calculates two angles per arm in each frame:
   - Elbow angle: the angle at the elbow between the shoulder and the wrist. 180 degrees means the arm is fully straight.
   - Shoulder angle: the angle at the shoulder between the hip and the elbow.
5. SciPy smooths the angles over time with a Savitzky-Golay filter (`savgol_filter`) so the curves are not jumpy.
6. SciPy's `find_peaks` finds the peaks in the elbow angle. Each peak is the arm going from bent to extended, so each peak counts as one punch.
7. The app shows the video with the arms drawn on, a line chart of the angles, the punch count, and the max elbow extension for each arm.
8. If the max elbow angle goes above a threshold (175 degrees by default, adjustable with a slider), the app shows a note that the elbow may be hyperextending.

Frames where no person or keypoints are detected are skipped.

## How to run it

You need Python 3.9 or newer.

```bash
git clone https://github.com/aducadu1-a11y/PunchKinematics.git
cd PunchKinematics
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at http://localhost:8501. The pose model downloads automatically the first time you run it.

## Files

- `app.py`: the Streamlit app
- `pose_utils.py`: keypoint detection, angle math, smoothing, and punch counting
- `requirements.txt`: packages needed

## Limitations

- It only uses 2D video, so the angles are what the camera sees, not the true 3D joint angles. A punch thrown toward or away from the camera will look more bent than it really is.
- Results depend a lot on camera angle. A side view of the punching arm works best.
- It only tracks one person. If two people are in frame (like sparring), it just picks the first one the model detects.
- Fast punches can blur at normal frame rates, which makes the keypoints less accurate.
- The punch counter is based on elbow extension only, so it can miss hooks and uppercuts where the arm stays bent.
- The hyperextension note is a rough flag for review, not a medical assessment.
