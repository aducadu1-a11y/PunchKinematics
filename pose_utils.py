# pose_utils.py
# Helper functions for finding arm keypoints and calculating joint angles.

import cv2
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from ultralytics import YOLO

# YOLOv8-Pose gives 17 keypoints per person (COCO format).
# These are the index numbers of the ones we need for each arm.
LEFT_ARM = {"shoulder": 5, "elbow": 7, "wrist": 9, "hip": 11}
RIGHT_ARM = {"shoulder": 6, "elbow": 8, "wrist": 10, "hip": 12}

# A keypoint is only trusted if the model is at least this confident about it
MIN_CONFIDENCE = 0.5


def load_model():
    # Loads the small pose model. It downloads itself the first time.
    return YOLO("yolov8n-pose.pt")


def get_keypoints(model, frame):
    # Run the pose model on one frame.
    # Returns (points, confidence) for the first person, or (None, None) if nobody is found.
    results = model(frame, verbose=False)
    keypoints = results[0].keypoints

    if keypoints is None or keypoints.conf is None or len(keypoints.xy) == 0:
        return None, None

    points = keypoints.xy[0].cpu().numpy()        # 17 rows of (x, y) pixel positions
    confidence = keypoints.conf[0].cpu().numpy()  # 17 confidence scores from 0 to 1
    return points, confidence


def calculate_angle(point_a, point_b, point_c):
    # Angle in degrees at point_b, between the lines b->a and b->c.
    # Example: shoulder, elbow, wrist gives the elbow angle (180 = straight arm).
    vector_1 = np.array(point_a) - np.array(point_b)
    vector_2 = np.array(point_c) - np.array(point_b)

    length_1 = np.linalg.norm(vector_1)
    length_2 = np.linalg.norm(vector_2)
    if length_1 == 0 or length_2 == 0:
        return np.nan

    cos_angle = np.dot(vector_1, vector_2) / (length_1 * length_2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # avoid rounding errors outside -1 to 1
    return np.degrees(np.arccos(cos_angle))


def keypoints_visible(confidence, indexes):
    # True if every keypoint in the list was detected with enough confidence
    return all(confidence[i] >= MIN_CONFIDENCE for i in indexes)


def get_arm_angles(points, confidence, arm):
    # Returns (elbow_angle, shoulder_angle) for one arm.
    # If the needed keypoints are not visible the angle is NaN (missing).
    shoulder, elbow, wrist, hip = arm["shoulder"], arm["elbow"], arm["wrist"], arm["hip"]

    elbow_angle = np.nan
    if keypoints_visible(confidence, [shoulder, elbow, wrist]):
        elbow_angle = calculate_angle(points[shoulder], points[elbow], points[wrist])

    shoulder_angle = np.nan
    if keypoints_visible(confidence, [hip, shoulder, elbow]):
        shoulder_angle = calculate_angle(points[hip], points[shoulder], points[elbow])

    return elbow_angle, shoulder_angle


def draw_arm(frame, points, confidence, arm, color):
    # Draw the shoulder-elbow-wrist lines and a dot on each joint
    joint_order = [arm["shoulder"], arm["elbow"], arm["wrist"]]

    # Lines between joints
    for start, end in zip(joint_order[:-1], joint_order[1:]):
        if keypoints_visible(confidence, [start, end]):
            start_xy = tuple(int(v) for v in points[start])
            end_xy = tuple(int(v) for v in points[end])
            cv2.line(frame, start_xy, end_xy, color, 3)

    # Dots on joints
    for joint in joint_order:
        if keypoints_visible(confidence, [joint]):
            joint_xy = tuple(int(v) for v in points[joint])
            cv2.circle(frame, joint_xy, 6, color, -1)


def smooth_angles(angles, window_length=11):
    # Fill in missing values, then smooth the curve with a Savitzky-Golay filter.
    angles = np.array(angles, dtype=float)
    is_missing = np.isnan(angles)

    # If the arm was never seen there is nothing to smooth
    if is_missing.all():
        return angles

    # Fill gaps by drawing a straight line between the nearest good values
    frame_numbers = np.arange(len(angles))
    angles[is_missing] = np.interp(frame_numbers[is_missing], frame_numbers[~is_missing], angles[~is_missing])

    # The filter window has to be odd and no longer than the data
    if len(angles) < window_length:
        window_length = len(angles) if len(angles) % 2 == 1 else len(angles) - 1
    if window_length < 5:
        return angles  # too short to smooth

    smoothed = savgol_filter(angles, window_length=window_length, polyorder=2)
    return np.clip(smoothed, 0, 180)  # smoothing can overshoot a little past 180


def find_punches(smoothed_elbow_angles, frames_per_second, min_extension_angle=140):
    # A punch shows up as a peak in the elbow angle (arm goes from bent to straight).
    # Returns the positions of those peaks.
    if np.isnan(smoothed_elbow_angles).all():
        return np.array([], dtype=int)

    peak_positions, _ = find_peaks(
        smoothed_elbow_angles,
        height=min_extension_angle,                    # arm must get fairly straight
        prominence=20,                                 # must open at least 20 degrees from bent
        distance=max(1, int(frames_per_second * 0.2)), # punches at least 0.2 s apart
    )
    return peak_positions
