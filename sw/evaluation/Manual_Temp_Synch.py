import os
import json
import cv2
import numpy as np
from typing import Dict, List, Tuple
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt

def parse_time_to_ms(folder_name: str) -> int:
    start_time_str = folder_name.split('-')[0]
    h, m, s, micros = map(int, start_time_str.split('_'))
    return ((h * 3600 + m * 60 + s) * 1000 + micros/1000)

def extract_camera_info(json_path: str) -> Dict[str, Dict[str, float]]:
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def compute_y_cam_i_values(
    cam_info: Dict[str, Dict[str, float]], 
    t_folder_ms: int, 
    x_user: int
) -> list[float]:
    y_values = []
    for _, info in cam_info.items():
        t_first = info['first_frame']
        fps = info['measured_fps']
        t_last = info['last_frame']
        n_frames = info['n_frames']
        slope = (t_last-t_first) /n_frames
        x = int(round(1/slope * (t_folder_ms - t_first) + x_user)) # get the frame number that is used as a correspondence by the aligned videos
        y = (1000/fps* x + t_first) # get the aligned timestamp value at the given frame number
        y_values.append(y)
    return y_values

def detect_light_switch_frames(
    video_path: str, 
    time_window: Tuple[int, int],  
    output_dir: str, 
    pre_frames: int = 5, 
    post_frames: int = 5,
    light_switch: bool = True
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(time_window[0] * fps / 1000)
    end_frame = int(time_window[1] * fps / 1000)

    brightness = []
    frames = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for i in range(start_frame, end_frame): #extract the brightness values
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness.append(np.mean(gray))
        frames.append((i, frame))

    if light_switch: #find maximum brightness increase/decrease
        diffs = np.diff(brightness)
        max_change_idx = np.argmax(np.abs(diffs))
    else: # use the center of the time window
        max_change_idx = int(round(1/2*(end_frame-start_frame)))
        print(max_change_idx)

    event_frame_idx = start_frame + max_change_idx + 1

    os.makedirs(output_dir, exist_ok=True)
    for i in range(-pre_frames, post_frames + 1): #extract frames
        idx = max_change_idx + i
        if 0 <= idx < len(frames):
            frame_number, frame = frames[idx]
            frame_name = os.path.join(output_dir, f"frame_{frame_number}.png")
            cv2.imwrite(frame_name, frame)

    cap.release()
    return event_frame_idx

def robust_regression(
    x_canon1: int, 
    x_canon2: int, 
    ys: List[float],
    fps : float,
    expected_duration: int,
    n_frames: int
) -> Tuple[float, float]:
    half_len = len(ys) // 2
    xs = np.concatenate((x_canon1 * np.ones((half_len,1)), x_canon2 * np.ones(((len(ys) - half_len),1))))
    model = RANSACRegressor(
        residual_threshold=1000 / fps,  # max one frame deviation
        max_trials=1000,  # more trials for more consistent results
        random_state=0,  # deterministic results
    )
    model.fit(xs, ys)
    x_range = np.arange(0, n_frames).reshape(-1, 1)
    y_pred = model.predict(x_range)
    measured_duration = y_pred[-1] - y_pred[0]
    first_frame = y_pred[0]
    last_frame = y_pred[-1]
    speed_factor = measured_duration / expected_duration
    slope = (y_pred[-1] - y_pred[0]) / n_frames

    # Plotting xs vs ys
    plt.figure(figsize=(6, 4))
    plt.scatter(xs, ys, label='Data points')
    plt.plot(x_range, y_pred, color='red', label='Robust regression fit')
    plt.xlabel("xs")
    plt.ylabel("ys")
    plt.title("Robust Regression")
    plt.legend()
    plt.grid(True)
    plt.show()

    return first_frame, speed_factor, slope, last_frame

# Semi-Automatic Manual Synchronization starts here:
folder1 = "00_03_22_718000-00_03_32_718000" # first correspondence
t_ms1 = parse_time_to_ms(folder1) # get aligned timestamp value in ms of the start of the clip 
cam_info = extract_camera_info("calibration/time_synchronization_RecD.json") # adjust to have the correct time_synchronization file
y_values_switch1 = compute_y_cam_i_values(cam_info, t_ms1, x_user=116) # for each camera it evaluates the model y = mx+b
folder2 = "00_59_17_773000-00_59_27_773000" # analogously for second correspondence
t_ms2 = parse_time_to_ms(folder2)
y_values_switch2 = compute_y_cam_i_values(cam_info, t_ms2, x_user=134)
camera_name = "canon2_closer_to_OR_door_RecD"
detect_light_switch_frames(f"RecordingD/{camera_name}.mp4", (454000, 458000), f"output_frames_{camera_name}_switch1") # detect light switches inside given time window, or takes the middle of the time window and extracts pre and post frames if it's a movement
detect_light_switch_frames(f"RecordingD/{camera_name}.mp4", (3808000, 3812000), f"output_frames_{camera_name}_switch2")
cap = cv2.VideoCapture(f"RecordingD/{camera_name}.mp4")
fps = cap.get(cv2.CAP_PROP_FPS) 
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
expected_duration = (n_frames - 1) / fps * 1000
cap.release()
x1_cam = 11399 # after you run the script for the first time and identified the frame index that should be aligned, adjust the values here
x2_cam = 95246
y_values = np.concatenate((y_values_switch1, y_values_switch2))
first_frame, speed_factor, slope, last_frame = robust_regression(x1_cam, x2_cam, y_values, fps, expected_duration, n_frames)
print(speed_factor) # all the stats that should be written to the time_synchronization.json file
print(first_frame)
print(n_frames)
print(fps)
print(slope)
print(last_frame)

