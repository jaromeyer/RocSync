import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm

from rocsync.printer import *
from rocsync.video_statistics import VideoStatistics
from rocsync.vision import process_frame


def read_frames_async(cap, frame_queue):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame = cap.read()
        if not ret:
            frame_queue.put((None, None))
            break
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
        frame_queue.put((frame, frame_number))


def export_frame_async(frame_queue, y_pred, path):
    frame, frame_number = frame_queue.get()  # blocking wait
    if frame is None:
        errprint("Error: Input stream ended unexpectedly.")
        return
    cv2.imwrite(f"{path}/f{frame_number}_s{y_pred[frame_number]:.0f}.png", frame)


def export_frames(video_path, output_path, y_pred):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        errprint(f"Error: Could not open video: {video_path}")
        return
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(output_path, exist_ok=True)

    # Read frames in separate thread
    frame_queue = queue.Queue(maxsize=100)
    thread = threading.Thread(target=read_frames_async, args=(cap, frame_queue))
    thread.daemon = True
    thread.start()

    # Export frames concurrently using multiple threads
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = []
        for _ in range(n_frames):
            futures.append(executor.submit(export_frame_async, frame_queue, y_pred, output_path))
        for future in tqdm(
            as_completed(futures), total=n_frames, desc="Exporting frames", position=1
        ):
            future.result()
    cap.release()


def process_video(video_path, camera_type, export_dir=None, stride=None, debug_dir=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        errprint(f"Error: Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    expected_duration = (n_frames - 1) / fps * 1000

    # Read frames in separate thread
    frame_queue = queue.Queue(maxsize=100)
    thread = threading.Thread(target=read_frames_async, args=(cap, frame_queue))
    thread.daemon = True
    thread.start()

    # Analyze frames
    timestamps = {}
    scan_window = 0
    if stride is None:
        stride = int(fps)
    for frame_number in tqdm(range(n_frames), desc="Analyzing frames", position=1):
        frame, frame_number = frame_queue.get()  # blocking wait
        if frame is None:
            errprint("Error: Input stream ended unexpectedly. Could be a sign of skipped frames.")
            break
        if scan_window > 0 or frame_number % stride == 0:
            timestamp = process_frame(frame, camera_type, frame_number, debug_dir)
            scan_window -= 1
            if timestamp is not None:
                timestamps[frame_number] = timestamp
                scan_window = 5
    cap.release()

    if len(timestamps) == 0:
        errprint("Error: Unable to timestamp any frames.")
        return

    # Assuming constant frame rate, fit robust linear model
    x = np.array(list(timestamps.keys())).reshape(-1, 1)
    y = np.array([start for start, _ in timestamps.values()])
    model = RANSACRegressor(residual_threshold=1000 / fps)  # max one frame deviation
    model.fit(x, y)

    # Predict timestamps for all frames
    x_range = np.arange(0, n_frames).reshape(-1, 1)
    y_pred = model.predict(x_range)

    # errors = np.abs(y - model.predict(x))
    # e_max = np.max(errors)
    # e_max_frame = list(timestamps.keys())[np.argmax(errors)]
    # print(f"Max error: {np.max(errors)} (frame {e_max_frame})")

    # Remove outliers
    filtered_timestamps = {
        k: v for i, (k, v) in enumerate(timestamps.items()) if model.inlier_mask_[i]
    }
    filtered_x = np.array(list(filtered_timestamps.keys())).reshape(-1, 1)
    filtered_y = np.array([start for start, _ in filtered_timestamps.values()])

    # Calculate statistics
    exposure_times = [end - start for start, end in filtered_timestamps.values()]
    measured_duration = y_pred[-1] - y_pred[0]
    statistics = VideoStatistics(
        n_frames=n_frames,
        n_considered_frames=len(filtered_timestamps),
        n_rejected_frames=len(timestamps) - len(filtered_timestamps),
        r2_before=model.score(x, y),
        rmse_before=root_mean_squared_error(y, model.predict(x)),
        r2_after=model.score(filtered_x, filtered_y),
        rmse_after=root_mean_squared_error(filtered_y, model.predict(filtered_x)),
        expected_duration=expected_duration,
        measured_duration=measured_duration,
        expected_fps=fps,
        measured_fps=(n_frames - 1) / measured_duration * 1000,
        speed_factor=measured_duration / expected_duration,
        first_frame=y_pred[0],
        last_frame=y_pred[-1],
        mean_exposure_time=np.mean(exposure_times),
        min_exposure_time=np.min(exposure_times),
        max_exposure_time=np.max(exposure_times),
        std_exposure_time=np.std(exposure_times),
        measured_timestamps=filtered_timestamps,
        interpolated_timestamps=y_pred.tolist(),
    )

    print_statistics(statistics)

    if debug_dir:
        plot_timechart(
            filtered_x, filtered_y, x_range, y_pred, exposure_times, expected_duration, debug_dir
        )
        plot_exposure_histogram(exposure_times, debug_dir)

    if export_dir:
        export_frames(video_path, export_dir, y_pred)

    return statistics


def print_statistics(statistics: VideoStatistics):
    format_str = "{:<40} {:>30}"
    print(61 * "-")
    # TODO: find proper thesholds
    printresult(
        "Number of considered frames",
        statistics.n_considered_frames,
        statistics.n_considered_frames > 10,
    )
    printresult(
        "Number of rejected outliers",
        statistics.n_rejected_frames,
        statistics.n_rejected_frames == 0,
    )
    printresult(
        "R2 score (before/after outlier rejection)",
        f"{statistics.r2_before:.4f}/{statistics.r2_after:.4f}",
        statistics.r2_after > 0.99,
    )
    printresult(
        "RMSE (before/after outlier rejection)",
        f"{statistics.rmse_before:.2f}/{statistics.rmse_after:.2f} ms",
        statistics.rmse_after < 2,
    )
    print(61 * "-")
    print(format_str.format("First frame:", f"{statistics.first_frame:.1f} ms"))
    print(format_str.format("Last frame:", f"{statistics.last_frame:.1f} ms"))
    print(
        format_str.format(
            "Expected duration (fps):",
            f"{statistics.expected_duration:.1f} ms ({statistics.expected_fps:.2f} fps)",
        )
    )
    print(
        format_str.format(
            "Measured duration (fps):",
            f"{statistics.measured_duration:.1f} ms ({statistics.measured_fps:.2f} fps)",
        )
    )
    print(
        format_str.format(
            "Delta (measured - expected)",
            f"{statistics.measured_duration-statistics.expected_duration:.2f} ms ({statistics.speed_factor:.6f}x speed)",
        )
    )
    print(
        format_str.format(
            "Exposure time (mean/min/max/std):",
            f"{statistics.mean_exposure_time:.2f}/{statistics.min_exposure_time:.2f}/{statistics.max_exposure_time:.2f}/{statistics.std_exposure_time:.2f} ms",
        )
    )
    print(61 * "-")


def plot_timechart(x, y, x_range, y_pred, exposure_times, expected_duration, debug_dir):
    plt.figure()
    plt.scatter(x, y, color="blue", label="Measurements")
    plt.plot(x_range, y_pred, color="blue", label="Measured frametime")
    plt.plot(
        x_range,
        np.linspace(y_pred[0], y_pred[0] + expected_duration, len(x_range)),
        color="red",
        label="Calculated frametime",
    )
    plt.xlabel("Frame number")
    plt.ylabel("Time relative to RocSync [ms]")
    plt.title("Frame timing")
    plt.gca().ticklabel_format(style="plain", useOffset=False)
    plt.legend()
    plt.grid(True)
    ax2 = plt.gca().twinx()
    ax2.scatter(x, exposure_times, color="green", label="Exposure time [ms]")
    ax2.set_ylabel("Exposure time [ms]")
    ax2.ticklabel_format(style="plain", useOffset=False)
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.legend(loc="upper right")
    plt.savefig(f"{debug_dir}/timestamps.png")


def plot_exposure_histogram(exposure_times, debug_dir):
    plt.figure()
    unique_values, counts = np.unique(exposure_times, return_counts=True)
    bar = plt.bar(unique_values, counts)
    plt.bar_label(bar, counts)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Exposure time [ms]")
    plt.ylabel("Number of measured frames")
    plt.title("Exposure time histogram")
    plt.savefig(f"{debug_dir}/exposure.png")
