import math
import os
import queue
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

from rocsync.printer import errprint, warnprint
from rocsync.regression import fit_timestamps
from rocsync.video_statistics import VideoMetadata
from rocsync.vision import CameraType, process_frame


def read_frames_async(cap, frame_queue, start_frame=0, end_frame=None):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    while True:
        ret, frame = cap.read()
        if not ret:
            frame_queue.put((None, None))
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)
        if end_frame is not None and frame_number >= end_frame:
            frame_queue.put((None, None))
            break

        frame_queue.put((frame, frame_number))


def export_frame_async(frame_queue, y_pred, path):
    frame, frame_number = frame_queue.get()  # blocking wait
    if frame is None:
        errprint("Error: Input stream ended unexpectedly.")
        return
    cv2.imwrite(f"{path}/f{frame_number}_s{y_pred[frame_number]:.0f}.png", frame)


def export_frames(video_path, output_path, y_pred):
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    # cap.set(cv2.CAP_PROP_FFMPEG_HWACCEL, cv2.CAP_FFMPEG_HWACCEL_NVDEC)  # try to use
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
            futures.append(
                executor.submit(export_frame_async, frame_queue, y_pred, output_path)
            )
        for future in tqdm(
            as_completed(futures), total=n_frames, desc="Exporting frames", position=1
        ):
            future.result()
    cap.release()


def process_video_window(
    video_path: str,
    camera_type: CameraType,
    window_start: int,
    window_end: int,
    stride=None,
    debug_dir: str = None,
    brightness_boost: int = None,
):
    cap = cv2.VideoCapture(video_path)

    # Extract video metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(max(0, math.floor(window_start * fps)))
    end_frame = int(
        min(math.ceil(window_end * fps) + 1, cap.get(cv2.CAP_PROP_FRAME_COUNT))
    )  # end_frame is exclusive

    # Read frames in separate thread
    frame_queue = queue.Queue(maxsize=100)
    thread = threading.Thread(
        target=read_frames_async, args=(cap, frame_queue, start_frame, end_frame)
    )
    thread.daemon = False
    thread.start()

    timestamps = {}
    scan_window = 0
    if stride is None:
        stride = int(fps)

    pbar = tqdm(
        range(start_frame, end_frame),
        desc=f"Analyzing frames in time window [{window_start:.3f}s, {window_end:.3f}s] --> Found {len(timestamps)} timestamps",
        position=1,
    )
    for _ in pbar:
        frame, frame_number = frame_queue.get()  # blocking wait
        if frame is None:
            errprint(
                "Error: Input stream ended unexpectedly. Could be a sign of skipped frames."
            )
            break
        if scan_window > 0 or frame_number % stride == 0:
            rocsync_detected, timestamp = process_frame(
                frame, camera_type, frame_number, debug_dir, brightness_boost
            )
            scan_window -= 1
            if timestamp is not None:
                timestamps[frame_number] = timestamp
            if rocsync_detected:
                scan_window = 5
                pbar.set_description(
                    f"Analyzing frames in time window [{window_start:.3f}s, {window_end:.3f}s] --> Found {len(timestamps)} timestamps"
                )

    thread.join()
    cap.release()

    return timestamps


def process_video(
    video_path,
    camera_type,
    export_dir=None,
    stride=None,
    debug_dir=None,
    window1_start=None,
    window1_end=None,
    window2_start=None,
    window2_end=None,
    brightness_boost=None,
):
    # Get video metadata
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    # cap.set(cv2.CAP_PROP_FFMPEG_HWACCEL, cv2.CAP_FFMPEG_HWACCEL_NVDEC)  # try to use

    if not cap.isOpened():
        errprint(f"Error: Could not open video: {video_path}")
        return

    video = VideoMetadata(
        path=video_path,
        fps=cap.get(cv2.CAP_PROP_FPS),
        duration_ms=(cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
        / cap.get(cv2.CAP_PROP_FPS)
        * 1000,
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()

    if window1_start is None:
        window1_start = 0
    elif window1_start < 0:
        window1_start = max(0, (video.duration_ms / 1000) + window1_start)

    if window1_end is None:
        window1_end = video.duration_ms / 1000
    elif window1_end < 0:
        window1_end = max(0, (video.duration_ms / 1000) + window1_end)

    if window2_start is None:
        window2_start = 0
    elif window2_start < 0:
        window2_start = max(0, (video.duration_ms / 1000) + window2_start)

    if window2_end is None:
        window2_end = video.duration_ms / 1000
    elif window2_end < 0:
        window2_end = max(0, (video.duration_ms / 1000) + window2_end)

    # Analyze frames
    timestamps = process_video_window(
        video_path,
        camera_type,
        window1_start,
        window1_end,
        stride,
        debug_dir,
        brightness_boost,
    )

    if (
        window2_start > window1_end or window2_end < window1_start
    ):  # check if window2 is not overlapping with window1
        # TODO: better window checking
        timestamps2 = process_video_window(
            video_path,
            camera_type,
            window2_start,
            window2_end,
            stride,
            debug_dir,
            brightness_boost,
        )
        timestamps = {**timestamps, **timestamps2}

    if len(timestamps) == 0:
        errprint("Error: Unable to timestamp any frames.")
        return

    statistics = fit_timestamps(video, timestamps)
    print(statistics)
    # print_statistics(statistics)

    # if debug_dir:
    #     plot_timechart(
    #         filtered_x,
    #         filtered_y,
    #         x_range,
    #         y_pred,
    #         exposure_times,
    #         expected_duration,
    #         debug_dir,
    #     )
    #     plot_exposure_histogram(exposure_times, debug_dir)

    # if export_dir:
    #     export_frames(video_path, export_dir, y_pred)

    return statistics


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
