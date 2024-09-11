import os
import queue
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm

from printer import *
from vision import process_frame


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


def export_frames(cap, path, y_pred):
    n_frames = len(y_pred)
    os.makedirs(path, exist_ok=True)
    # Read frames in separate thread
    frame_queue = queue.Queue(maxsize=100)
    threading.Thread(target=read_frames_async, args=(cap, frame_queue)).start()

    # Export frames concurrently using multiple threads
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = []
        for _ in range(n_frames):
            futures.append(executor.submit(export_frame_async, frame_queue, y_pred, path))
        for future in tqdm(
            as_completed(futures), total=n_frames, desc="Exporting frames", position=1
        ):
            future.result()


def process_video(video_path, camera_type, export_frames=None, stride=None, debug_dir=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        errprint(f"Error: Could not open video: {video_path}")
        return

    # name = os.path.splitext(os.path.basename(video_path))[-2]
    # output_path = f"output/{name}"
    # debug_path = f"{output_path}/debug"
    # if os.path.exists(output_path):
    #     shutil.rmtree(output_path)
    # os.makedirs(debug_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    expected_duration = (n_frames - 1) / fps * 1000

    # Read frames in separate thread
    frame_queue = queue.Queue(maxsize=100)
    threading.Thread(target=read_frames_async, args=(cap, frame_queue)).start()

    # Analyze frames
    timestamps = {}
    scan_window = 0
    if stride is None:
        stride = int(fps)
    for frame_number in tqdm(range(n_frames), desc="Analyzing frames", position=1):
        frame, frame_number = frame_queue.get()  # blocking wait
        # ret, frame = cap.read()
        if frame is None:
            errprint("Error: Input stream ended unexpectedly.")
            return
        if scan_window > 0 or frame_number % stride == 0:
            timestamp = process_frame(frame, camera_type, debug_dir)
            scan_window -= 1
            if timestamp is not None:
                timestamps[frame_number] = timestamp
                scan_window = 5

    if len(timestamps) == 0:
        errprint("Error: Unable to timestamp any frames.")
        return

    # Assuming constant frame rate, fit linear model
    x = np.array(list(timestamps.keys())).reshape(-1, 1)
    y = np.array([start for start, _ in timestamps.values()])
    model = RANSACRegressor(residual_threshold=20)  # max accepted distance to linear model in ms
    model.fit(x, y)
    r2_score = model.score(x, y)
    rmse = root_mean_squared_error(y, model.predict(x))
    x_range = np.arange(0, n_frames).reshape(-1, 1)
    y_pred = model.predict(x_range)

    # remove outliers
    timestamps = {
        k: v for (k, v), keep in zip(list(timestamps.items()), model.inlier_mask_) if keep
    }

    if export_frames:
        export_dir = f"{debug_dir}/export"
        os.makedirs(export_dir)
        export_frames(cap, export_dir, y_pred)
    cap.release()

    x = np.array(list(timestamps.keys())).reshape(-1, 1)
    y = np.array([start for start, _ in timestamps.values()])

    exposure_times = [end - start for start, end in timestamps.values()]
    measured_duration = y_pred[-1] - y_pred[0]
    measured_fps = (n_frames - 1) / measured_duration * 1000
    speed_factor = measured_duration / expected_duration

    result = {
        "n_frames": n_frames,
        "r2": r2_score,
        "rmse": rmse,
        "expected_duration": expected_duration,
        "measured_duration": measured_duration,
        "speed_factor": speed_factor,
        "start": y_pred[0],
        "end": y_pred[-1],
        "mean_exposure_time": np.mean(exposure_times),
        "min_exposure_time": float(np.min(exposure_times)),
        "max_exposure_time": float(np.max(exposure_times)),
        "measured_timestamps": timestamps,
        "interpolated_timestamps": y_pred.tolist(),
    }

    # Print result
    format_str = "{:<30} {:>30}"
    printresult("Number of considered frames", len(timestamps), len(timestamps) > 10)
    printresult(
        "Number of rejected outliers",
        np.sum(~np.array(model.inlier_mask_)),
        np.all(model.inlier_mask_),
    )
    printresult("R2 score", f"{r2_score:.4f}", r2_score > 0.99)
    printresult("RMSE", f"{rmse:.2f} ms", rmse < 2)
    print(format_str.format("First frame:", f"{y_pred[0]:.1f} ms"))
    print(format_str.format("Last frame:", f"{y_pred[-1]:.1f} ms"))
    print(
        format_str.format("Expected duration (fps):", f"{expected_duration:.1f} ms ({fps:.2f} fps)")
    )
    print(
        format_str.format(
            "Actual duration (fps):", f"{measured_duration:.1f} ms ({measured_fps:.2f} fps)"
        )
    )
    print(
        format_str.format(
            "Delta (actual - expected)",
            f"{measured_duration-expected_duration:.2f} ms ({speed_factor/100:.3f}% speed)",
        )
    )
    print(
        format_str.format(
            "Exposure time (mean/min/max):",
            f"{np.mean(exposure_times):.2f}/{np.min(exposure_times):.2f}/{np.max(exposure_times):.2f} ms",
        )
    )
    print(61 * "-")

    # print_statistics(result)
    # if debug_dir:
    #     plot_statistics(result, debug_dir)
    return result


# def print_statistics(result):
#     format_str = "{:<30} {:>30}"
#     printresult("Number of considered frames", len(timestamps), len(timestamps) > 10)
#     printresult(
#         "Number of rejected outliers",
#         np.sum(~np.array(model.inlier_mask_)),
#         np.all(model.inlier_mask_),
#     )
#     printresult("R2 score", f"{result["r2_score"]:.4f}", ["r2_score"] > 0.99)
#     printresult("RMSE", f"{result["rmse"]:.2f} ms", result["rmse"] < 2)
#     print(format_str.format("First frame:", f"{y_pred[0]:.1f} ms"))
#     print(format_str.format("Last frame:", f"{y_pred[-1]:.1f} ms"))
#     print(
#         format_str.format("Expected duration (fps):", f"{expected_duration:.1f} ms ({fps:.2f} fps)")
#     )
#     print(
#         format_str.format(
#             "Actual duration (fps):",
#             f"{measured_duration:.1f} ms ({measured_fps:.2f} fps)",
#         )
#     )
#     print(
#         format_str.format(
#             "Delta (actual - expected)",
#             f"{measured_duration-expected_duration:.2f} ms ({speed_factor/100:.3f}% speed)",
#         )
#     )
#     print(
#         format_str.format(
#             "Exposure time (mean/min/max):",
#             f"{np.mean(exposure_times):.2f}/{np.min(exposure_times):.2f}/{np.max(exposure_times):.2f} ms",
#         )
#     )
#     print(61 * "-")
#     # print(f"ffmpeg settings: '-ss {-y_pred[0]/speed_factor/1000:.3f}' 'setpts=PTS*{speed_factor}'")


# def plot_statistics(result, debug_dir):
#     # Plot timechart
#     plt.figure()
#     plt.scatter(x, y, color="blue", label="Measurements")
#     plt.plot(x_range, y_pred, color="blue", label="Measured frametime")
#     plt.plot(
#         x_range,
#         np.linspace(y_pred[0], y_pred[0] + expected_duration, n_frames),
#         color="red",
#         label="Calculated frametime",
#     )
#     plt.xlabel("Frame number")
#     plt.ylabel("Time relative to RocSync [ms]")
#     plt.title("Frame timing")
#     plt.gca().ticklabel_format(style="plain", useOffset=False)
#     plt.legend()
#     plt.grid(True)
#     ax2 = plt.gca().twinx()
#     ax2.scatter(x, exposure_times, color="green", label="Exposure time [ms]")
#     ax2.set_ylabel("Exposure time [ms]")
#     ax2.ticklabel_format(style="plain", useOffset=False)
#     ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
#     ax2.legend(loc="upper right")
#     plt.savefig(f"{debug_dir}/timestamps.png")

#     # Plot exposure time histogram
#     plt.figure()
#     unique_values, counts = np.unique(exposure_times, return_counts=True)
#     bar = plt.bar(unique_values, counts)
#     plt.bar_label(bar, counts)
#     plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
#     plt.xlabel("Exposure time [ms]")
#     plt.ylabel("Number of measured frames")
#     plt.title("Exposure time histogram")
#     plt.savefig(f"{debug_dir}/exposure.png")
