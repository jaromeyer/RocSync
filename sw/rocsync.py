import argparse
import glob
import json
import math
import os
import queue
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm


class CameraType(Enum):
    RGB = "rgb"
    INFRARED = "ir"


# Blob detector params
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = True
params.blobColor = 255
params.filterByArea = False
params.minArea = 100
params.maxArea = 1000
params.filterByCircularity = True
params.minCircularity = 0.5
params.filterByInertia = True
params.minInertiaRatio = 0.5
params.filterByConvexity = True
params.minConvexity = 0.8
params.minDistBetweenBlobs = 0.01
blob_detector = cv2.SimpleBlobDetector_create(params)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
aruco_detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# Board layout
frquency = 1000
aruco_marker_id = 0
board_size = 640
period = 100
led_size = 8
ir_radius = 280
visible_radius = 294

ir_corners = np.array(
    [
        [13, 13],
        [board_size - 14, 13],
        [board_size - 14, board_size - 13],
        [13, board_size - 14],
    ],
    dtype=np.float32,
)

corner_dots = np.array(
    [
        [51, 51],
        [board_size - 52, 51],
        [board_size - 52, board_size - 52],
        [51, board_size - 52],
    ],
    dtype=np.float32,
)

aruco_corners_coords = np.array(
    [
        [202, 202],
        [board_size - 203, 202],
        [board_size - 203, board_size - 203],
        [202, board_size - 203],
    ],
    dtype=np.float32,
)

# Global variables
filter_movement = False
movement_threshold = 5
camera_type = None
debug = False
debug_path = None
frame_number = 0
prev_corners = np.zeros((4, 2))


def print(string):
    tqdm.write(string)

def errprint(string):
    print(f"\033[91m{string}\033[0m")


def warnprint(string):
    print(f"\033[93m{string}\033[0m")


def succprint(string):
    print(f"\033[92m{string}\033[0m")
    
def boldprint(string):
    print(f"\033[1m{string}\033[0m")


def draw_polygon(points, image, color):
    for i in range(len(points)):
        cv2.line(
            image,
            tuple(map(int, points[i])),
            tuple(map(int, points[(i + 1) % len(points)])),
            color,
            2,
        )


def read_ring(extracted_board, draw_result=False):
    radius = visible_radius if camera_type == CameraType.RGB else ir_radius
    leds = np.zeros(period, dtype=bool)
    for i in range(period):
        angle = -(i / period + 0.25) * 2 * math.pi
        x = int(board_size / 2 + radius * math.cos(angle))
        y = int(board_size / 2 + radius * math.sin(angle))

        # Create mask and sample image
        led_mask = np.zeros((board_size, board_size), dtype=np.uint8)
        cv2.circle(led_mask, (x, y), led_size, (255), -1)
        mean_intensity = cv2.mean(extracted_board, led_mask)[0]
        leds[i] = mean_intensity > 20  # TODO make param

        if draw_result:
            color = (0, 0, 255) if leds[i] else (255, 0, 0)
            cv2.circle(extracted_board, (x, y), led_size, color, 1)

    potential_starts = []
    potential_ends = []
    for i in range(period):
        if not leds[(i - 1) % period] and leds[i]:
            potential_starts.append(i)
        if leds[i] and not leds[(i + 1) % period]:
            potential_ends.append(i)

    # Make sure that there is exactly ONE of enabled
    if len(potential_starts) == 1 and len(potential_ends) == 1:
        return (potential_starts[0], potential_ends[0])


def read_counter(extracted_board, draw_result=False):
    y = int(53 / 250 * 640) if camera_type == CameraType.RGB else int(48 / 250 * 640)
    counter = 0
    for i in range(1, 16):
        x = int((65 + i * 8) / 250 * 640)

        # Create mask and sample image
        led_mask = np.zeros((board_size, board_size), dtype=np.uint8)
        cv2.circle(led_mask, (x, y), led_size, (255), -1)
        mean_intensity = cv2.mean(extracted_board, led_mask)[0]
        enabled = mean_intensity > 20  # TODO make param
        if enabled:
            counter += 2 ** (15 - i)
        if draw_result:
            color = (0, 0, 255) if enabled else (255, 0, 0)
            cv2.circle(extracted_board, (x, y), led_size, color, 1)
    return counter


def find_corners_convexhull(mask):
    points = blob_detector.detect(mask)

    # Draw detected blobs as red circles
    debug_image = cv2.drawKeypoints(
        mask,
        points,
        np.array([]),
        (0, 0, 255),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    points = [kp.pt for kp in points]

    # Find the convex hull and identify the corners
    if len(points) >= 4:
        hull = cv2.convexHull(np.array(points, dtype=np.float32))
        corners = hull.reshape(-1, 2)
        draw_polygon(corners, debug_image, (0, 255, 0))

        if len(hull) > 4:
            # Approximate to 4 points
            epsilon_factor = 0.02
            n_points = len(hull)
            approx_hull = hull
            while n_points > 4:
                epsilon = epsilon_factor * cv2.arcLength(hull, True)
                approx_hull = cv2.approxPolyDP(hull, epsilon, True)
                n_points = len(approx_hull)
                epsilon_factor += 0.02

            # Draw the approxcuimated convex hull
            corners = approx_hull.reshape(-1, 2)
            draw_polygon(corners, debug_image, (255, 0, 0))

        if len(corners) == 4:
            # cv2.imwrite(f"{debug_path}/convexhull_{frame_number}.png", debug_image)
            return corners  # np.roll(corners, -topleft_corner, axis=0)
    # cv2.imwrite(f"{debug_path}/convexhull_{frame_number}.png", debug_image)


def find_corners_dots(mask):
    points = blob_detector.detect(mask)
    if not points:
        return
    closest_points = [min(points, key=lambda p: np.linalg.norm(p.pt - target)) for target in corner_dots]
    if max([np.linalg.norm(closest_points[i].pt - target) for i, target in enumerate(corner_dots)]) > 50:  # TODO: check this shit
        return

    if debug:
        debug_image = cv2.drawKeypoints(
            mask,
            points,
            np.array([]),
            (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        cv2.imwrite(f"{debug_path}/corner_{frame_number}.png", debug_image)

    return np.array([kp.pt for kp in closest_points], dtype=np.float32)


def find_corners_aruco(mask):
    markers, marker_ids, _ = aruco_detector.detectMarkers(mask)
    if debug:
        debug_image = mask.copy()
        cv2.aruco.drawDetectedMarkers(debug_image, markers, marker_ids)
        cv2.imwrite(f"{debug_path}/aruco_{frame_number}.png", debug_image)

    if marker_ids is None:
        return
    marker_dict = {id[0]: marker for id, marker in zip(marker_ids, markers)}
    if aruco_marker_id in marker_dict.keys():
        return marker_dict[aruco_marker_id]


def process_frame(image):
    match camera_type:
        case CameraType.RGB:
            # First extract course PCB using ArUco marker
            aruco_corners = find_corners_aruco(image)
            if aruco_corners is None:
                return
            red_channel = image[:, :, 2]
            _, mask = cv2.threshold(red_channel, 220, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

            # Use course PCB to accurately extract corner
            rough_transformation_matrix = cv2.getPerspectiveTransform(aruco_corners, aruco_corners_coords)
            rough_pcb = cv2.warpPerspective(mask, rough_transformation_matrix, (board_size, board_size))
            corners = find_corners_dots(rough_pcb)
            if corners is None:
                return
            transformation_matrix = np.dot(cv2.getPerspectiveTransform(corners, corner_dots), rough_transformation_matrix)
            pcb = cv2.warpPerspective(mask, transformation_matrix, (board_size, board_size))
        case CameraType.INFRARED:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            corners = find_corners_convexhull(mask)
            if corners is None:
                return
            transformation_matrix = cv2.getPerspectiveTransform(corners, ir_corners)
            pcb = cv2.warpPerspective(mask, transformation_matrix, (board_size, board_size))
            # Find correct rotation
            for _ in range(4):
                if read_counter(pcb) == 0:
                    pcb = cv2.rotate(pcb, cv2.ROTATE_90_CLOCKWISE)

    if debug:
        cv2.imwrite(f"{debug_path}/reprojected_{frame_number}.png", pcb)
    counter = read_counter(pcb, draw_result=True)
    match (read_ring(pcb, draw_result=True)):
        case None:
            return
        case start, end:
            if start > end or min(start, period - start) <= 2 or min(period, 100 - end) <= 2:
                # Counter increment during exposure
                return

            start += counter * period
            end += counter * period
            if debug:
                cv2.putText(
                    pcb,
                    f"Start: {start} ms, End: {end} ms",
                    (50, board_size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                )
                cv2.imwrite(f"{debug_path}/f{frame_number}_s{start}_e{end}.png", pcb)
            return (start, end)


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


def process_video(video_path, export_frames=False, stride=None):
    global frame_number
    global debug_path
    
    print(f"Working on {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        errprint(f"Error: Could not open video: {video_path}")
        return

    name = os.path.splitext(os.path.basename(video_path))[-2]

    output_path = f"output/{name}"
    debug_path = f"{output_path}/debug"
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(debug_path)

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
    for _ in tqdm(range(n_frames), desc="Analyzing video", position=1):
        frame, frame_number = frame_queue.get()  # blocking wait
        if frame is None:
            errprint("Error: Input stream ended unexpectedly.")
            return
        if scan_window > 0 or frame_number % stride == 0:
            timestamp = process_frame(frame)
            scan_window -= 1
            if timestamp is not None:
                timestamps[frame_number] = timestamp
                scan_window = 5

    # Assuming constant frame rate, fit linear model
    x = np.array(list(timestamps.keys())).reshape(-1, 1)
    y = np.array([start for start, _ in timestamps.values()])
    model = RANSACRegressor(residual_threshold=20)  # max accepted distance to linear model in ms
    model.fit(x, y)

    # remove outliers from timestamps
    timestamps = {k: v for (k, v), keep in zip(list(timestamps.items()), model.inlier_mask_) if keep}
    x = np.array(list(timestamps.keys())).reshape(-1, 1)
    y = np.array([start for start, _ in timestamps.values()])

    r2_score = model.score(x, y)
    rmse = root_mean_squared_error(y, model.predict(x))
    x_range = np.arange(0, n_frames).reshape(-1, 1)
    y_pred = model.predict(x_range)

    exposure_times = [end - start for start, end in timestamps.values()]
    measured_duration = y_pred[-1] - y_pred[0]
    measured_fps = (n_frames - 1) / measured_duration * 1000
    speed_factor = measured_duration / expected_duration

    if export_frames:
        os.makedirs(f"output/{name}/frames", exist_ok=True)
        # Read frames in separate thread
        frame_queue = queue.Queue(maxsize=100)
        threading.Thread(target=read_frames_async, args=(cap, frame_queue)).start()

        # Export frames concurrently using multiple threads
        with ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(export_frame_async, frame_queue, y_pred, f"output/{name}/frames") for _ in range(n_frames)]
            for future in tqdm(as_completed(futures), total=n_frames, desc="Exporting frames", position=1):
                future.result()
    cap.release()

    def printresult(name, value, is_valid):
        string = f"{name+":":<40} {value:>20}"
        if is_valid:
            succprint(string)
        else:
            errprint(string)

    # Print result
    format_str = "{:<30} {:>30}"
    printresult("Number of considered frames", len(timestamps), len(timestamps) > 10)
    printresult("Number of rejected outliers", np.sum(~np.array(model.inlier_mask_)), np.all(model.inlier_mask_))
    printresult("R2 score", f"{r2_score:.4f}", r2_score > 0.99)
    printresult("RMSE", f"{rmse:.2f} ms", rmse < 2)
    print(format_str.format("First frame:", f"{y_pred[0]:.1f} ms"))
    print(format_str.format("Last frame:", f"{y_pred[-1]:.1f} ms"))
    print(format_str.format("Expected duration (fps):", f"{expected_duration:.1f} ms ({fps:.2f} fps)"))
    print(format_str.format("Actual duration (fps):", f"{measured_duration:.1f} ms ({measured_fps:.2f} fps)"))
    print(format_str.format("Delta (actual - expected)", f"{measured_duration-expected_duration:.2f} ms ({speed_factor/100:.3f}% speed)"))
    print(format_str.format("Exposure time (mean/min/max):", f"{np.mean(exposure_times):.2f}/{np.min(exposure_times):.2f}/{np.max(exposure_times):.2f} ms"))
    print(61 * "-")
    # print(f"ffmpeg settings: '-ss {-y_pred[0]/speed_factor/1000:.3f}' 'setpts=PTS*{speed_factor}'")

    # Plot timechart
    plt.figure()
    plt.scatter(x, y, color="blue", label="Measurements")
    plt.plot(x_range, y_pred, color="blue", label="Measured frametime")
    plt.plot(x_range, np.linspace(y_pred[0], y_pred[0] + expected_duration, n_frames), color="red", label="Calculated frametime")
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
    plt.savefig(f"output/{name}/timestamps.png")

    # Plot exposure time histogram
    plt.figure()
    unique_values, counts = np.unique(exposure_times, return_counts=True)
    bar = plt.bar(unique_values, counts)
    plt.bar_label(bar, counts)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Exposure time [ms]")
    plt.ylabel("Number of measured frames")
    plt.title("Exposure time histogram")
    plt.savefig(f"output/{name}/exposure.png")
    
    return {
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


def process_image(path):
    image = cv2.imread(path)
    timestamp = process_frame(image)
    if timestamp is not None:
        succprint(f"Measured duration from {timestamp[0]} to {timestamp[1]} ms")
    else:
        warnprint("Timestamp not available")


def process_directory(path):
    extensions = ["jpg", "jpeg", "png", "bmp", "tiff"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(path, f"*.{ext}")))
    image_paths = sorted(image_paths)

    image_idx = 0
    while True:
        process_image(image_paths[image_idx])

        # Handle user input
        match cv2.waitKeyEx(1):
            case 113 | 27:  # 'q' or ESC to quit
                break
            case 2555904:  # right arrow to show next image
                image_idx = (image_idx + 1) % len(image_paths)
            case 2424832:  # left arrow to show previous image
                image_idx = (image_idx - 1) % len(image_paths)

    # Cleanup
    cv2.destroyAllWindows()


def main():
    global debug
    global camera_type

    parser = argparse.ArgumentParser(description="Extract timestamps from images and videos showing the RocSync device.")
    parser.add_argument("path", type=str, nargs="+", help="path to the input file/directory")
    parser.add_argument("--camera_type", choices=[e.value for e in CameraType], default=CameraType.RGB.value, help="specify the type of camera (default: rgb)")
    parser.add_argument("--stride", type=int, metavar="N", help="only scan every N-th frame (only applies to videos)")
    parser.add_argument("--export_frames", type=str, metavar="PATH", help="export all raw frames as PNGs with timestamp (only applies to videos)")
    parser.add_argument("--output", type=str, metavar="PATH", help="directory to store result (will be created if it does not exist)")
    parser.add_argument("--debug", action="store_true", help="save debug images (very slow)")

    args = parser.parse_args()

    # if args.video is None:
    #     if args.stride is not None:
    #         parser.error("--stride can only be used with --video")
    #     elif args.export_frames is not None:
    #         parser.error("--export_frames can only be used with --video")

    debug = args.debug
    camera_type = CameraType(args.camera_type)

    files = []
    for path in args.path:
        if os.path.isdir(path):
            for root, _, dir_files in os.walk(path):
                for file in dir_files:
                    file_path = os.path.join(root, file)
                    files.append(file_path)
        elif os.path.isfile(path):
            files.append(path)
        else:
            print(f"Invalid path: {path}")
    
    result = {}
    for file in tqdm(files, desc="Processing files", position=0):
        if path.lower().endswith(('.png', '.jpg', '.jpeg')):
            result[file] = process_image(file)
        elif path.lower().endswith(('.mp4', '.avi', '.mov')):
            result[file] = process_video(file)
        else:
            errprint(f"Error: Unsupported file type: {file}")
    
    if args.output:
        with open(f"output/result.json", "w") as file:
            json.dump(result, file, indent=4)


if __name__ == "__main__":
    main()
