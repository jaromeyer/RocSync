import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

# Reuse the same video extensions as extract_synced_videos.py
camera_formats = {".mp4", ".mov", ".avi", ".mkv"}


@dataclass
class Config:
    dataset_folder: str                   # root folder containing camera videos and 'time sync/'
    time_sync_json_path: str             # path to time_synchronization_*.json
    from_camera: str                     # camera that defines the local time
    time_string: str                     # time in that camera's local time (HH:MM:SS.mmm)

def get_screen_size():
    """
    Try to detect the screen resolution using tkinter.
    Fallback to 1920x1080 if anything goes wrong.
    """
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return 1920, 1080

def _auto_find_time_sync_json(dataset_folder: Path) -> Path:
    """
    Look for 'time sync/time_synchronization_*.json' inside dataset_folder.
    Prefer a single match; if multiple, pick the first in sorted order.
    """
    base = dataset_folder / "time sync"
    candidates = sorted(base.glob("time_synchronization_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No time_synchronization_*.json found under: {base}"
        )
    return candidates[0]


def _parse_timecode_to_milliseconds(timecode: str) -> int:
    """
    'HH:MM:SS.mmm' -> milliseconds since 00:00:00.000
    """
    dt = datetime.strptime(timecode, "%H:%M:%S.%f")
    delta = (dt.hour * 3600 + dt.minute * 60 + dt.second) * 1000 + dt.microsecond // 1000
    return delta


def _ms_to_timecode(ms: float) -> str:
    """
    milliseconds -> 'HH:MM:SS.mmm' (string)
    """
    ms_int = int(round(ms))
    seconds, milliseconds = divmod(ms_int, 1000)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{secs:02}.{milliseconds:03}"


def _get_theoretical_timestamps_of_all_frames(time_sync_data: dict) -> List[float]:
    """
    Build per-frame REAL timeline (in ms) for this camera.
    Prefer measured_fps (+ optional speed_factor) when available; fall back to linear interpolation.
    """
    first_frame = float(time_sync_data["first_frame"])
    last_frame = float(time_sync_data["last_frame"])
    n_frames = int(time_sync_data["n_frames"])

    measured_fps = time_sync_data.get("measured_fps", None)
    speed_factor = time_sync_data.get("speed_factor", None)

    if measured_fps and measured_fps > 0:
        dt = 1000.0 / float(measured_fps)
        if speed_factor and speed_factor > 0:
            dt *= float(speed_factor)
        return [first_frame + n * dt for n in range(n_frames)]
    else:
        if n_frames <= 1:
            return [first_frame]
        step = (last_frame - first_frame) / (n_frames - 1)
        return [first_frame + n * step for n in range(n_frames)]


def _build_arg_parser() -> argparse.ArgumentParser:
    # Default dataset folder = parent of script (same logic as extract_synced_videos.py)
    try:
        script_parent_dir = Path(__file__).resolve().parent.parent
        default_dataset_folder = str(script_parent_dir)
    except Exception:
        default_dataset_folder = os.getcwd()

    p = argparse.ArgumentParser(
        description=(
            "Check time synchronization by extracting a single moment from ALL cameras.\n"
            "You specify a local time in one camera (via --from-camera and --time); "
            "the script converts it to global time, checks that it lies inside the common "
            "overlap of all cameras, then shows the corresponding frame of each camera."
        )
    )
    p.add_argument(
        "--dataset-folder",
        default=default_dataset_folder,
        help=(
            "Root folder containing camera videos and a 'time sync' subfolder. "
            "Defaults to the parent folder of the script's location."
        ),
    )
    p.add_argument(
        "--from-camera",
        required=True,
        help="Camera basename that defines the local time (e.g. 'canon33').",
    )
    p.add_argument(
        "--time",
        dest="time_string",
        required=True,
        help="Time in that camera's local time, format 'HH:MM:SS.mmm' (e.g. '00:05:00.000').",
    )
    p.add_argument(
        "--time-sync-json",
        dest="time_sync_json_path",
        help="Path to time_synchronization_*.json (optional; auto-detected if omitted).",
    )

    return p


def compute_global_time_from_camera(
    time_string: str, camera_time_sync_data: dict
) -> float:
    """
    Convert a time given in camera-local units to GLOBAL ms using first_frame & speed_factor.
    (Same math as in Clip._convert_to_real_time, but for a single time.)
    """
    local_ms = _parse_timecode_to_milliseconds(time_string)
    first_frame = float(camera_time_sync_data["first_frame"])
    speed_factor = float(camera_time_sync_data.get("speed_factor", 1.0))
    global_ms = first_frame + speed_factor * local_ms
    return float(global_ms)


def validate_moment_in_overlap(global_ms: float, time_sync_data: Dict[str, dict]) -> None:
    """
    Check that the chosen global moment lies inside the COMMON temporal overlap of all cameras:
        global_ms >= max(first_frame_i)
        global_ms <= min(last_frame_i)
    Raise ValueError if not.
    """
    if not time_sync_data:
        raise ValueError("Time synchronization JSON is empty – no cameras found.")

    first_frames = [float(d["first_frame"]) for d in time_sync_data.values()]
    last_frames = [float(d["last_frame"]) for d in time_sync_data.values()]

    overlap_start = max(first_frames)
    overlap_end = min(last_frames)

    if overlap_start >= overlap_end:
        raise ValueError(
            f"No temporal overlap between cameras:\n"
            f"  max(first_frame)={overlap_start:.2f} ms ({_ms_to_timecode(overlap_start)}) >=\n"
            f"  min(last_frame)={overlap_end:.2f} ms ({_ms_to_timecode(overlap_end)})."
        )

    if not (overlap_start <= global_ms <= overlap_end):
        raise ValueError(
            "Requested moment is OUTSIDE the common overlap of all cameras.\n"
            f"  Moment: {global_ms:.2f} ms ({_ms_to_timecode(global_ms)})\n"
            f"  Overlap (all cameras):\n"
            f"    [{overlap_start:.2f} ms ({_ms_to_timecode(overlap_start)}), "
            f"{overlap_end:.2f} ms ({_ms_to_timecode(overlap_end)})]\n\n"
            "Choose a different --time (in the from-camera) that maps into this overlap."
        )


def extract_frame_for_moment(
    video_path: str, time_sync_data: dict, global_ms: float
) -> Optional[np.ndarray]:
    """
    For a given camera:
      - build its theoretical per-frame timestamps (in global ms),
      - find the frame whose timestamp is closest to global_ms,
      - grab that frame from the video and return it as an image (BGR).
    """
    if not os.path.exists(video_path):
        print(f"[WARN] Video not found: {video_path}")
        return None

    suffix = Path(video_path).suffix.lower()
    if suffix not in camera_formats:
        print(f"[WARN] Not a recognized video format, skipping: {video_path}")
        return None

    timestamps = np.asarray(
        _get_theoretical_timestamps_of_all_frames(time_sync_data),
        dtype=np.float64,
    )
    # Find closest frame
    idx = int(np.argmin(np.abs(timestamps - global_ms)))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"[WARN] Could not read frame {idx} from {video_path}")
        return None

    return frame


def main(cfg: Config) -> None:
    dataset_folder = Path(cfg.dataset_folder)
    print(f"[INFO] Dataset folder: {dataset_folder}")
    print(f"[INFO] Using time sync JSON: {cfg.time_sync_json_path}")
    print(f"[INFO] From-camera: {cfg.from_camera}")
    print(f"[INFO] Local time in that camera: {cfg.time_string}")
    
    screen_w, screen_h = get_screen_size()
    # Load time sync JSON
    with open(cfg.time_sync_json_path, "r", encoding="utf-8") as f:
        raw_time_sync_data: Dict[str, dict] = json.load(f)

    # Normalize keys to '<parent>/<file>' so they can be joined with dataset_folder,
    # exactly like in extract_synced_videos.py.
    time_sync_data = {
        os.path.join(*camera.replace("\\", "/").split("/")[-2:]): data
        for camera, data in raw_time_sync_data.items()
    }

    # Find the time-defining camera entry by basename match
    matching_key = next(
        (
            k
            for k in time_sync_data
            if os.path.splitext(os.path.basename(k))[0] == cfg.from_camera
        ),
        None,
    )
    if not matching_key:
        raise ValueError(
            f"No match found for camera basename '{cfg.from_camera}' in time sync JSON.\n"
            f"Available basenames: "
            f"{sorted({os.path.splitext(os.path.basename(k))[0] for k in time_sync_data.keys()})}"
        )

    time_defining_camera_data = time_sync_data[matching_key]

    # Convert local camera time -> global ms
    global_ms = compute_global_time_from_camera(cfg.time_string, time_defining_camera_data)
    print(
        f"[INFO] Local time '{cfg.time_string}' in '{cfg.from_camera}' "
        f"maps to global time: {global_ms:.2f} ms ({_ms_to_timecode(global_ms)})"
    )

    # Validate that this moment lies in the common overlap of all cameras
    validate_moment_in_overlap(global_ms, time_sync_data)
    print("[INFO] Moment lies inside the common overlap of all cameras. Extracting frames...")

    # For each camera, extract and show the frame corresponding to this global time
        # Pre-extract frames for each camera for this global time
    frames_info = []  # list of (camera_basename, frame)
    for camera_rel_path, cam_sync_data in tqdm(
        time_sync_data.items(), desc="Extracting frames for each camera"
    ):
        video_path = dataset_folder / camera_rel_path
        camera_basename = os.path.splitext(os.path.basename(camera_rel_path))[0]

        frame = extract_frame_for_moment(str(video_path), cam_sync_data, global_ms)
        if frame is None:
            continue

        frames_info.append((camera_basename, frame))

    if not frames_info:
        print("[WARN] No frames could be extracted for any camera.")
        return

    print(
        "[INFO] Use RIGHT arrow to go forward, LEFT arrow to go back. "
        "Press 'q' or ESC to quit."
    )

    idx = 0
    window_name = "Time Sync Check"

    while True:
        camera_basename, frame = frames_info[idx]

        # --- Resize so width <= 500 px while keeping aspect ratio ---
        max_width = 1000
        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / float(w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame_display = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w
        else:
            frame_display = frame.copy()
        # ------------------------------------------------------------

        title = (
            f"{camera_basename}  @  {cfg.time_string}  "
            f"(global: {_ms_to_timecode(global_ms)})  "
            f"[{idx + 1}/{len(frames_info)}]"
        )

        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setWindowTitle(window_name, title)
        cv2.imshow(window_name, frame_display)

        # Try to get the actual window size (OpenCV >= 4.5)
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
        except Exception:
            win_w, win_h = w, h

        # Center window on screen
        x = max(0, int((screen_w - win_w) / 2))
        y = max(0, int((screen_h - win_h) / 2))
        cv2.moveWindow(window_name, x, y)

        key = cv2.waitKeyEx(0)

        # Handle quit: q or ESC
        if key in (27, ord("q"), ord("Q")):
            break

        # Handle LEFT / RIGHT arrows
        # OpenCV arrow key codes:
        #   left  = 81 or 2424832
        #   right = 83 or 2555904
        if key in (81, 2424832):  # LEFT
            idx = (idx - 1) % len(frames_info)
        elif key in (83, 2555904):  # RIGHT
            idx = (idx + 1) % len(frames_info)
        else:
            # Any other key: move forward
            idx = (idx + 1) % len(frames_info)

    cv2.destroyAllWindows()
    print("[INFO] Done. You’ve inspected this moment across all cameras.")



if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    dataset_folder = Path(args.dataset_folder)

    # Auto-detect time sync JSON unless explicitly provided
    if args.time_sync_json_path:
        time_sync_path = Path(args.time_sync_json_path)
    else:
        time_sync_path = _auto_find_time_sync_json(dataset_folder)

    cfg = Config(
        dataset_folder=str(dataset_folder),
        time_sync_json_path=str(time_sync_path),
        from_camera=args.from_camera,
        time_string=args.time_string,
    )
    main(cfg)
