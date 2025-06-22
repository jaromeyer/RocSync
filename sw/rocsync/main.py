import argparse
import json
import os
import pathlib
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from rocsync.printer import *
from rocsync.video import process_video
from rocsync.vision import CameraType, process_frame
from rocsync.ftk import process_ftk_recording

import subprocess


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


def process_image(path, camera_type, debug_dir=None):
    image = cv2.imread(path)
    timestamp = process_frame(image, camera_type, debug_dir)
    if timestamp is not None:
        succprint(f"start: {timestamp[0]} ms, end {timestamp[1]} ms")
        return {"start": timestamp[0], "end": timestamp[1]}
    else:
        errprint("Error: Unable to decode timestamp.")


def mkdir_unique(name, parent_dir):
    parent_path = Path(parent_dir)
    debug_dir = parent_path / name
    if not debug_dir.exists():
        debug_dir.mkdir(parents=True)
    else:
        counter = 2
        debug_dir = parent_path / f"{name} ({counter})"
        while debug_dir.exists():
            counter += 1
            debug_dir = parent_path / f"{name} ({counter})"
        debug_dir.mkdir(parents=True)
    return str(debug_dir)


def parse_time(time_str: str) -> float:
    """Expects a delta time string with format hh:mm:ss.ms"""
    if time_str is None:
        return None

    time = time_str.split(":")
    if len(time) != 3:
        errprint(f"Invalid time format: {time_str}, expected hh:mm:ss.ms")
        raise ValueError
    try:
        h = int(time[0])
        m = int(time[1])
        s = float(time[2])
    except ValueError:
        errprint(f"Invalid time format: {time_str}, expected hh:mm:ss.ms")
        raise ValueError

    return h * 3600 + m * 60 + s


def sync_video(
    video_path: str,
    stats: dict,
    offset: float = 0,
    output_file: str = "synced.mp4",
    frame_rate: int = 30,
    compensate_drift: bool = True,
) -> subprocess.Popen:
    cut_time = stats["first_frame"] * (-1 / 1000) + offset  # in seconds
    speed_factor = stats["speed_factor"]

    # Check if nvenc is available for speed up
    nvenc_available = False
    if compensate_drift:
        try:
            cmd = "ffmpeg -hide_banner -encoders | grep hevc_nvenc"
            encoders = subprocess.check_output(cmd, shell=True).decode("utf-8")
            if "hevc_nvenc" not in encoders:
                raise subprocess.CalledProcessError(1, cmd)
            else:
                nvenc_available = True
        except subprocess.CalledProcessError:
            warnprint(
                "hevc_nvenc not available, encoding will be very slow. Install NVIDIA drivers and ffmpeg with nvenc support or disable drift compensation."
            )

    ffmpeg_command = [
        "ffmpeg",
        "-ss",
        str(cut_time),
        "-i",
        video_path,
    ]

    if compensate_drift:
        ffmpeg_command += [
            "-c:v",
            "hevc_nvenc" if nvenc_available else "libx265",
            "-crf",
            "0",
            "-filter_complex",
            f'"setpts=PTS*{speed_factor}"',
            "-r",
            str(frame_rate),
        ]
    else:
        ffmpeg_command += [
            "-c:v",
            "copy",
        ]
    ffmpeg_command += [
        "-y",
        output_file,
    ]

    cmd_str = " ".join(ffmpeg_command)
    print(cmd_str)

    process = subprocess.Popen(cmd_str, shell=True)
    stdout, sterr = process.communicate()

    return process


def main():
    parser = argparse.ArgumentParser(
        description="Extract timestamps from images and videos showing the RocSync device."
    )
    parser.add_argument(
        "path",
        type=str,
        metavar="PATH",
        nargs="+",
        help="path to a video, image, or directory containing videos and/or images",
    )
    parser.add_argument(
        "-c",
        "--camera_type",
        choices=[e.value for e in CameraType],
        default=CameraType.RGB.value,
        help="specify the type of camera (default: rgb)",
    )
    parser.add_argument(
        "-s",
        "--stride",
        type=int,
        metavar="N",
        help="scan every N-th frame only (default: same as framerate, only applies to videos)",
    )
    parser.add_argument(
        "-e",
        "--export_frames",
        type=str,
        metavar="DIRECTORY",
        help="directory to store all raw frames as PNGs with timestamp (only applies to videos)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        metavar="FILE",
        help="JSON file to store results",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="do not ask for confirmation when processing multiple files",
    )
    parser.add_argument(
        "--sync_video",
        action="store_true",
        help="sync and cut video to predicted timestamps",
    )
    parser.add_argument(
        "--synced_folder",
        type=str,
        default="synced",
        help="folder to store synced videos (default: synced)",
    )
    parser.add_argument(
        "--compensate_video_drift",
        action="store_true",
        help="whether to compensate for video drift (very slow)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="desired FPS for time-synced videos, if not provided the FPS will be determined from the videos",
    )
    parser.add_argument(
        "--debug",
        type=str,
        metavar="DIRECTORY",
        help="directory to store debug images (very slow)",
    )

    # Specify time windows to search for ROCsync
    parser.add_argument(
        "--start1",
        type=str,
        default=None,
        help="start time of first window to search, in hh:mm:ss.ms format",
    )
    parser.add_argument(
        "--end1",
        type=str,
        default=None,
        help="end time window of first window to search, in hh:mm:ss.ms format",
    )
    parser.add_argument(
        "--start2",
        type=str,
        default=None,
        help="start time window of second window to search, in hh:mm:ss.ms format",
    )
    parser.add_argument(
        "--end2",
        type=str,
        default=None,
        help="end time window of second window to search, in hh:mm:ss.ms format",
    )
    parser.add_argument(
        "--recurse_in_dir",
        action="store_true",
        help="recursively search for videos and images in directories",
    )

    args = parser.parse_args()

    # Parse time arguments
    start_time1, end_time1 = parse_time(args.start1), parse_time(args.end1)
    start_time2, end_time2 = parse_time(args.start2), parse_time(args.end2)

    files = set()
    for path in args.path:
        path_obj = Path(path)
        if path_obj.is_dir():
            # walk dir recursively
            for file in (
                path_obj.rglob("*") if args.recurse_in_dir else path_obj.glob("*")
            ):
                if file.is_file():
                    files.add(file.resolve())
        elif path_obj.is_file():
            files.add(path_obj.resolve())
        else:
            errprint(f"Invalid path: {path}")
            return

    videos = sorted([f for f in files if f.suffix.lower() in [".mp4", ".avi", ".mov"]])
    images = sorted([f for f in files if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    ftk_recordings = sorted([f for f in files if f.suffix.lower() == ".csv"])


    if len(videos) + len(images) + len(ftk_recordings) > 1:
        print(f"Found {len(videos)} videos, {len(images)} images, and {len(ftk_recordings)} ftk recordings:")
        for file in videos + images + ftk_recordings:
            print(f"    {file}")
        while True and not args.yes:
            response = input("Do you want to continue (Y/n): ").strip().lower()
            if response in ["y", "yes", ""]:
                break
            elif response in ["n", "no"]:
                return
            else:
                print("Please enter 'y' or 'n'.")

    if args.debug:
        os.makedirs(args.debug, exist_ok=True)
        warnprint(f"Debug images will be stored in {args.debug}")

    if args.export_frames:
        os.makedirs(args.export_frames, exist_ok=True)
        warnprint(f"Exported frames will be stored in {args.export_frames}")

    result = {}
    if args.output:
        output_path = pathlib.Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            with output_path.open("r") as file:
                result = json.load(file)
            print(f"Loaded previous results from {args.output}")

    for file in tqdm(videos + images + ftk_recordings, desc="Processing files", position=0):
        if str(file) in result:
            print(f"Skipping {file}, already processed.")
            continue
        print(f"Working on {file}")

        debug_dir = None
        if args.debug:
            name, _ = os.path.splitext(os.path.basename(file))
            debug_dir = mkdir_unique(name, args.debug)

        export_dir = None
        if args.export_frames:
            name, _ = os.path.splitext(os.path.basename(file))
            export_dir = mkdir_unique(name, args.export_frames)

        if file in videos:
            ret = process_video(
                file,
                CameraType(args.camera_type),
                export_dir,
                args.stride,
                debug_dir,
                start_time1,
                end_time1,
                start_time2,
                end_time2,
            )
            if ret is not None:
                result[str(file)] = ret.to_dict()
            else:
                errprint(f"Error: Unable to time-sync {file}.")
        elif file in images:
            ret = process_image(file, CameraType(args.camera_type), debug_dir)
            if ret is not None:
                result[str(file)] = ret
            else:
                errprint(f"Error: Unable to time-sync {file}.")
        elif file in ftk_recordings:
            timestamps = process_ftk_recording(file)
            if timestamps is not None:
                result[str(file)] = timestamps
            else:
                errprint(f"Error: Unable to process FTK recording {file}.")

        # Save result to file after every video to avoid data loss
        if args.output:
            with output_path.open("w") as f:
                json.dump(result, f, indent=4, cls=NpEncoder)
            print(f"Result written to {args.output}")

    if args.sync_video:
        output_path = pathlib.Path(args.output)
        with open(output_path, "r") as file:
            stats = json.load(file)

        expected_fps = (
            int(round(list(stats.values())[0]["expected_fps"]))
            if args.fps is None
            else args.fps
        )
        print(f"Syncing all videos to {expected_fps} FPS")

        processes = []
        for file in stats:
            if stats[file]:
                # Check if the output file already exists
                video_name, _ = os.path.splitext(os.path.basename(file))
                video_folder = os.path.dirname(file)
                output_folder = os.path.join(video_folder, args.synced_folder)
                output_file = os.path.join(output_folder, f"{video_name}.mp4")
                os.makedirs(output_folder, exist_ok=True)

                if os.path.exists(output_file):
                    try:
                        vid = cv2.VideoCapture(output_file)
                        if not vid.isOpened():
                            raise ValueError("Could not open video file")
                    except Exception as e:
                        pass
                    else:
                        print(f"Skipping {file}, already synced.")
                        continue

                processes.append(
                    sync_video(
                        file,
                        stats[file],
                        output_file=output_file,
                        frame_rate=expected_fps,
                        compensate_drift=args.compensate_video_drift,
                    )
                )

        for p in processes:
            p.wait()


if __name__ == "__main__":
    main()
