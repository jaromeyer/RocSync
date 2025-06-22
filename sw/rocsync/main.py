import argparse
import json
import os
import pathlib
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from rocsync import syncvid
from rocsync.ftk import process_ftk_recording
from rocsync.printer import errprint, succprint, warnprint
from rocsync.video import process_video
from rocsync.vision import CameraType, process_frame


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


def main():
    parser = argparse.ArgumentParser(
        description="Extract timestamps from images and videos showing the RocSync device."
    )
    subparsers = parser.add_subparsers(
        title="subcommands", dest="subcommand", required=True
    )
    syncvid.add_subparser(subparsers)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze videos")
    analyze_parser.set_defaults(func=run)
    analyze_parser.add_argument(
        "path",
        type=str,
        metavar="PATH",
        nargs="+",
        help="path to a video, image, or directory containing videos and/or images",
    )
    analyze_parser.add_argument(
        "-c",
        "--camera_type",
        choices=[e.value for e in CameraType],
        default=CameraType.RGB.value,
        help="specify the type of camera (default: rgb)",
    )
    analyze_parser.add_argument(
        "-s",
        "--stride",
        type=int,
        metavar="N",
        help="scan every N-th frame only (default: same as framerate, only applies to videos)",
    )
    analyze_parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=str,
        metavar="FILE",
        help="JSON file to store results",
    )
    analyze_parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="do not ask for confirmation when processing multiple files",
    )
    analyze_parser.add_argument(
        "--debug",
        type=str,
        metavar="DIRECTORY",
        help="directory to store debug images (very slow)",
    )
    analyze_parser.add_argument(
        "--recurse_in_dir",
        action="store_true",
        help="recursively search for videos and images in directories",
    )
    analyze_parser.add_argument(
        "-e",
        "--export_frames",
        type=str,
        metavar="DIRECTORY",
        help="directory to store all raw frames as PNGs with timestamp (only applies to videos)",
    )

    # Specify two time windows to search for ROCsync
    analyze_parser.add_argument(
        "--start1",
        type=str,
        default=None,
        help="start time of first window to search, in hh:mm:ss.ms format",
    )
    analyze_parser.add_argument(
        "--end1",
        type=str,
        default=None,
        help="end time window of first window to search, in hh:mm:ss.ms format",
    )
    analyze_parser.add_argument(
        "--start2",
        type=str,
        default=None,
        help="start time window of second window to search, in hh:mm:ss.ms format",
    )
    analyze_parser.add_argument(
        "--end2",
        type=str,
        default=None,
        help="end time window of second window to search, in hh:mm:ss.ms format",
    )

    # Dispatch subcommand
    args = parser.parse_args()
    args.func(args)


def run(args):
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
        print(
            f"Found {len(videos)} videos, {len(images)} images, and {len(ftk_recordings)} ftk recordings:"
        )
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
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        with output_path.open("r") as file:
            result = json.load(file)
        print(f"Output file {args.output} already exists, appending results")

    for file in tqdm(
        videos + images + ftk_recordings, desc="Processing files", position=0
    ):
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

        ret = None
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
        elif file in images:
            ret = process_image(file, CameraType(args.camera_type), debug_dir)
        elif file in ftk_recordings:
            ret = process_ftk_recording(file)

        if ret is not None:
            result[str(file)] = ret
        else:
            errprint(f"Error: Unable to time-sync {file}.")

        # Save result to file after every video to avoid data loss
        with output_path.open("w") as f:
            json.dump(result, f, indent=4, cls=NpEncoder)
        print(f"Result written to {args.output}")


if __name__ == "__main__":
    main()
