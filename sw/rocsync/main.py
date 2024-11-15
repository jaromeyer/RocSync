import argparse
import json
import os

import cv2
import numpy as np
from tqdm import tqdm

from rocsync.printer import *
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
    existing_dirs = os.listdir(parent_dir)
    if name not in existing_dirs:
        debug_dir = f"{parent_dir}/{name}"
    else:
        counter = 2
        while f"{name} ({counter})" in existing_dirs:
            counter += 1
        debug_dir = f"{parent_dir}/{name} ({counter})"
    os.makedirs(debug_dir)
    return debug_dir


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
        type=str,
        metavar="FILE",
        help="JSON file to store results",
    )
    parser.add_argument(
        "--debug",
        type=str,
        metavar="DIRECTORY",
        help="directory to store debug images (very slow)",
    )

    args = parser.parse_args()

    files = set()
    for path in args.path:
        if os.path.isdir(path):
            # walk dir recursively
            for root, _, dir_files in os.walk(path):
                for file in dir_files:
                    file_path = os.path.join(root, file)
                    files.add(file_path)
        elif os.path.isfile(path):
            files.add(path)
        else:
            errprint(f"Invalid path: {path}")
            return

    videos = [f for f in files if f.lower().endswith((".mp4", ".avi", ".mov"))]
    images = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    if len(videos) + len(images) > 1:
        print(f"Found {len(videos)} videos and {len(images)} images:")
        for file in videos + images:
            print(f"    {file}")
        while True:
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
    for file in tqdm(videos + images, desc="Processing files", position=0):
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
            result[file] = process_video(
                file, CameraType(args.camera_type), export_dir, args.stride, debug_dir
            ).to_dict()
        elif file in images:
            result[file] = process_image(file, CameraType(args.camera_type), debug_dir)

    # TODO: fix missing parent dir path
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as file:
            json.dump(result, file, indent=4, cls=NpEncoder)
        print(f"Result written to {args.output}")


if __name__ == "__main__":
    main()
