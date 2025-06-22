import json
import os
import pathlib
import subprocess

import cv2

from rocsync.printer import warnprint


def add_subparser(subparsers):
    parser = subparsers.add_parser(
        "synchronize", help="Synchronize videos with previously analyzed timestamps"
    )
    parser.set_defaults(func=run)
    parser.add_argument(
        "input",
        type=str,
        metavar="PATH",
        help="Path to timestamped videos metadata file (JSON format)",
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


def run(args):
    output_path = pathlib.Path(args.input)
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
