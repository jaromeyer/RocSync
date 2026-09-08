import argparse
import json
import os
import subprocess

import cv2
from rocsync.printer import warnprint


def main():
    parser = argparse.ArgumentParser(
        description="Create aligned video files based on previously computed synchronization metadata."
    )
    parser.add_argument(
        "sync_file",
        type=str,
        metavar="sync.json",
        help="Path to JSON file containing video synchronization metadata with timestamps and frame offsets",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="synced",
        help="Output directory where synchronized videos will be saved (default: synced)",
    )
    parser.add_argument(
        "--compensate-drift",
        action="store_true",
        help="Enable video drift compensation via re-encoding (significantly slower but more accurate)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Target frame rate for synchronized videos. If not specified, uses the source frame rate of the first video",
    )

    args = parser.parse_args()

    with open(args.sync_file, "r") as file:
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
            print(f"video_folder: {video_folder}, video_name: {video_name}")
            output_folder = os.path.join(video_folder, args.output_dir)
            output_file = os.path.join(output_folder, f"{video_name}.mp4")
            os.makedirs(output_folder, exist_ok=True)
            print(f"Output file will be saved to {output_file}. Input file: {file}")

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
                    compensate_drift=args.compensate_drift,
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


if __name__ == "__main__":
    main()
