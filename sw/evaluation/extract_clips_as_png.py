from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
import cv2
import tyro
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm
from typing import Optional


@dataclass
class Config:
    
    dataset_folder: str  # root folder containing subfolders with raw and internal calibration videos
    clips_to_extract_json: str
    target_fps: float
    from_raw_camera_time_of_camera: Optional[str] = None # Instead of using the synced timestamps, this allows to work with raw video timestamps of the chosen camera

def main(config: Config) -> None:
    print(f"Processing dataset folder: {config.dataset_folder}")
    
    output_folder_name = "synced_clips_pngs"
    
    time_sync_json_path = os.path.join(config.dataset_folder, "calibration", "time_synchronization_Rec_C_egos.json") # make sure to double check this path
    
    with open(time_sync_json_path, 'r') as f:
        time_sync_data = json.load(f)
        
    # Adjust keys to reflect relative raw video paths
    time_sync_data = {
        os.path.join(*camera.split("/")[-2:]): data
        for camera, data in time_sync_data.items()
    }
    
    if config.from_raw_camera_time_of_camera != None:
        matching_key = next(
            (k for k in time_sync_data if os.path.splitext(os.path.basename(k))[0] == config.from_raw_camera_time_of_camera),
            None
        )

        if not matching_key:
            raise ValueError(f"No match found for camera name: {config.from_raw_camera_time_of_camera}")

        time_defining_camera_time_sync_data = time_sync_data[matching_key]
        clips = _parse_clips_to_extract_json(config.clips_to_extract_json,
                                             from_camera_time=True,
                                             first_frame_time=time_defining_camera_time_sync_data["first_frame"],
                                             speed_factor=time_defining_camera_time_sync_data["speed_factor"])
    else:
        clips = _parse_clips_to_extract_json(config.clips_to_extract_json)
        
    for clip in clips:
        for camera, camera_time_sync_data in time_sync_data.items():
            raw_video_path = os.path.join(config.dataset_folder, camera)
            print(f"Processing {raw_video_path}.")
            if not os.path.exists(raw_video_path):
                print(f"No video found at {raw_video_path}. Will skip this camera.")
                continue
            actual_timestamps = _get_theoretical_timestamps_of_all_frames(camera_time_sync_data)
            frames_to_extract = _get_frames_to_extract(actual_timestamps, clip.start, clip.end)
            
            camera_name = os.path.splitext(os.path.basename(camera))[0]
            output_folder = os.path.join(config.dataset_folder, output_folder_name, f"{clip.start_string_formatted}-{clip.end_string_formatted}", camera_name)
            
            _save_frames_as_png(raw_video_path, frames_to_extract, output_folder)

class Clip:
    
    def __init__(self, start_string: str, end_string: str, from_camera_time: bool = False, first_frame_time: float = None, speed_factor: float = None) -> None:
        if from_camera_time == True:
            start_string, end_string = self._convert_to_real_time(start_string, end_string, first_frame_time, speed_factor)
        self.start = self._parse_timecode_to_milliseconds(start_string)
        self.end = self._parse_timecode_to_milliseconds(end_string)
        self.start_string_formatted = self._parse_timecode_to_string_formatted(start_string)
        self.end_string_formatted = self._parse_timecode_to_string_formatted(end_string)

    def _convert_to_real_time(self, start_string: str, end_string: str, first_frame_time: float, speed_factor: float) -> tuple[str, str]:
        start = self._parse_timecode_to_milliseconds(start_string)
        end = self._parse_timecode_to_milliseconds(end_string)
        
        start_real = int(round(first_frame_time + speed_factor*start))
        end_real = int(round(first_frame_time + speed_factor*end))
        
        start_real_string = self._parse_timecode_to_milliseconds_inverse(start_real)
        end_real_string = self._parse_timecode_to_milliseconds_inverse(end_real)

        return start_real_string, end_real_string

    @staticmethod
    def _parse_timecode_to_milliseconds(timecode: str) -> int:
        dt = datetime.strptime(timecode, "%H:%M:%S.%f")
        delta = (dt.hour * 3600 + dt.minute * 60 + dt.second) * 1000 + dt.microsecond // 1000
        return delta
    
    @staticmethod
    def _parse_timecode_to_milliseconds_inverse(ms: int) -> str:
        seconds, milliseconds = divmod(ms, 1000)
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{secs:02}.{milliseconds:03}"
    
    @staticmethod
    def _parse_timecode_to_string_formatted(timecode: str) -> str:
        dt = datetime.strptime(timecode, "%H:%M:%S.%f")
        timecode_formatted = dt.strftime("%H_%M_%S_%f")
        return timecode_formatted
        
def _parse_clips_to_extract_json(path: str, from_camera_time: bool = False, first_frame_time: float = None, speed_factor: float = None) -> list[Clip]:
    with open(path, 'r', encoding='utf-8') as f:
        clips_raw = json.load(f)

    clips = []
    for clip in clips_raw:
        clip = Clip(clip["start"], clip["end"], from_camera_time=from_camera_time, first_frame_time=first_frame_time, speed_factor=speed_factor)
        clips.append(clip)

    return clips

def _get_frames_to_extract(frame_to_timestamp_map: list, start_time: int, end_time: int) -> list[int]:
    target_timestamps  = np.arange(start_time, end_time, 1000 / config.target_fps) # independent of the camera
    actual_timestamps = np.array(frame_to_timestamp_map)

    frames_to_extract = []

    i = 0
    for t in tqdm(target_timestamps, desc="Matching timestamps"):
        # Move i forward while the next actual timestamp is closer to t
        while i + 1 < len(actual_timestamps) and abs(actual_timestamps[i + 1] - t) < abs(actual_timestamps[i] - t): #chooses always the closest timestamp to the target timestamp
            i += 1
        frames_to_extract.append(i)

    return frames_to_extract

def _get_theoretical_timestamps_of_all_frames(time_sync_data: dict) -> list[float]:
    first_frame = time_sync_data["first_frame"]
    last_frame = time_sync_data["last_frame"]
    n_frames = time_sync_data["n_frames"]

    return [first_frame +  n*(last_frame - first_frame)/n_frames for n in range(n_frames)] #adjusted here the slope such that it takes into account the temporal drift


def _save_frames_as_png(video_path: str, frames: list[int], output_folder: str) -> None:
    '''
    Assumes frames is sorted ascendingly.
    '''
    if not frames:
        return

    os.makedirs(output_folder, exist_ok=True)
    extracted_frames = _extract_frames(video_path, frames)
    _write_frames_to_png_async(extracted_frames, frames, output_folder)

def _extract_frames(video_path: str, frames: list[int]) -> list:
    enumerated_frames = list(enumerate(frames))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    extracted_frames = [None] * len(frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, enumerated_frames[0][1])
    current_frame_idx = enumerated_frames[0][1]
    max_frame_needed = enumerated_frames[-1][1]

    sorted_idx = 0
    n_targets = len(enumerated_frames)

    pbar = tqdm(total=n_targets, desc="Extracting frames")
    
    while current_frame_idx <= max_frame_needed:
        success, frame = cap.read()
        if not success:
            break

        while (sorted_idx < n_targets and
               enumerated_frames[sorted_idx][1] == current_frame_idx):
            i, _ = enumerated_frames[sorted_idx]
            extracted_frames[i] = frame.copy()
            pbar.update(1)
            sorted_idx += 1

        current_frame_idx += 1
        if sorted_idx >= n_targets:
            break

    cap.release()
    pbar.close()
    return extracted_frames

def _write_frames_to_png_async(extracted_frames: list, frames: list[int], output_folder: str) -> None:
    def write_one_frame(i, frame_idx):
        frame = extracted_frames[i]
        filename = os.path.join(output_folder, f"{i:03d}.png")
        if frame is None:
            raise ValueError(f"Error: Could not read frame {frame_idx}")
        cv2.imwrite(filename, frame)

    with ThreadPoolExecutor() as executor:
        futures = []
        for i, frame_idx in enumerate(tqdm(frames, desc="Queueing writes")):
            futures.append(executor.submit(write_one_frame, i, frame_idx))

        for f in tqdm(futures, desc="Writing PNGs"):
            f.result()

if __name__ == "__main__":
    config = tyro.cli(Config)
    main(config)
