# RocSync decoding software

This folder contains the Python application for detecting and decoding the RocSync device in videos and images where it is visible.

### How it works:

1. **Find ArUco marker**: Detect the ArUco marker to determine the approximate position and orientation.
2. **Coarse homographic reprojection**: Use the detected marker's corners to perform a coarse homographic reprojection of the image.
3. **Locate corner LEDs**: Identify the corner LEDs in the reprojected image.
4. **Accurate reprojection**: Use the corner LEDs to refine the reprojection for higher accuracy.
5. **Decode LEDs**: Decode the circle and binary counter LEDs by thresholding their general areas to obtain an exact timestamp.
6. **Timestamp fitting**: If the input was a video, perform robust linear regression on all extracted timestamps to reject outliers and estimate timestamps for all frames.

## Installation
To install RocSync as a Python module, run the following commands:

```bash
git clone https://github.com/jaromeyer/RocSync.git
pip install ./RocSync/sw
```

## Usage
```
$ rocsync -h
usage: rocsync [-h] [-c {rgb,ir}] [-s N] [-e DIRECTORY] [-o FILE] [--debug DIRECTORY] PATH [PATH ...]

Extract timestamps from images and videos showing the RocSync device.

positional arguments:
  PATH                  path to a video, image, or directory containing videos and/or images

options:
  -h, --help            show this help message and exit
  -c, --camera_type {rgb,ir}
                        specify the type of camera (default: rgb)
  -s, --stride N        scan every N-th frame only (default: same as framerate, only applies to videos)
  -e, --export_frames DIRECTORY
                        directory to store all raw frames as PNGs with timestamp (only applies to videos)
  -o, --output FILE     JSON file to store results
  --debug DIRECTORY     directory to store debug images (very slow)
  ```


## Example
```
$ rocsync ./examples/h10.MP4 ./examples
Working on ./examples/h10.MP4
Analyzing frames: 100%|████████████████████| 6520/6520 [02:21<00:00, 45.93it/s]
Number of considered frames:                             1888
Number of rejected outliers:                                0
R2 score:                                              1.0000
RMSE:                                                 0.44 ms
First frame:                                       -4333.4 ms
Last frame:                                        22858.4 ms
Expected duration (fps):              27189.7 ms (239.76 fps)
Actual duration (fps):                27191.7 ms (239.74 fps)
Delta (actual - expected)              2.08 ms (0.010% speed)
Exposure time (mean/min/max):               3.97/3.00/5.00 ms
-------------------------------------------------------------
Processing files: 100%|█████████████████████████| 1/1 [02:22<00:00, 142.04s/it]
```

### FFmpeg
To visually inspect the synchronization you can use FFmpeg to combine the aligned videos side-by-side:
```bash
ffmpeg \
	-ss 4.2803 -i video1.mp4 \
	-ss -8.1024 -i video2.mp4 \
	-filter_complex "[0:v]setpts=PTS*1.000020938577059[v0];[1:v]setpts=PTS*1.000083866934668[v1];[v0][v1]hstack=inputs=2" \
	-c:v h264_nvenc \
	-preset p1 \
	-r 30 \
	synchronized.mp4
```

## TODO
- [ ] Write debug images in separate thread
- [ ] More filters for IR (e.g., uniform corner distance)