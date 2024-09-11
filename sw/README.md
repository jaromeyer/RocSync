# RocSync decoding software
This folder contains the computer vision software for detecting and decoding videos/images showing the RocSync PCB.

### Pipeline:
1. Find ArUco marker
2. Use marker corners to perform coarse homographic reprojection
3. Find corner LEDs in the reprojected image
4. Use LED coordinates to perform accurate reprojection
5. Read circle and binary counter by thresholding given areas
6. Fit robust linear model to all extracted timestamps and reject outliers

## Usage
```
$ python ./rocsync.py -h
usage: rocsync.py [-h] [-c {rgb,ir}] [-s N] [-e DIRECTORY] [-o FILE] [--debug DIRECTORY] PATH [PATH ...]

Extract timestamps from images and videos showing the RocSync device.

positional arguments:
  PATH                  path to a video, image, or directory containing videos and/or images

options:
  -h, --help            show this help message and exit
  -c {rgb,ir}, --camera_type {rgb,ir}
                        specify the type of camera (default: rgb)
  -s N, --stride N      scan every N-th frame only (default: framerate, only applies to videos)
  -e DIRECTORY, --export_frames DIRECTORY
                        export all raw frames as PNGs with timestamp (only applies to videos)
  -o FILE, --output FILE
                        JSON file to store results
  --debug DIRECTORY     directory to store debug images (very slow)
```

## Example output
```
python ./rocsync.py ./examples/h10.MP4 ./examples
Working on ./examples/h10.MP4
Analyzing frames: 100%|███████████████████████████████████████████████████████████| 6520/6520 [02:21<00:00, 45.93it/s] 
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
Processing files: 100%|████████████████████████████████████████████████████████████████| 1/1 [02:22<00:00, 142.04s/it] 
```

## FFmpeg
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