# RocSync decoding software
This folder contains the computer vision software for detecting and decoding videos/images showing the RocSync PCB.

Pipeline:
1. Find ArUco marker
2. Use marker corners to perform coarse homographic reprojection
3. Find corner LEDs in the reprojected image
4. Use LED coordinates to perform accurate reprojection
5. Read circle and binary counter by thresholding given areas
(6. Fit robust linear model to all extracted timestamps and reject outliers)

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