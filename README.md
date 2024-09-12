# RocSync: temporal multi-camera synchronization

![RocSync_front](https://github.com/user-attachments/assets/09734239-36fa-4ac8-a8b1-2877538088eb)

RocSync is an open-source solution for millisecond-accurate temporal synchronization of multiple cameras. The core component is a 25x25 cm PCB featuring 100 dual LEDs (each consisting of a visible red LED and an infrared LED) arranged in a circle and 16 additional dual LEDs to display a binary counter. There is always exactly one illuminated LED in the circle and it advances one step every millisecond, while the binary display counts the number of complete rotations. The second part is a Python application that decodes images and videos of the device and provides exact timestamps (relative to the RocSync device's internal clock) for every frame. It supports synchronization of both standard RGB cameras (e.g., Kinect, GoPro, Canon) and specialized IR tracking cameras, such as the [Atracsys fusionTrack 500](https://atracsys.com/fusiontrack-500/), making it ideal for research and applications requiring high-precision multi-camera setups.

This repository is organized into three main sections. See the respective READMEs for more details:
- `hw/`: KiCad files for the PCB design & CAD models
- `fw/`: PlatformIO project for the microcontroller firmware
- `sw/`: Python source code for computer vision processing
