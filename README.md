# 🎥 RocSync
 
**Millisecond-Accurate Temporal Synchronization for Heterogeneous Camera Systems**
 
[![Paper](https://img.shields.io/badge/Paper-MDPI%20Sensors-blue?style=flat-square)](https://www.mdpi.com/3715386)
[![ETH Zurich](https://img.shields.io/badge/Institution-ETH%20Zurich-red?style=flat-square)](https://ethz.ch)
[![Balgrist](https://img.shields.io/badge/Institution-Balgrist%20ROCS-green?style=flat-square)](https://rocs.balgrist.ch/en/)
[![License](https://img.shields.io/badge/License-MPL--2.0-yellow?style=flat-square)](LICENSE)
 
> *Jaro Meyer¹, Frédéric Giraud², Joschua Wüthrich¹, Marc Pollefeys¹, Philipp Fürnstahl², Lilian Calvet²*
>
> ¹ Department of Computer Science, ETH Zurich, Switzerland\
> ² Research in Orthopedic Computer Science, Balgrist University Hospital, University of Zurich, Switzerland

![RocSync Front](https://github.com/user-attachments/assets/09734239-36fa-4ac8-a8b1-2877538088eb)

**RocSync**  is an open-source solution for millisecond-accurate temporal synchronization of multiple cameras. The core component is a 25x25 cm PCB featuring 100 dual LEDs (each consisting of a visible red LED and an infrared LED) arranged in a circle and 16 additional dual LEDs to display a binary counter. There is always exactly one illuminated LED in the circle and it advances one step every millisecond, while the binary display counts the number of complete rotations. The second part is a Python application that decodes images and videos of the device and provides exact timestamps (relative to the RocSync device's internal clock) for every frame. It supports synchronization of both standard RGB cameras (e.g., Kinect, GoPro, Canon) and specialized IR tracking cameras, such as the [Atracsys fusionTrack 500](https://atracsys.com/fusiontrack-500/), making it ideal for research and applications requiring high-precision multi-camera setups.

## Repository Structure

This repository contains three main sections, each with its own README:

- **`hw/`**: KiCad files for PCB design and CAD models.
- **`fw/`**: PlatformIO project for microcontroller firmware.
- **`sw/`**: Python source code for computer vision processing.

## Getting Started

For detailed instructions on setting up and using RocSync, please refer to the README files in the respective sections.

## Citation
If you use RocSync in your research, please cite:
```
@Article{s26031036,
AUTHOR = {Meyer, Jaro and Giraud, Frédéric and Wüthrich, Joschua and Pollefeys, Marc and Fürnstahl, Philipp and Calvet, Lilian},
TITLE = {RocSync: Millisecond-Accurate Temporal Synchronization for Heterogeneous Camera Systems},
JOURNAL = {Sensors},
VOLUME = {26},
YEAR = {2026},
NUMBER = {3},
ARTICLE-NUMBER = {1036},
URL = {https://www.mdpi.com/1424-8220/26/3/1036},
PubMedID = {41682551},
ISSN = {1424-8220},
DOI = {10.3390/s26031036}
}
```

## Credits

- **Frédéric Giraud**: Creator of the initial prototype.
- **Lilian Calvet**: Conceptual designer.

Feel free to explore, contribute, and use RocSync in your projects!
