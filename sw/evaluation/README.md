# Multi-Camera Sync Toolkit
## 📂 Folder Setup

Your dataset folder should look like this:

```
dataset_folder/
├── raw_videos/
│   ├── camera1_raw.mp4
│   ├── camera2_raw.mp4
│   ├── camera3_raw.mp4
│   └── ...
│
└── time sync/
    ├── clips_config.json
    ├── time_synchronization.json
    ├── extract_synced_videos.py
    ├── extract_clips_as_png.py
    ├── check_time_sync_all.py
    └── ...
```

### Description:

* **`raw_videos/`** – contains all raw camera recordings (e.g., `.mp4`, `.MOV`).
* **`time sync/`** – contains synchronization files:

  * `clips_config.json`: defines which clips to extract.
  * `time_synchronization_*.json`: contains timing information for each camera.

---

## ✅ Check synchronization of all clips

Once you have a finished `time_synchronization_*.json` file, check if all videos have been synchronized correctly by the RocSync script and/or the manual synchronization. Check a moment near the beginning of the Recording and a moment near the end of the Recording to make sure there is no drift. Run this before synchronizing huge datasets, it will save you a lot of time down the road.

Run the script:
```bash
python path/to/check_time_sync_all.py \
    --time HH:MM:SS.mmm \
    --from-camera "cameraX"
```

### Arguments:

* `--time`
  The time of the reference video you want to display in this format: HH:MM:SS.mmm.

* `--from-camera`
  Name of the reference camera (e.g., `"camera1_raw"` or `"gopro7_raw"`) — this camera’s timing is used as the reference for synchronization.

---

## 🎥 Extract Synced Videos

This guide explains how to organize your dataset and run the `extract_synced_videos.py` script to generate **time-synchronized video clips** from multiple cameras.


### 🧭 Usage

Run the extraction script from your terminal:

```bash
python extract_synced_videos.py 
```

### Optional Arguments:

* `--dataset-folder`
  Path to your dataset folder (the one containing `raw_videos/` and `time sync/`). Not necessary if you follow the **folder setup** above.

* `--target-fps`
  Desired output frame rate (e.g., 30).

* `--from-camera`
  Name of the reference camera (e.g., `"camera1_raw"` or `"gopro7_raw"`) — this camera’s timing is used as the reference for synchronization.

* `--time-sync-json`
  Path to your `time_synchronization_*.json` file

* `--clips-json`
  Path to the `clips_config.json` file

* `--ignore-overlap`
  If set, the script will not restrict synchronization to the time interval shared by **all** files (the global overlap).  
  Use this if some videos do **not** cover the requested timestamp/time range (e.g., some cameras started later or stopped earlier).
### Output:
* Folder called synced_videos with the syncronized clips


## 📷 Extract Synced Frames
This guide explains how to organize your dataset and run the `extract_clips_as_png.py` script to generate **time-synchronized pictures** from multiple cameras.


### 🧭 Usage

Run the script from your terminal:

```bash
python extract_clips_as_png.py 
```

### Optional Arguments:

* `--dataset-folder`
  Path to your dataset folder (the one containing `raw_videos/` and `time sync/`). Not necessary if you follow the **folder setup** above.

* `--target-fps`
  Desired output frame rate (e.g., 30).

* `--from-camera`
  Name of the reference camera (e.g., `"camera1_raw"` or `"gopro7_raw"`) — this camera’s timing is used as the reference for synchronization.

* `--time-sync-json`
  Path to your `time_synchronization_*.json` file

* `--clips-json`
  Path to the `clips_config.json` file

* `--only-specified`
  If set, only produce PNGs for the camera given by `--from-camera`.
  

### Output:
* Folder called synced_clips_png with the synchronized png frames


