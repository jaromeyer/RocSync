# Benchmark Tools

Tools for evaluating and annotating the RocSync vision pipeline against validation images.

## Annotation Tool

Interactive GUI for creating ground truth annotations. Runs the pipeline on each image, displays the result with LED overlays, and lets you verify or correct the decoded values. Produces a `ground_truth.json` file for benchmark evaluation.

### Usage

```bash
python -m rocsync.benchmark.annotate [data_dir] [--try-hard] [-o ground_truth.json]
```

- `data_dir`: Directory containing validation images (default: `validation_data/`). Images are collected recursively (`.png`, `.jpg`, `.jpeg`).
- `--try-hard`: Enable relaxed corner detection and ArUco brightness boost.
- `-o`: Output file (default: `<data_dir>/ground_truth.json`).

The tool resumes from the first unannotated image on restart.

### Display

The window shows two panels side-by-side:

- **Left**: Original image
- **Right**: Rectified 640×640 board with color-coded LED overlays
  - Red circle = ON, Blue circle = OFF, Gray circle = not visible
  - If the pipeline failed (e.g., no ArUco detected), a black image is shown with LED positions at their expected locations.

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Enter / Space | Accept annotation, save, advance to next image |
| D / Right arrow | Skip to next image without saving |
| A / Left arrow | Go back to previous image |
| N | Jump to next unannotated image |
| B | Jump to previous unannotated image |
| C / Backspace | Clear annotation for current image (deletes from ground truth, re-runs pipeline) |
| Q | Quit |
| H | Toggle help overlay |
| Esc | Cancel ring LED selection |

### Mouse Interactions (right panel only)

**Corner LEDs** — Click on a corner LED circle to toggle visible/hidden. Drag a corner LED to refine its position.

**Counter LEDs** — Click on a counter LED circle to toggle ON/OFF. Click the counter bounding box (outside individual LEDs) to toggle counter visibility.

**ArUco area** — Click the ArUco region to cycle through marker IDs (0 → 21 → none).

**Ring LEDs** — Two-click protocol: click a ring LED to set the first ON LED (arc start), then click another to set the first OFF LED (arc end). Clicking the same LED for both sets the ring as undecodable. Press `Esc` to cancel after the first click.

### Ground Truth Format

Annotations are saved to a JSON file with the following structure:

```json
{
  "images": {
    "subdir/image.png": {
      "aruco": {"visible": true, "id": 0},
      "corners": [
        {"visible": true, "position": [123.4, 567.8]},
        {"visible": true, "position": [890.1, 567.8]},
        {"visible": false},
        {"visible": true, "position": [123.4, 890.1]}
      ],
      "homography": [[...], [...], [...]],
      "counter": {"visible": true, "value": 42},
      "ring": {"start": 10, "end": 55}
    }
  }
}
```

Fields:
- `aruco.visible` / `aruco.id`: Whether the ArUco marker is visible and its detected ID (omitted when not visible).
- `corners[i].visible` / `corners[i].position`: Per-corner-LED visibility and position in original image coordinates (position omitted when not visible).
- `homography`: 3×3 matrix mapping original image → rectified board coordinates.
- `counter.visible` / `counter.value`: Whether the binary counter is readable and its decoded value (omitted when not visible).
- `ring.start` / `ring.end`: First ON and first OFF LED indices (0–99), half-open interval `[start, end)`. `start == end` means the ring is undecodable.

## Validation Benchmark

Runs the pipeline on all images in a directory and saves per-image results in a structure mirroring the ground truth format, plus per-step timing.

```bash
python -m rocsync.benchmark.validate [data_dir] [--try-hard] [-o results.json] [--debug DIR]
```

- `data_dir`: Directory containing validation images (default: `validation_data/`).
- `--try-hard`: Enable try_hard mode for the pipeline.
- `-o`: Output JSON file (default: `benchmark_results.json`).
- `--debug`: Directory for debug images.

Results JSON contains a `config` section and an `images` section with per-image aruco, corners, counter, ring, timestamp, success flag, and timing breakdown.

## Benchmark Evaluation

Compares benchmark results against ground truth annotations. Reports per-step detection metrics (TPR, FPR, precision, F1), value accuracy, and position errors.

```bash
python -m rocsync.benchmark.evaluate [paths...] [-g ground_truth.json] [-t]
```

- `paths`: Directory with benchmark `.json` files, or one or more explicit `.json` filepaths (default: `output/benchmark/`).
- `-g`: Path to ground truth JSON (default: `validation_data/ground_truth.json`).
- `-t`: Include per-step timing statistics (all/positive/negative subsets).

Metrics are computed per pipeline step:
- **ArUco**: detection rates + corner pixel error in image space
- **Corners**: detection rates in image space and board space + pixel errors in both spaces
- **Counter**: detection rates + value accuracy
- **Ring**: detection rates + start/end value accuracy
- **Overall**: timestamp detection + exact match accuracy + start/end/exposure error statistics