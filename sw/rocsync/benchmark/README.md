# Benchmark Tools

Tools for evaluating and annotating the RocSync vision pipeline against validation images.

## Annotation Tool

Interactive GUI for creating ground truth annotations. Runs the pipeline on each image, displays the result with LED overlays, and lets you verify or correct the decoded values.

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
- **Right**: Rectified 640x640 board with color-coded LED overlays
  - Red circle = ON, Blue circle = OFF, Gray circle = not visible
  - If the pipeline failed (e.g., no ArUco detected), a black image is shown with LED positions at their expected locations.

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Enter / Space | Accept annotation, save, advance to next image |
| X | Mark image as undecodable (information not recoverable), save, advance |
| S | Skip without saving |
| B | Go back to previous image |
| Q | Save and quit |
| A | Toggle ArUco marker overlay (renders the marker with the annotated ID for visual comparison) |
| R | Toggle ring LED overlay |
| C | Toggle counter visibility |
| H | Toggle help overlay |
| + / - | Increment / decrement the annotated ArUco marker ID |
| Esc | Cancel ring LED selection |

### Mouse Interactions (right panel only)

**Corner LEDs** — Click on a corner LED circle to cycle its state: ON -> OFF -> not visible -> ON. Drag a corner LED to refine its position.

**Counter LEDs** — Click on a counter LED circle to toggle ON/OFF. The counter value is recomputed automatically. Use `C` to toggle counter visibility for the entire row.

**Ring LEDs** — Two-click protocol: click a ring LED to set the arc start, then click another to set the arc end. Press `Esc` to cancel after the first click.

### Ground Truth Format

Annotations are saved to a JSON file with the following structure:

```json
{
  "images": {
    "subdir/image.png": {
      "aruco": {"visible": true, "id": 0},
      "corners": [
        {"visible": true, "state": true, "position": [51, 51]},
        {"visible": true, "state": false, "position": [588, 51]},
        {"visible": false, "state": false, "position": [588, 588]},
        {"visible": true, "state": true, "position": [51, 588]}
      ],
      "counter": {"visible": true, "value": 42},
      "ring": {"visible": true, "start": 10, "end": 55},
      "status": "annotated"
    }
  }
}
```

Fields:
- `aruco.visible` / `aruco.id`: Whether the ArUco marker is visible and its detected ID.
- `corners[i]`: Per-corner-LED visibility, ON/OFF state, and position in rectified board coordinates.
- `counter.visible` / `counter.value`: Whether the binary counter is readable and its decoded value.
- `ring.visible` / `ring.start` / `ring.end`: Whether the ring is readable and the start/end LED indices (0-99) of the illuminated arc.
- `status`: `"annotated"` (verified by human) or `"undecodable"` (information objectively lost, e.g., too low resolution or RocSync not visible).

## Validation Benchmark

Runs the pipeline on all images in a directory and saves per-image timing and detection statistics.

```bash
python -m rocsync.benchmark.validate [data_dir] [--try-hard] [-o results.json] [--debug DIR]
```

Results are saved as JSON with per-step timing breakdowns and aggregate statistics.

## Benchmark Evaluation

Compares multiple benchmark JSON files side-by-side.

```bash
python -m rocsync.benchmark.evaluate [benchmark_dir]
```

Loads all `.json` files from the given directory (default: `output/benchmark/`) and prints comparison tables for detection rates, per-step timing, and per-image disagreements.