#!/usr/bin/env python3
"""Benchmark rocsync pipeline on validation data.

Runs process_frame on all images, collects per-image results in a structure
mirroring the ground truth format (aruco, corners, counter, ring), plus
per-step timing. Saves results as JSON without aggregate statistics — those
are computed by the evaluate tool.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

from rocsync.vision import CameraType, period, process_frame

STEP_ORDER = [
    "aruco_detection",
    "corner_detection",
    "fine_rectification",
    "counter_reading",
    "ring_reading",
]


def collect_images(root_dir):
    """Collect all image files recursively, sorted by path."""
    exts = {".png", ".jpg", ".jpeg"}
    return sorted(
        p for p in Path(root_dir).rglob("*") if p.suffix.lower() in exts
    )


def build_result(stats):
    """Extract a ground-truth-compatible result dict from pipeline stats.

    Returns a dict with keys: aruco, corners, counter, ring, timestamp.
    Structure mirrors ground_truth.json to simplify comparison.
    """
    steps = stats.get("steps", {})

    # -- ArUco --
    aruco_step = steps.get("aruco_detection", {})
    aruco_visible = aruco_step.get("success", False)
    aruco = {
        "visible": aruco_visible,
        "id": stats.get("aruco_id") if aruco_visible else None,
        "corners": stats.get("aruco_corners") if aruco_visible else None,
    }

    # -- Corners --
    corner_positions = stats.get("corner_positions")
    if corner_positions is None:
        corner_positions = [None] * 4
    corners = [{"visible": pos is not None, "position": pos} for pos in corner_positions]

    # -- Counter --
    counter_step = steps.get("counter_reading", {})
    counter_value = counter_step.get("value")
    counter = {
        "visible": counter_value is not None,
        "value": counter_value,
    }

    # -- Ring --
    ring_step = steps.get("ring_reading", {})
    ring_visible = ring_step.get("success", False)
    timestamp = stats.get("timestamp")

    if ring_visible and timestamp is not None:
        # read_ring returns inclusive end (last ON LED); convert to half-open
        # (first OFF LED) to match ground truth convention.
        ring = {
            "start": timestamp[0] % period,
            "end": (timestamp[1] + 1) % period,
        }
    else:
        ring = {"start": 0, "end": 0}

    return {
        "aruco": aruco,
        "corners": corners,
        "counter": counter,
        "ring": ring,
        "timestamp": timestamp,
    }


def build_timing(stats):
    """Extract per-step timing from pipeline stats."""
    steps = stats.get("steps", {})
    timing = {}
    for step in STEP_ORDER:
        if step in steps:
            timing[f"{step}_ms"] = steps[step]["time_ms"]
    timing["total_ms"] = stats.get("total_time_ms")
    return timing


def run_benchmark(data_dir, images, try_hard=False, debug_dir=None):
    """Run pipeline on all images, returning results dict keyed by relative path."""
    results = {}
    for i, path in enumerate(tqdm(images)):
        image = cv2.imread(str(path))
        if image is None:
            print(f"  WARNING: could not read {path}", file=sys.stderr)
            continue

        stats = {}
        success, timestamp = process_frame(
            image, CameraType.RGB, i,
            debug_dir=debug_dir,
            try_hard=try_hard,
            stats=stats,
        )

        rel_path = str(path.relative_to(data_dir))
        result = build_result(stats)
        timing = build_timing(stats)

        results[rel_path] = {
            **result,
            "success": success and timestamp is not None,
            "timing": timing,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark rocsync on validation data")
    parser.add_argument("data_dir", nargs="?", default="validation_data",
                        help="Path to validation data directory (default: validation_data)")
    parser.add_argument("--try-hard", action="store_true",
                        help="Enable try_hard mode for corner detection")
    parser.add_argument("-o", "--output", default="benchmark_results.json",
                        help="Output JSON file (default: benchmark_results.json)")
    parser.add_argument("--debug", default=None,
                        help="Directory for debug images")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    images = collect_images(data_dir)
    if not images:
        print(f"No images found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    if args.debug:
        Path(args.debug).mkdir(parents=True, exist_ok=True)

    print(f"Found {len(images)} images (try_hard={'on' if args.try_hard else 'off'})")
    results = run_benchmark(data_dir, images, try_hard=args.try_hard, debug_dir=args.debug)

    n_success = sum(1 for r in results.values() if r["success"])
    print(f"Detection rate: {n_success}/{len(results)} ({n_success/len(results):.1%})")

    output = {
        "config": {"try_hard": args.try_hard, "data_dir": str(data_dir)},
        "images": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
