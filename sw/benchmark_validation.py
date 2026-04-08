#!/usr/bin/env python3
"""Benchmark rocsync pipeline on validation data.

Runs process_frame on all images, collects per-step timing and detection
statistics, saves results as JSON, and prints a summary table.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from rocsync.vision import CameraType, process_frame

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


def run_benchmark(images, try_hard=False, debug_dir=None):
    """Run pipeline on all images, returning per-image stats dicts."""
    results = []
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
        stats["image"] = str(path)
        results.append(stats)

    return results


def describe(values):
    """Return dict with mean, std, min, max for a list of numbers."""
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    a = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "n": len(values),
    }


def aggregate_stats(results):
    """Aggregate per-image results into summary statistics."""
    success_results = [r for r in results if r.get("success") and r.get("timestamp")]
    failure_results = [r for r in results if not (r.get("success") and r.get("timestamp"))]

    summary = {
        "total_images": len(results),
        "total_success": len(success_results),
        "total_failure": len(failure_results),
        "detection_rate": len(success_results) / len(results) if results else 0,
    }

    # Aggregate per-step and total timing/detection for each group
    for group_name, group in [("all", results), ("success", success_results), ("failure", failure_results)]:
        group_stats = {}

        # Per-step stats
        for step in STEP_ORDER:
            times = [r["steps"][step]["time_ms"] for r in group if step in r.get("steps", {})]
            group_stats[f"{step}_time"] = describe(times)

            if step in ("aruco_detection", "corner_detection", "ring_reading"):
                successes = [1 if r["steps"][step].get("success") else 0 for r in group if step in r.get("steps", {})]
                group_stats[f"{step}_rate"] = describe(successes)

            if step == "corner_detection":
                counts = [r["steps"][step].get("count", 0) for r in group if step in r.get("steps", {})]
                group_stats[f"{step}_count"] = describe(counts)

            if step == "counter_reading":
                values = [r["steps"][step].get("value", 0) for r in group if step in r.get("steps", {})]
                group_stats[f"{step}_value"] = describe(values)

        # Total timing
        total_times = [r["total_time_ms"] for r in group if "total_time_ms" in r]
        group_stats["total_time"] = describe(total_times)

        summary[group_name] = group_stats

    return summary


def fmt(val, width=8):
    """Format a numeric value or None for table display."""
    if val is None:
        return "-".rjust(width)
    if isinstance(val, float):
        return f"{val:.2f}".rjust(width)
    return str(val).rjust(width)


def print_table(summary):
    """Print formatted summary table to stdout."""
    print(f"\n{'=' * 80}")
    print(f"  BENCHMARK SUMMARY — {summary['total_images']} images")
    print(f"  Detection rate: {summary['total_success']}/{summary['total_images']}"
          f" ({summary['detection_rate']:.1%})")
    print(f"{'=' * 80}")

    for group_name in ["all", "success", "failure"]:
        group = summary[group_name]
        n = group.get("total_time", {}).get("n", 0)
        if n == 0:
            continue

        label = {"all": "ALL IMAGES", "success": "SUCCESS (timestamp extracted)", "failure": "FAILURE"}[group_name]
        print(f"\n  {label} (n={n})")
        print(f"  {'-' * 76}")

        # Timing table
        print(f"  {'Timing (ms)':<28} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'N':>6}")
        print(f"  {'-' * 28} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 6}")
        for step in STEP_ORDER:
            key = f"{step}_time"
            if key in group:
                s = group[key]
                print(f"  {step:<28} {fmt(s['mean'])} {fmt(s['std'])} {fmt(s['min'])} {fmt(s['max'])} {fmt(s['n'], 6)}")
        s = group["total_time"]
        print(f"  {'TOTAL':<28} {fmt(s['mean'])} {fmt(s['std'])} {fmt(s['min'])} {fmt(s['max'])} {fmt(s['n'], 6)}")

        # Detection table
        det_rows = []
        for step in STEP_ORDER:
            rate_key = f"{step}_rate"
            count_key = f"{step}_count"
            if rate_key in group:
                s = group[rate_key]
                det_rows.append((step + " rate", s))
            if count_key in group:
                s = group[count_key]
                det_rows.append((step + " count", s))

        if det_rows:
            print()
            print(f"  {'Detection':<28} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'N':>6}")
            print(f"  {'-' * 28} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 6}")
            for label, s in det_rows:
                print(f"  {label:<28} {fmt(s['mean'])} {fmt(s['std'])} {fmt(s['min'])} {fmt(s['max'])} {fmt(s['n'], 6)}")


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

    images = collect_images(args.data_dir)
    if not images:
        print(f"No images found in {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    if args.debug:
        Path(args.debug).mkdir(parents=True, exist_ok=True)

    print(f"Found {len(images)} images (try_hard={'on' if args.try_hard else 'off'})")
    results = run_benchmark(images, try_hard=args.try_hard, debug_dir=args.debug)

    summary = aggregate_stats(results)

    output = {"config": {"try_hard": args.try_hard, "data_dir": str(args.data_dir)},
              "per_image": results, "summary": summary}
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")

    print_table(summary)


if __name__ == "__main__":
    main()