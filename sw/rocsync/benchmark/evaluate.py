#!/usr/bin/env python3
"""Compare multiple benchmark JSON files side-by-side.

Reads all .json files from output/benchmark/ (or a custom directory),
using the file stem as the method name, and prints comparison tables.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


STEP_ORDER = [
    "aruco_detection",
    "corner_detection",
    "fine_rectification",
    "counter_reading",
    "ring_reading",
]


def load_benchmarks(directory):
    """Load all benchmark JSON files, keyed by stem name."""
    benchmarks = {}
    for path in sorted(Path(directory).glob("*.json")):
        with open(path) as f:
            benchmarks[path.stem] = json.load(f)
    return benchmarks


def per_image_comparison(benchmarks):
    """Build per-image success/timestamp comparison across methods.

    Returns list of dicts, one per image, with per-method results.
    """
    # Use the first benchmark to get the canonical image list
    first = next(iter(benchmarks.values()))
    image_keys = [r["image"] for r in first["per_image"]]

    # Index each benchmark's per_image by image path
    indexed = {}
    for method, data in benchmarks.items():
        indexed[method] = {r["image"]: r for r in data["per_image"]}

    rows = []
    for img in image_keys:
        row = {"image": img}
        for method in benchmarks:
            r = indexed[method].get(img)
            if r:
                row[method] = {
                    "success": r.get("success", False) and r.get("timestamp") is not None,
                    "timestamp": r.get("timestamp"),
                    "total_time_ms": r.get("total_time_ms"),
                    "steps": r.get("steps", {}),
                }
            else:
                row[method] = {"success": False, "timestamp": None, "total_time_ms": None, "steps": {}}
        rows.append(row)
    return rows


def fmt(val, width=10):
    if val is None:
        return "-".rjust(width)
    if isinstance(val, float):
        return f"{val:.2f}".rjust(width)
    return str(val).rjust(width)


def pct(val, width=10):
    if val is None:
        return "-".rjust(width)
    return f"{val:.1%}".rjust(width)


def print_header(methods, col_width=12):
    label_col = 30
    print(f"  {'':>{label_col}}", end="")
    for m in methods:
        print(f"  {m:>{col_width}}", end="")
    print()
    print(f"  {'':>{label_col}}", end="")
    for _ in methods:
        print(f"  {'-' * col_width}", end="")
    print()


def print_overview(benchmarks):
    """Print high-level detection rate comparison."""
    methods = list(benchmarks.keys())
    col_w = max(12, max(len(m) for m in methods) + 2)

    print(f"\n{'=' * 80}")
    print(f"  BENCHMARK COMPARISON — {len(methods)} methods")
    print(f"{'=' * 80}")

    print(f"\n  Detection Rate Overview")
    print_header(methods, col_w)

    label_col = 30
    n = benchmarks[methods[0]]["summary"]["total_images"]
    print(f"  {'total images':>{label_col}}", end="")
    for m in methods:
        print(f"  {benchmarks[m]['summary']['total_images']:>{col_w}}", end="")
    print()

    print(f"  {'successes':>{label_col}}", end="")
    for m in methods:
        print(f"  {benchmarks[m]['summary']['total_success']:>{col_w}}", end="")
    print()

    print(f"  {'detection rate':>{label_col}}", end="")
    for m in methods:
        print(f"  {pct(benchmarks[m]['summary']['detection_rate'], col_w)}", end="")
    print()


def print_step_comparison(benchmarks, group="all"):
    """Print per-step timing and detection comparison for a given group."""
    methods = list(benchmarks.keys())
    col_w = max(12, max(len(m) for m in methods) + 2)
    label_col = 30

    group_label = {"all": "ALL IMAGES", "success": "SUCCESS ONLY", "failure": "FAILURE ONLY"}[group]
    n_per_method = {m: benchmarks[m]["summary"][group].get("total_time", {}).get("n", 0) for m in methods}
    print(f"\n  {group_label} — Timing mean (ms)")
    print_header(methods, col_w)

    for step in STEP_ORDER:
        key = f"{step}_time"
        print(f"  {step:>{label_col}}", end="")
        for m in methods:
            val = benchmarks[m]["summary"][group].get(key, {}).get("mean")
            print(f"  {fmt(val, col_w)}", end="")
        print()

    print(f"  {'TOTAL':>{label_col}}", end="")
    for m in methods:
        val = benchmarks[m]["summary"][group].get("total_time", {}).get("mean")
        print(f"  {fmt(val, col_w)}", end="")
    print()

    # Detection rates per step
    det_steps = ["aruco_detection", "corner_detection", "ring_reading"]
    print(f"\n  {group_label} — Detection rates (mean)")
    print_header(methods, col_w)

    for step in det_steps:
        key = f"{step}_rate"
        print(f"  {step:>{label_col}}", end="")
        for m in methods:
            val = benchmarks[m]["summary"][group].get(key, {}).get("mean")
            print(f"  {pct(val, col_w)}", end="")
        print()

    # Corner detection count
    print(f"\n  {group_label} — Corner detection count (mean)")
    print_header(methods, col_w)
    print(f"  {'corner_detection count':>{label_col}}", end="")
    for m in methods:
        val = benchmarks[m]["summary"][group].get("corner_detection_count", {}).get("mean")
        print(f"  {fmt(val, col_w)}", end="")
    print()


def print_per_image_diff(benchmarks):
    """Print images where methods disagree on success."""
    methods = list(benchmarks.keys())
    rows = per_image_comparison(benchmarks)

    disagree = [r for r in rows if len(set(r[m]["success"] for m in methods)) > 1]
    if not disagree:
        print(f"\n  All methods agree on every image.")
        return

    col_w = max(12, max(len(m) for m in methods) + 2)
    print(f"\n  Per-image disagreements ({len(disagree)} images)")
    print(f"  {'-' * 80}")

    # Header
    img_col = 45
    print(f"  {'image':<{img_col}}", end="")
    for m in methods:
        print(f"  {m:>{col_w}}", end="")
    print()
    print(f"  {'-' * img_col}", end="")
    for _ in methods:
        print(f"  {'-' * col_w}", end="")
    print()

    for r in disagree:
        img_short = Path(r["image"]).name
        print(f"  {img_short:<{img_col}}", end="")
        for m in methods:
            if r[m]["success"]:
                ts = r[m]["timestamp"]
                label = f"{ts[0]}-{ts[1]}" if ts else "ok"
            else:
                # Find which step failed
                steps = r[m]["steps"]
                failed_at = "?"
                for step in STEP_ORDER:
                    if step in steps and steps[step].get("success") is False:
                        failed_at = step.split("_")[0]
                        break
                    if step not in steps:
                        # Pipeline stopped before this step
                        prev_idx = STEP_ORDER.index(step) - 1
                        if prev_idx >= 0:
                            failed_at = STEP_ORDER[prev_idx].split("_")[0]
                        break
                label = f"FAIL@{failed_at}"
            print(f"  {label:>{col_w}}", end="")
        print()


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results across methods")
    parser.add_argument("benchmark_dir", nargs="?", default="output/benchmark",
                        help="Directory containing benchmark .json files (default: output/benchmark)")
    args = parser.parse_args()

    benchmarks = load_benchmarks(args.benchmark_dir)
    if not benchmarks:
        print(f"No .json files found in {args.benchmark_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(benchmarks)} methods: {', '.join(benchmarks.keys())}")

    print_overview(benchmarks)

    for group in ["all", "success", "failure"]:
        # Skip if all methods have 0 entries for this group
        if all(
            benchmarks[m]["summary"][group].get("total_time", {}).get("n", 0) == 0
            for m in benchmarks
        ):
            continue
        print_step_comparison(benchmarks, group)

    print_per_image_diff(benchmarks)
    print()


if __name__ == "__main__":
    main()