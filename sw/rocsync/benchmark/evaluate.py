#!/usr/bin/env python3
"""Compare benchmark results against ground truth annotations.

Loads validation result JSON files from output/benchmark/ and compares them
against validation_data/ground_truth.json. Prints detection metrics (TPR, FPR,
F1), result accuracy, and per-step timing statistics.

Positive/negative annotation is determined per-step from the ground truth
visible flags, not from a global status field.
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

def _ring_visible(d):
    """Ring is visible when start != end."""
    r = d.get("ring", {})
    return r.get("start", 0) != r.get("end", 0)


# Map pipeline steps to ground truth visibility checks.
# Each returns True if the ground truth considers that step's output "positive".
STEP_GT_POSITIVE = {
    "aruco":   lambda gt: gt.get("aruco", {}).get("visible", False),
    "corners": lambda gt: any(c.get("visible", False) for c in gt.get("corners", [])),
    "counter": lambda gt: gt.get("counter", {}).get("visible", False),
    "ring":    _ring_visible,
    # Overall: a timestamp is extractable when both counter and ring are visible
    "overall": lambda gt: (gt.get("counter", {}).get("visible", False)
                           and _ring_visible(gt)),
}

# Map pipeline steps to prediction visibility checks.
STEP_PRED_POSITIVE = {
    "aruco":   lambda p: p.get("aruco", {}).get("visible", False),
    "corners": lambda p: any(c.get("visible", False) for c in p.get("corners", [])),
    "counter": lambda p: p.get("counter", {}).get("visible", False),
    "ring":    _ring_visible,
    "overall": lambda p: p.get("success", False),
}


def load_benchmarks(directory):
    """Load all benchmark JSON files, keyed by stem name."""
    benchmarks = {}
    for path in sorted(Path(directory).glob("*.json")):
        with open(path) as f:
            benchmarks[path.stem] = json.load(f)
    return benchmarks


def load_ground_truth(path):
    """Load ground truth JSON file."""
    with open(path) as f:
        return json.load(f)


def describe(values):
    """Return dict with mean, median, std, min, max for a list of numbers."""
    if not values:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None, "n": 0}
    a = np.array(values, dtype=np.float64)
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "n": len(values),
    }


# ── Detection metrics ────────────────────────────────────────────────────────

def compute_detection_metrics(benchmark_images, gt_images):
    """Compute per-step and overall TP/FP/FN/TN counts and derived rates.

    For each step, "positive" is defined by the ground truth visible flag.
    Only images present in both benchmark and ground truth are considered.
    """
    results = {}
    for step, gt_pos_fn in STEP_GT_POSITIVE.items():
        pred_pos_fn = STEP_PRED_POSITIVE[step]
        tp = fp = fn = tn = 0

        for img_key, gt in gt_images.items():
            pred = benchmark_images.get(img_key)
            if pred is None:
                continue
            gt_positive = gt_pos_fn(gt)
            pred_positive = pred_pos_fn(pred)

            if gt_positive and pred_positive:
                tp += 1
            elif not gt_positive and pred_positive:
                fp += 1
            elif gt_positive and not pred_positive:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        fpr = fp / (fp + tn) if (fp + tn) > 0 else None
        f1 = 2 * precision * recall / (precision + recall) if precision and recall else None

        results[step] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "fpr": fpr, "f1": f1,
        }

    return results


# ── Result accuracy ──────────────────────────────────────────────────────────

def compute_accuracy(benchmark_images, gt_images):
    """Compare decoded values on true positives (both GT and pred have a timestamp).

    Returns per-field correct counts and timestamp error statistics in ms.
    The timestamp is already in ms: counter * 100 + ring_position.
    """
    n_compared = 0
    correct = {"aruco_id": 0, "counter_value": 0, "ring_start": 0, "ring_end": 0, "timestamp": 0}
    # Absolute errors in ms
    start_errors = []
    end_errors = []
    exposure_errors = []

    for img_key, gt in gt_images.items():
        pred = benchmark_images.get(img_key)
        if pred is None:
            continue
        if not STEP_GT_POSITIVE["overall"](gt) or not STEP_PRED_POSITIVE["overall"](pred):
            continue

        n_compared += 1

        # Per-field exact match counts
        if pred.get("aruco", {}).get("id") == gt.get("aruco", {}).get("id"):
            correct["aruco_id"] += 1
        gt_counter = gt.get("counter", {}).get("value")
        pred_counter = pred.get("counter", {}).get("value")
        if pred_counter == gt_counter:
            correct["counter_value"] += 1
        gt_ring = gt.get("ring", {})
        pred_ring = pred.get("ring", {})
        if pred_ring.get("start") == gt_ring.get("start"):
            correct["ring_start"] += 1
        if pred_ring.get("end") == gt_ring.get("end"):
            correct["ring_end"] += 1

        gt_ts = _gt_timestamp(gt)
        pred_ts = pred.get("timestamp")
        if pred_ts == gt_ts:
            correct["timestamp"] += 1

        # Absolute timestamp error in ms
        if pred_ts is not None and gt_ts is not None:
            start_errors.append(abs(pred_ts[0] - gt_ts[0]))
            end_errors.append(abs(pred_ts[1] - gt_ts[1]))
            pred_exposure = pred_ts[1] - pred_ts[0]
            gt_exposure = gt_ts[1] - gt_ts[0]
            exposure_errors.append(abs(pred_exposure - gt_exposure))

    return {
        "n_compared": n_compared,
        "correct": correct,
        "timestamp_error": {
            "start": describe(start_errors),
            "end": describe(end_errors),
            "exposure": describe(exposure_errors),
        },
    }


def _gt_timestamp(gt):
    """Reconstruct [start, end] timestamp from ground truth fields."""
    counter = gt.get("counter", {}).get("value")
    ring = gt.get("ring", {})
    if counter is None or ring.get("start", 0) == ring.get("end", 0):
        return None
    start = ring["start"] + counter * 100
    end = ring["end"] + counter * 100
    return [start, end]


# ── Timing statistics ─────────────────────────────────────────────────────────

def compute_timing(benchmark_images, gt_images):
    """Compute per-step timing statistics for all / positive / negative subsets.

    For each step, positive/negative is defined by the ground truth visible
    flag for that step. The "total" row uses the overall criterion (counter +
    ring visible).
    """
    # Map step timing keys to the GT criterion used for subsetting
    step_to_gt_key = {
        "aruco_detection":   "aruco",
        "corner_detection":  "corners",
        "fine_rectification": "corners",  # fine rectification depends on corners
        "counter_reading":   "counter",
        "ring_reading":      "ring",
    }

    result = {}
    for subset_name in ["all", "positive", "negative"]:
        step_stats = {}

        for step in STEP_ORDER:
            timing_key = f"{step}_ms"
            gt_key = step_to_gt_key[step]
            gt_pos_fn = STEP_GT_POSITIVE[gt_key]

            values = []
            for img_key, pred in benchmark_images.items():
                if timing_key not in pred.get("timing", {}):
                    continue
                gt = gt_images.get(img_key)
                if gt is None:
                    continue

                if subset_name == "all":
                    values.append(pred["timing"][timing_key])
                elif subset_name == "positive" and gt_pos_fn(gt):
                    values.append(pred["timing"][timing_key])
                elif subset_name == "negative" and not gt_pos_fn(gt):
                    values.append(pred["timing"][timing_key])

            step_stats[step] = describe(values)

        # Total timing uses overall criterion
        gt_pos_fn = STEP_GT_POSITIVE["overall"]
        total_values = []
        for img_key, pred in benchmark_images.items():
            total_ms = pred.get("timing", {}).get("total_ms")
            if total_ms is None:
                continue
            gt = gt_images.get(img_key)
            if gt is None:
                continue

            if subset_name == "all":
                total_values.append(total_ms)
            elif subset_name == "positive" and gt_pos_fn(gt):
                total_values.append(total_ms)
            elif subset_name == "negative" and not gt_pos_fn(gt):
                total_values.append(total_ms)

        step_stats["total"] = describe(total_values)
        result[subset_name] = step_stats

    return result


# ── Printing ──────────────────────────────────────────────────────────────────

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


def print_header(methods, col_width, label_col=30):
    print(f"  {'':>{label_col}}", end="")
    for m in methods:
        print(f"  {m:>{col_width}}", end="")
    print()
    print(f"  {'':>{label_col}}", end="")
    for _ in methods:
        print(f"  {'-' * col_width}", end="")
    print()


def print_detection(methods, metrics, col_w, label_col=30):
    step_labels = {
        "aruco": "ArUco detection",
        "corners": "Corner detection",
        "counter": "Counter reading",
        "ring": "Ring reading",
        "overall": "Overall (timestamp)",
    }

    print(f"\n{'=' * 80}")
    print(f"  DETECTION METRICS (per-step)")
    print(f"{'=' * 80}")

    for step in ["aruco", "corners", "counter", "ring", "overall"]:
        print(f"\n  {step_labels[step]}")
        print_header(methods, col_w, label_col)

        for key, label in [("tp", "TP"), ("fp", "FP"), ("fn", "FN"), ("tn", "TN")]:
            print(f"  {label:>{label_col}}", end="")
            for m in methods:
                print(f"  {fmt(metrics[m][step][key], col_w)}", end="")
            print()

        for key, label, formatter in [("recall", "TPR (Recall)", pct),
                                       ("fpr", "FPR", pct),
                                       ("precision", "Precision", pct),
                                       ("f1", "F1 Score", pct)]:
            print(f"  {label:>{label_col}}", end="")
            for m in methods:
                print(f"  {formatter(metrics[m][step][key], col_w)}", end="")
            print()


def print_accuracy(methods, accuracy, col_w, label_col=30):
    print(f"\n{'=' * 80}")
    print(f"  RESULT ACCURACY (on overall True Positives)")
    print(f"{'=' * 80}")
    print_header(methods, col_w, label_col)

    for field, label in [("aruco_id", "aruco.id correct"),
                          ("counter_value", "counter.value correct"),
                          ("ring_start", "ring.start correct"),
                          ("ring_end", "ring.end correct"),
                          ("timestamp", "timestamp exact match")]:
        print(f"  {label:>{label_col}}", end="")
        for m in methods:
            n = accuracy[m]["n_compared"]
            n_correct = accuracy[m]["correct"][field]
            if n > 0:
                text = f"{n_correct}/{n} ({n_correct/n:.0%})"
            else:
                text = "-"
            print(f"  {text:>{col_w}}", end="")
        print()

    # Timestamp error statistics
    print(f"\n{'=' * 80}")
    print(f"  TIMESTAMP ERROR (ms, on overall True Positives)")
    print(f"{'=' * 80}")

    stat_keys = ["mean", "median", "std", "min", "max"]
    error_fields = [
        ("start", "start error"),
        ("end", "end error"),
        ("exposure", "exposure error"),
    ]

    print_header(methods, col_w, label_col)
    for err_key, err_label in error_fields:
        print(f"  {err_label:>{label_col}}")
        for stat in stat_keys:
            print(f"  {'  ' + stat:>{label_col}}", end="")
            for m in methods:
                val = accuracy[m]["timestamp_error"].get(err_key, {}).get(stat)
                print(f"  {fmt(val, col_w)}", end="")
            print()


def print_timing(methods, timing, col_w, label_col=30):
    subset_labels = {
        "all": "ALL IMAGES",
        "positive": "POSITIVE ANNOTATIONS (per-step)",
        "negative": "NEGATIVE ANNOTATIONS (per-step)",
    }
    stat_keys = ["mean", "median", "std", "min", "max"]

    for subset_name in ["all", "positive", "negative"]:
        # Check if any method has data for this subset
        any_data = any(
            timing[m][subset_name].get("total", {}).get("n", 0) > 0
            for m in methods
        )
        if not any_data:
            continue

        print(f"\n{'=' * 80}")
        print(f"  TIMING — {subset_labels[subset_name]} (ms)")
        print(f"{'=' * 80}")
        print_header(methods, col_w, label_col)

        for step in STEP_ORDER + ["total"]:
            step_label = step.upper() if step == "total" else step
            print(f"  {step_label:>{label_col}}")
            for stat in stat_keys:
                print(f"  {'  ' + stat:>{label_col}}", end="")
                for m in methods:
                    val = timing[m][subset_name].get(step, {}).get(stat)
                    print(f"  {fmt(val, col_w)}", end="")
                print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate benchmark results against ground truth")
    parser.add_argument("benchmark_dir", nargs="?", default="output/benchmark",
                        help="Directory containing benchmark .json files (default: output/benchmark)")
    parser.add_argument("-g", "--ground-truth", default="validation_data/ground_truth.json",
                        help="Path to ground truth JSON (default: validation_data/ground_truth.json)")
    parser.add_argument("-t", "--timing", action="store_true",
                        help="Include per-step timing statistics")
    args = parser.parse_args()

    gt = load_ground_truth(args.ground_truth)
    gt_images = gt["images"]

    benchmarks = load_benchmarks(args.benchmark_dir)
    if not benchmarks:
        print(f"No .json files found in {args.benchmark_dir}", file=sys.stderr)
        sys.exit(1)

    methods = list(benchmarks.keys())
    col_w = max(14, max(len(m) for m in methods) + 2)

    n_images = len(gt_images)
    print(f"Ground truth: {args.ground_truth} ({n_images} images)")
    print(f"Methods: {', '.join(methods)}")

    # Compute all metrics
    metrics = {}
    accuracy = {}
    for m in methods:
        bm_images = benchmarks[m]["images"]
        metrics[m] = compute_detection_metrics(bm_images, gt_images)
        accuracy[m] = compute_accuracy(bm_images, gt_images)

    print_detection(methods, metrics, col_w)
    print_accuracy(methods, accuracy, col_w)

    if args.timing:
        timing = {}
        for m in methods:
            timing[m] = compute_timing(benchmarks[m]["images"], gt_images)
        print_timing(methods, timing, col_w)

    print()


if __name__ == "__main__":
    main()