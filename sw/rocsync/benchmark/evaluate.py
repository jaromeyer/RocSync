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

import cv2
import numpy as np

from rocsync.vision import aruco_corners_coords, corner_dots, led_size
from rocsync.benchmark.common import (
    STEP_ORDER,
    confusion_metrics,
    descriptive_stats,
    reconstruct_timestamp,
    ring_visible,
)

CORNER_IMAGE_SPACE_THRESHOLD_PX = 3
CORNER_BOARD_SPACE_THRESHOLD_PX = led_size
LABEL_WIDTH_DEFAULT = 40


# Map pipeline steps to ground truth visibility checks.
# Each returns True if the ground truth considers that step's output "positive".
STEP_GT_POSITIVE = {
    "aruco":   lambda gt: gt.get("aruco", {}).get("visible", False),
    "corners": lambda gt: any(c.get("visible", False) for c in gt.get("corners", [])),
    "counter": lambda gt: gt.get("counter", {}).get("visible", False),
    "ring":    ring_visible,
    # Overall: a timestamp is extractable when both counter and ring are visible
    "overall": lambda gt: (gt.get("counter", {}).get("visible", False)
                           and ring_visible(gt)),
}

# Map pipeline steps to prediction visibility checks.
STEP_PRED_POSITIVE = {
    "aruco":   lambda p: p.get("aruco", {}).get("visible", False),
    "corners": lambda p: any(c.get("visible", False) for c in p.get("corners", [])),
    "counter": lambda p: p.get("counter", {}).get("visible", False),
    "ring":    ring_visible,
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


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _gt_aruco_corners(gt):
    """Derive ground-truth ArUco corners in original image space.

    Inverse-transforms the known board-space ArUco corner coordinates through
    the ground-truth homography.  Returns a (4, 2) array or None.
    """
    H = gt.get("homography")
    if H is None or not gt.get("aruco", {}).get("visible", False):
        return None
    inv_H = np.linalg.inv(np.array(H, dtype=np.float64))
    pts = np.array([aruco_corners_coords], dtype=np.float64)
    return cv2.perspectiveTransform(pts, inv_H).reshape(4, 2)


def _gt_corner_positions(gt):
    """Derive ground-truth corner LED positions in original image space.

    Inverse-transforms the known board-space corner LED coordinates through
    the ground-truth homography.  Returns a (4, 2) list or None.
    """
    H = gt.get("homography")
    if H is None:
        return None
    inv_H = np.linalg.inv(np.array(H, dtype=np.float64))
    pts = np.array([corner_dots], dtype=np.float64)
    return cv2.perspectiveTransform(pts, inv_H).reshape(4, 2).tolist()


def _pred_corner_positions_image(pred, gt):
    """Transform predicted corner positions from board space to image space.

    The pipeline detects corners in the rough-rectified (board-space) image,
    so positions need to be inverse-transformed through the GT homography to
    get image-space coordinates for comparison with GT image-space positions.

    Returns a list of 4 image-space [x, y] positions (or None per corner).
    """
    H = gt.get("homography")
    if H is None:
        return [None] * 4
    inv_H = np.linalg.inv(np.array(H, dtype=np.float64))
    result = []
    for c in pred.get("corners", []):
        if c.get("visible") and c.get("position") is not None:
            pt = np.array([[c["position"]]], dtype=np.float64)
            img_pt = cv2.perspectiveTransform(pt, inv_H).reshape(2)
            result.append(img_pt.tolist())
        else:
            result.append(None)
    return result


# ── Per-step metric computation ──────────────────────────────────────────────

def compute_step_detection(benchmark_images, gt_images, step):
    """Compute TP/FP/FN/TN and derived rates for a single step.

    Corner detection uses per-corner matching with a position error threshold
    of CORNER_MAX_ERROR_PX in image coordinates.
    """
    gt_pos_fn = STEP_GT_POSITIVE[step]
    pred_pos_fn = STEP_PRED_POSITIVE[step]
    tp = fp = fn = tn = 0

    for img_key, gt in gt_images.items():
        pred = benchmark_images.get(img_key)
        if pred is None:
            continue

        if step == "corners":
            gt_corners = gt.get("corners", [])
            pred_corners = pred.get("corners", [])
            gt_positions = _gt_corner_positions(gt)

            gt_any_visible = any(c.get("visible", False) for c in gt_corners)
            pred_any_visible = any(c.get("visible", False) for c in pred_corners)
            any_tp = False

            if gt_positions is not None:
                pred_img = _pred_corner_positions_image(pred, gt)
                for i in range(min(len(gt_corners), len(pred_corners), 4)):
                    gt_vis = gt_corners[i].get("visible", False)
                    pred_vis = pred_corners[i].get("visible", False)

                    if gt_vis and pred_vis and pred_img[i] is not None:
                        err = np.linalg.norm(
                            np.array(pred_img[i]) - np.array(gt_positions[i]))
                        if err <= CORNER_IMAGE_SPACE_THRESHOLD_PX:
                            any_tp = True

            if gt_any_visible and any_tp:
                tp += 1
            elif not gt_any_visible and pred_any_visible:
                fp += 1
            elif gt_any_visible and not any_tp:
                fn += 1
            else:
                tn += 1
        else:
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

    return confusion_metrics(tp, fp, fn, tn)


def compute_aruco_metrics(benchmark_images, gt_images):
    """Compute ArUco-specific metrics: detection + corner pixel error."""
    detection = compute_step_detection(benchmark_images, gt_images, "aruco")

    mean_errors = []
    for img_key, gt in gt_images.items():
        pred = benchmark_images.get(img_key)
        if pred is None:
            continue
        if not gt.get("aruco", {}).get("visible", False):
            continue
        if not pred.get("aruco", {}).get("visible", False):
            continue

        gt_corners = _gt_aruco_corners(gt)
        pred_corners = pred.get("aruco", {}).get("corners")
        if gt_corners is None or pred_corners is None:
            continue

        pred_corners = np.array(pred_corners, dtype=np.float64).reshape(4, 2)
        dists = np.linalg.norm(pred_corners - gt_corners, axis=1)
        mean_errors.append(float(np.mean(dists)))

    return {
        "detection": detection,
        "corner_error_px": descriptive_stats(mean_errors),
    }


def compute_corner_metrics(benchmark_images, gt_images):
    """Compute corner LED metrics: image-space and board-space detection + errors."""
    detection_image = compute_step_detection(benchmark_images, gt_images, "corners")

    # Per-corner board-space detection and pixel errors in both spaces
    tp_board = fp_board = fn_board = tn_board = 0
    errors_image = []
    errors_board = []

    for img_key, gt in gt_images.items():
        pred = benchmark_images.get(img_key)
        if pred is None:
            continue

        gt_corners = gt.get("corners", [])
        pred_corners = pred.get("corners", [])
        gt_positions = _gt_corner_positions(gt)
        pred_img = _pred_corner_positions_image(pred, gt)

        for i in range(min(len(gt_corners), 4)):
            gt_vis = gt_corners[i].get("visible", False)
            pred_vis = i < len(pred_corners) and pred_corners[i].get("visible", False)
            pred_pos = pred_corners[i].get("position") if pred_vis else None

            # Image-space pixel error (on TPs where both are visible with positions)
            if gt_vis and pred_vis and pred_img[i] is not None and gt_positions is not None:
                err_img = np.linalg.norm(
                    np.array(pred_img[i]) - np.array(gt_positions[i]))
                errors_image.append(float(err_img))

            # Board-space detection + pixel error
            # Predicted positions are already in board space (from rough rectification)
            if gt_vis and pred_vis and pred_pos is not None:
                err_board = np.linalg.norm(
                    np.array(pred_pos) - np.array(corner_dots[i]))
                errors_board.append(float(err_board))
                if err_board <= CORNER_BOARD_SPACE_THRESHOLD_PX:
                    tp_board += 1
                else:
                    fn_board += 1
            elif gt_vis:
                fn_board += 1
            elif pred_vis:
                fp_board += 1
            else:
                tn_board += 1

    return {
        "detection_image": detection_image,
        "detection_board": confusion_metrics(tp_board, fp_board, fn_board, tn_board),
        "error_image_px": descriptive_stats(errors_image),
        "error_board_px": descriptive_stats(errors_board),
    }


def compute_counter_metrics(benchmark_images, gt_images):
    """Compute counter detection + value accuracy."""
    detection = compute_step_detection(benchmark_images, gt_images, "counter")

    n_compared = 0
    n_correct = 0
    for img_key, gt in gt_images.items():
        pred = benchmark_images.get(img_key)
        if pred is None:
            continue
        if not gt.get("counter", {}).get("visible", False):
            continue
        if not pred.get("counter", {}).get("visible", False):
            continue
        n_compared += 1
        if pred.get("counter", {}).get("value") == gt.get("counter", {}).get("value"):
            n_correct += 1

    return {
        "detection": detection,
        "value_correct": n_correct,
        "value_compared": n_compared,
    }


def compute_ring_metrics(benchmark_images, gt_images):
    """Compute ring detection + start/end value accuracy."""
    detection = compute_step_detection(benchmark_images, gt_images, "ring")

    n_compared = 0
    start_correct = 0
    end_correct = 0
    for img_key, gt in gt_images.items():
        pred = benchmark_images.get(img_key)
        if pred is None:
            continue
        if not ring_visible(gt) or not ring_visible(pred):
            continue
        n_compared += 1
        if pred.get("ring", {}).get("start") == gt.get("ring", {}).get("start"):
            start_correct += 1
        if pred.get("ring", {}).get("end") == gt.get("ring", {}).get("end"):
            end_correct += 1

    return {
        "detection": detection,
        "start_correct": start_correct,
        "end_correct": end_correct,
        "value_compared": n_compared,
    }


def compute_overall_metrics(benchmark_images, gt_images):
    """Compute overall detection + timestamp accuracy + error stats."""
    detection = compute_step_detection(benchmark_images, gt_images, "overall")

    n_compared = 0
    n_correct = 0
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
        gt_ts = reconstruct_timestamp(gt)
        pred_ts = reconstruct_timestamp(pred)
        if pred_ts == gt_ts:
            n_correct += 1
        if pred_ts is not None and gt_ts is not None:
            start_errors.append(abs(pred_ts[0] - gt_ts[0]))
            end_errors.append(abs(pred_ts[1] - gt_ts[1]))
            pred_exposure = pred_ts[1] - pred_ts[0]
            gt_exposure = gt_ts[1] - gt_ts[0]
            exposure_errors.append(abs(pred_exposure - gt_exposure))

    return {
        "detection": detection,
        "timestamp_correct": n_correct,
        "timestamp_compared": n_compared,
        "timestamp_error": {
            "start": descriptive_stats(start_errors),
            "end": descriptive_stats(end_errors),
            "exposure": descriptive_stats(exposure_errors),
        },
    }


# ── Timing statistics ────────────────────────────────────────────────────────

def compute_timing(benchmark_images, gt_images):
    """Compute per-step timing statistics for all / positive / negative subsets."""
    step_to_gt_key = {
        "aruco_detection":   "aruco",
        "corner_detection":  "corners",
        "fine_rectification": "corners",
        "counter_reading":   "counter",
        "ring_reading":      "ring",
    }

    def collect_values(timing_key, gt_positive_fn, subset):
        """Collect timing values filtered by ground-truth subset."""
        values = []
        for img_key, pred in benchmark_images.items():
            value = pred.get("timing", {}).get(timing_key)
            if value is None:
                continue
            gt = gt_images.get(img_key)
            if gt is None:
                continue
            gt_positive = gt_positive_fn(gt)
            if subset == "all" or (subset == "positive") == gt_positive:
                values.append(value)
        return values

    result = {}
    for subset in ["all", "positive", "negative"]:
        step_stats = {}
        for step in STEP_ORDER:
            gt_key = step_to_gt_key[step]
            step_stats[step] = descriptive_stats(
                collect_values(f"{step}_ms", STEP_GT_POSITIVE[gt_key], subset))
        step_stats["total"] = descriptive_stats(
            collect_values("total_ms", STEP_GT_POSITIVE["overall"], subset))
        result[subset] = step_stats

    return result


# ── Printing ─────────────────────────────────────────────────────────────────

def format_value(val, width=10):
    if val is None:
        return "-".rjust(width)
    if isinstance(val, float):
        return f"{val:.2f}".rjust(width)
    return str(val).rjust(width)


def format_percent(val, width=10):
    if val is None:
        return "-".rjust(width)
    return f"{val:.1%}".rjust(width)


def print_header(methods, col_width, label_width=LABEL_WIDTH_DEFAULT, label=""):
    print(f"  {'':>{label_width}}", end="")
    for m in methods:
        print(f"  {m:>{col_width}}", end="")
    print()
    print(f"    {label.upper():-^{label_width - 2}}", end="")
    for _ in methods:
        print(f"  {'-' * col_width}", end="")
    print()


def _print_detection(methods, get_det, col_width, label_width=LABEL_WIDTH_DEFAULT):
    """Print TP/FP/FN/TN and derived rates from a detection dict."""
    for key, label in [("tp", "TP"), ("fp", "FP"), ("fn", "FN"), ("tn", "TN")]:
        print(f"  {label:>{label_width}}", end="")
        for m in methods:
            print(f"  {format_value(get_det(m)[key], col_width)}", end="")
        print()
    for key, label, formatter in [("recall", "TPR (Recall)", format_percent),
                                   ("fpr", "FPR", format_percent),
                                   ("precision", "Precision", format_percent),
                                   ("f1", "F1 Score", format_percent)]:
        print(f"  {label:>{label_width}}", end="")
        for m in methods:
            print(f"  {formatter(get_det(m)[key], col_width)}", end="")
        print()


def _print_error_stats(methods, get_stats, col_width, label_width=LABEL_WIDTH_DEFAULT):
    """Print descriptive statistics (mean/median/std/min/max/n)."""
    for stat in ["mean", "std", "min", "median", "max", "n"]:
        print(f"  {stat:>{label_width}}", end="")
        for m in methods:
            print(f"  {format_value(get_stats(m).get(stat), col_width)}", end="")
        print()


def _print_value_accuracy(methods, get_n, get_correct, label, col_width, label_width=LABEL_WIDTH_DEFAULT):
    """Print a single value accuracy row (e.g. '5/7 (71%)')."""
    print(f"  {label:>{label_width}}", end="")
    for m in methods:
        n = get_n(m)
        nc = get_correct(m)
        text = f"{nc}/{n} ({nc/n:.0%})" if n > 0 else "-"
        print(f"  {text:>{col_width}}", end="")
    print()


def print_report(methods, all_metrics, col_width, label_width=LABEL_WIDTH_DEFAULT):
    """Print the full evaluation report, grouped by pipeline step."""

    # ── ArUco detection ──────────────────────────────────────────────────
    print(f"{'=' * 100}")
    print(f"  ARUCO DETECTION")
    print(f"{'=' * 100}")

    print_header(methods, col_width, label_width, "Detection")
    _print_detection(methods, lambda m: all_metrics[m]["aruco"]["detection"],
                     col_width, label_width)

    print()
    print_header(methods, col_width, label_width, "Corner error (px, image space)")
    _print_error_stats(methods, lambda m: all_metrics[m]["aruco"]["corner_error_px"],
                       col_width, label_width)

    # ── Corner LED detection ─────────────────────────────────────────────
    print(f"{'=' * 100}")
    print(f"  CORNER LED DETECTION")
    print(f"{'=' * 100}")

    print_header(methods, col_width, label_width, f"Detection (thres: {CORNER_IMAGE_SPACE_THRESHOLD_PX}px in image space)")
    _print_detection(methods, lambda m: all_metrics[m]["corners"]["detection_image"],
                     col_width, label_width)

    print()
    print_header(methods, col_width, label_width, f"Detection (thres: {CORNER_BOARD_SPACE_THRESHOLD_PX}px in board space)")
    _print_detection(methods, lambda m: all_metrics[m]["corners"]["detection_board"],
                     col_width, label_width)

    print()
    print_header(methods, col_width, label_width, "Pixel error — image space")
    _print_error_stats(methods, lambda m: all_metrics[m]["corners"]["error_image_px"],
                       col_width, label_width)

    print()
    print_header(methods, col_width, label_width, "Pixel error — board space")
    _print_error_stats(methods, lambda m: all_metrics[m]["corners"]["error_board_px"],
                       col_width, label_width)

    # ── Counter reading ──────────────────────────────────────────────────
    print(f"{'=' * 100}")
    print(f"  COUNTER READING")
    print(f"{'=' * 100}")

    print_header(methods, col_width, label_width, "Detection")
    _print_detection(methods, lambda m: all_metrics[m]["counter"]["detection"],
                     col_width, label_width)

    print()
    print_header(methods, col_width, label_width, "Value accuracy")
    _print_value_accuracy(
        methods,
        lambda m: all_metrics[m]["counter"]["value_compared"],
        lambda m: all_metrics[m]["counter"]["value_correct"],
        "counter.value correct", col_width, label_width)

    # ── Ring reading ─────────────────────────────────────────────────────
    print(f"{'=' * 100}")
    print(f"  RING READING")
    print(f"{'=' * 100}")

    print_header(methods, col_width, label_width, "Detection")
    _print_detection(methods, lambda m: all_metrics[m]["ring"]["detection"],
                     col_width, label_width)

    print()
    print_header(methods, col_width, label_width, "Value accuracy")
    _print_value_accuracy(
        methods,
        lambda m: all_metrics[m]["ring"]["value_compared"],
        lambda m: all_metrics[m]["ring"]["start_correct"],
        "ring.start correct", col_width, label_width)
    _print_value_accuracy(
        methods,
        lambda m: all_metrics[m]["ring"]["value_compared"],
        lambda m: all_metrics[m]["ring"]["end_correct"],
        "ring.end correct", col_width, label_width)

    # ── Overall ──────────────────────────────────────────────────────────
    print(f"{'=' * 100}")
    print(f"  OVERALL (timestamp)")
    print(f"{'=' * 100}")

    print_header(methods, col_width, label_width, "Detection")
    _print_detection(methods, lambda m: all_metrics[m]["overall"]["detection"],
                     col_width, label_width)

    print()
    print_header(methods, col_width, label_width, "Timestamp accuracy")
    _print_value_accuracy(
        methods,
        lambda m: all_metrics[m]["overall"]["timestamp_compared"],
        lambda m: all_metrics[m]["overall"]["timestamp_correct"],
        "timestamp exact match", col_width, label_width)

    print()
    print_header(methods, col_width, label_width, "Timestamp error (ms)")
    for err_key, err_label in [("start", "start error"),
                                ("end", "end error"),
                                ("exposure", "exposure error")]:
        print(f"  {err_label.upper():>{label_width}}")
        for stat in ["mean", "std", "min", "median", "max"]:
            print(f"  {'  ' + stat:>{label_width}}", end="")
            for m in methods:
                val = all_metrics[m]["overall"]["timestamp_error"].get(err_key, {}).get(stat)
                print(f"  {format_value(val, col_width)}", end="")
            print()


def print_timing(methods, timing, col_width, label_width=30):
    subset_labels = {
        "all": "ALL IMAGES",
        "positive": "POSITIVE ANNOTATIONS (per-step)",
        "negative": "NEGATIVE ANNOTATIONS (per-step)",
    }
    stat_keys = ["mean", "std", "min", "median", "max"]

    for subset_name in ["all", "positive", "negative"]:
        any_data = any(
            timing[m][subset_name].get("total", {}).get("n", 0) > 0
            for m in methods
        )
        if not any_data:
            continue

        print(f"{'=' * 100}")
        print(f"  TIMING — {subset_labels[subset_name]} (ms)")
        print(f"{'=' * 100}")
        print_header(methods, col_width, label_width)

        for step in STEP_ORDER + ["total"]:
            step_label = step.upper() if step == "total" else step
            print(f"  {step_label:>{label_width}}")
            for stat in stat_keys:
                print(f"  {'  ' + stat:>{label_width}}", end="")
                for m in methods:
                    val = timing[m][subset_name].get(step, {}).get(stat)
                    print(f"  {format_value(val, col_width)}", end="")
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
    col_width = max(14, max(len(m) for m in methods) + 2)

    n_images = len(gt_images)
    print(f"Ground truth: {args.ground_truth} ({n_images} images)")
    print(f"Methods: {', '.join(methods)}")

    # Compute all metrics per method
    all_metrics = {}
    for m in methods:
        bm = benchmarks[m]["images"]
        all_metrics[m] = {
            "aruco": compute_aruco_metrics(bm, gt_images),
            "corners": compute_corner_metrics(bm, gt_images),
            "counter": compute_counter_metrics(bm, gt_images),
            "ring": compute_ring_metrics(bm, gt_images),
            "overall": compute_overall_metrics(bm, gt_images),
        }

    print_report(methods, all_metrics, col_width)

    if args.timing:
        timing = {}
        for m in methods:
            timing[m] = compute_timing(benchmarks[m]["images"], gt_images)
        print_timing(methods, timing, col_width)

    print()


if __name__ == "__main__":
    main()