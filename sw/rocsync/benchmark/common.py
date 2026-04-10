"""Shared utilities for RocSync benchmark tools."""

from pathlib import Path
import numpy as np

STEP_ORDER = [
    "aruco_detection",
    "corner_detection",
    "fine_rectification",
    "counter_reading",
    "ring_reading",
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def collect_images(root_dir):
    """Collect all image files recursively, sorted by path."""
    return sorted(
        p for p in Path(root_dir).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def ring_visible(image_data):
    """Ring is visible when start != end (half-open interval has nonzero length)."""
    ring = image_data.get("ring", {})
    return ring.get("start", 0) != ring.get("end", 0)


def reconstruct_timestamp(image_data, board):
    """Reconstruct [start, end] timestamp from counter value and ring position.

    Returns [start, end] list or None if counter or ring is not visible.
    """
    counter_value = image_data.get("counter", {}).get("value")
    ring = image_data.get("ring", {})
    if counter_value is None or ring.get("start", 0) == ring.get("end", 0):
        return None
    start = ring["start"] + counter_value * board.period
    end = ring["end"] + counter_value * board.period
    return [start, end]


def descriptive_stats(values):
    """Compute mean, median, std, min, max, n for a list of numbers."""
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


def confusion_metrics(tp, fp, fn, tn):
    """Derive precision, recall, FPR, and F1 from confusion matrix counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    fpr = fp / (fp + tn) if (fp + tn) > 0 else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "fpr": fpr, "f1": f1,
    }