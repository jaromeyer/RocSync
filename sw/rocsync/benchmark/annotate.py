#!/usr/bin/env python3
"""Interactive annotation and verification tool for RocSync benchmark images.

Runs the rocsync pipeline on validation images, displays results with LED
overlays, and lets the user verify or correct the decoded values. Produces
a ground_truth.json file for benchmark evaluation.

Usage:
    python -m rocsync.benchmark.annotate [data_dir] [--try-hard] [-o ground_truth.json]
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from rocsync.vision import (
    CameraType,
    aruco_corners_coords,
    board_size,
    corner_dots,
    dictionary,
    led_size,
    period,
    process_frame,
    visible_radius,
)
from rocsync.benchmark.validate import collect_images


# ── Board geometry helpers ──────────────────────────────────────────────────

def ring_led_positions():
    """Return list of 100 (x, y) tuples for ring LEDs in rectified board coords."""
    positions = []
    for i in range(period):
        angle = -(i / period + 0.25) * 2 * math.pi
        x = int(board_size / 2 + visible_radius * math.cos(angle))
        y = int(board_size / 2 + visible_radius * math.sin(angle))
        positions.append((x, y))
    return positions


def counter_led_positions():
    """Return list of 16 (x, y) tuples for counter LEDs in rectified board coords."""
    y = int(53 / 250 * 640)
    return [(int((65 + i * 8) / 250 * 640), y) for i in range(16)]


def corner_led_positions():
    """Return list of 4 (x, y) tuples for corner LEDs in rectified board coords."""
    return [(int(p[0]), int(p[1])) for p in corner_dots]


RING_POS = ring_led_positions()
COUNTER_POS = counter_led_positions()
CORNER_POS = corner_led_positions()

# Counter bounding box (x1, y1, x2, y2) with margin around LEDs
_cx = [p[0] for p in COUNTER_POS]
_cy = [p[1] for p in COUNTER_POS]
COUNTER_BBOX = (
    min(_cx) - led_size - 10,
    min(_cy) - led_size - 10,
    max(_cx) + led_size + 10,
    max(_cy) + led_size + 10,
)

# ArUco region in rectified board coords
ARUCO_X1, ARUCO_Y1 = 202, 202
ARUCO_X2, ARUCO_Y2 = 437, 437
ARUCO_IDS = [0, 21]  # known marker IDs to cycle through

# Hit-test radii (in board coordinates)
HIT_CORNER = 20
HIT_COUNTER = led_size
HIT_RING = 25
DRAG_THRESHOLD = 3  # px before a click becomes a drag


# ── Annotation data ────────────────────────────────────────────────────────

@dataclass
class ImageAnnotation:
    """Mutable annotation state for a single image.

    Corner LED positions are stored in original image coordinates (float).
    The homography maps original image → rectified board space and is needed
    to display / edit positions on the rectified view.

    Ring uses half-open semantics: ring_start = first ON LED,
    ring_end = first OFF LED.  ring_start == ring_end means undecodable.
    """
    aruco_visible: bool = False
    aruco_id: int = 0

    # Each corner: {"visible": bool, "position": [x, y]}
    # position is in *original image* coordinates (float)
    corners: list = field(default_factory=lambda: [
        {"visible": False, "position": list(p)}
        for p in CORNER_POS
    ])

    # Homography: original image → rectified board (3×3, stored as list-of-lists)
    homography: list | None = None

    counter_visible: bool = False
    counter_leds: list = field(default_factory=lambda: [False] * 16)
    counter_value: int = 0

    ring_start: int = 0  # first ON LED index
    ring_end: int = 0    # first OFF LED index (exclusive); == start → undecodable

    @classmethod
    def from_stats(cls, stats):
        """Pre-populate annotation from pipeline stats dict."""
        ann = cls()

        # Homography (original → rectified)
        H = stats.get("homography")
        if H is not None:
            H = np.array(H, dtype=np.float64)
            ann.homography = H.tolist()

        # ArUco
        aruco_id = stats.get("aruco_id")
        ann.aruco_visible = aruco_id is not None
        ann.aruco_id = aruco_id if aruco_id is not None else 0

        # Corner LEDs — stats positions are in rough-rectified space,
        # convert to original image space via the rough homography inverse.
        corner_positions = stats.get("corner_positions")
        aruco_corners = stats.get("aruco_corners")
        if corner_positions is not None and aruco_corners is not None:
            rough_H = cv2.getPerspectiveTransform(
                np.array(aruco_corners, dtype=np.float32),
                aruco_corners_coords,
            )
            inv_rough = np.linalg.inv(rough_H)
            for i, pos in enumerate(corner_positions):
                if i < 4 and pos is not None:
                    pt = np.array([[pos]], dtype=np.float64)
                    orig_pt = cv2.perspectiveTransform(pt, inv_rough).reshape(2)
                    ann.corners[i]["visible"] = True
                    ann.corners[i]["position"] = [float(orig_pt[0]), float(orig_pt[1])]

        # Counter
        counter_leds = stats.get("counter_leds")
        if counter_leds is not None:
            ann.counter_visible = True
            ann.counter_leds = list(counter_leds)
            ann.counter_value = sum(2 ** (15 - i) for i in range(16) if counter_leds[i])

        # Ring — pipeline returns CCW [start, end) half-open.
        # Convert to clockwise: first ON (CW) = last ON (CCW) = end-1,
        #                        first OFF (CW) = before first ON (CCW) = start-1.
        steps = stats.get("steps", {})
        ring_step = steps.get("ring_reading", {})
        if ring_step.get("success") and stats.get("timestamp"):
            ts = stats["timestamp"]
            counter_val = steps.get("counter_reading", {}).get("value", 0)
            ccw_start = ts[0] - counter_val * period
            ccw_end = ts[1] - counter_val * period + 1  # pipeline is inclusive, make half-open
            ann.ring_start = (ccw_end - 1) % period   # first ON clockwise
            ann.ring_end = (ccw_start - 1) % period    # first OFF clockwise
        elif stats.get("ring_leds") is not None:
            ann.ring_start = 0
            ann.ring_end = 0

        return ann

    def recompute_counter(self):
        self.counter_value = sum(2 ** (15 - i) for i in range(16) if self.counter_leds[i])

    def to_original(self, board_x, board_y):
        """Transform a point from rectified board coords to original image coords."""
        if self.homography is None:
            return [float(board_x), float(board_y)]
        inv_H = np.linalg.inv(np.array(self.homography, dtype=np.float64))
        pt = np.array([[[board_x, board_y]]], dtype=np.float64)
        orig = cv2.perspectiveTransform(pt, inv_H).reshape(2)
        return [float(orig[0]), float(orig[1])]

    def to_board(self, orig_x, orig_y):
        """Transform a point from original image coords to rectified board coords."""
        if self.homography is None:
            return int(round(orig_x)), int(round(orig_y))
        H = np.array(self.homography, dtype=np.float64)
        pt = np.array([[[orig_x, orig_y]]], dtype=np.float64)
        board = cv2.perspectiveTransform(pt, H).reshape(2)
        return int(round(board[0])), int(round(board[1]))

    def to_dict(self):
        """Serialize annotation. Invisible components are stripped to visibility only."""
        corners = []
        for c in self.corners:
            if c["visible"]:
                corners.append({"visible": True, "position": c["position"]})
            else:
                corners.append({"visible": False})

        result = {
            "aruco": {"visible": self.aruco_visible},
            "corners": corners,
            "homography": self.homography,
            "counter": {"visible": self.counter_visible},
        }

        if self.aruco_visible:
            result["aruco"]["id"] = self.aruco_id
        if self.counter_visible:
            result["counter"]["value"] = self.counter_value
        if self.ring_start == self.ring_end:
            self.ring_start = 0
            self.ring_end = 0
        result["ring"]["start"] = self.ring_start
        result["ring"]["end"] = self.ring_end

        return result

    @classmethod
    def from_dict(cls, data):
        ann = cls()
        aruco = data.get("aruco", {})
        ann.aruco_visible = aruco.get("visible", False)
        ann.aruco_id = aruco.get("id", 0)

        ann.homography = data.get("homography")

        for i, c in enumerate(data.get("corners", [])):
            if i < 4:
                if "position" in c:
                    pos = list(c["position"])
                elif ann.homography is not None:
                    pos = ann.to_original(*CORNER_POS[i])
                else:
                    pos = list(CORNER_POS[i])
                ann.corners[i] = {
                    "visible": c.get("visible", False),
                    "position": pos,
                }

        counter = data.get("counter", {})
        ann.counter_visible = counter.get("visible", False)
        ann.counter_value = counter.get("value", 0)
        ann.counter_leds = [
            bool(ann.counter_value & (2 ** (15 - i))) for i in range(16)
        ]

        ring = data.get("ring", {})
        ann.ring_start = ring.get("start", 0)
        ann.ring_end = ring.get("end", 0)
        return ann


# ── Interaction state ───────────────────────────────────────────────────────

class Mode(Enum):
    IDLE = "idle"
    DRAGGING_CORNER = "dragging"
    RING_AWAITING_END = "ring_end"


# ── Display and interaction ─────────────────────────────────────────────────

COLOR_ON = (0, 0, 255)           # red  — LED is ON
COLOR_OFF = (255, 0, 0)          # blue — LED is OFF
COLOR_NOT_VIS = (128, 128, 128)  # gray — component hidden/undecodable
COLOR_RING_SEL = (0, 255, 0)     # green — ring selection highlight
COLOR_TEXT = (0, 0, 0)           # black text on bright bg
COLOR_STATUS_BG = (220, 220, 220)  # light gray
COLOR_BOARD_TEXT = (255, 255, 255)  # white text on dark board

WINDOW_NAME = "RocSync Annotation"
TARGET_HEIGHT = 800

HELP_TEXT = [
    "=== Mouse (right panel) ===",
    "Click corner LED  : Toggle visible / hidden",
    "Drag corner LED   : Refine position",
    "Click counter LED : Toggle ON / OFF",
    "Click counter box : Toggle counter visibility",
    "Click ArUco area  : Cycle ID 0 / ID 21 / none",
    "Click ring LED    : Set first ON, then first OFF",
    "                    (same LED = undecodable)",
]


class AnnotationTool:
    def __init__(self, data_dir, output_path=None, try_hard=False):
        self.data_dir = Path(data_dir)
        self.output_path = Path(output_path) if output_path else self.data_dir / "ground_truth.json"
        self.try_hard = try_hard
        self.ground_truth = {"images": {}}
        self.images = []
        self.current_idx = 0

        # Display state
        self.show_help = False
        self.left_panel_w = 0
        self.left_scale = 1.0
        self.board_scale = 1.0
        self.status_msg = ""

        # Interaction state
        self.mode = Mode.IDLE
        self.annotation = None
        self.stats = None
        self._current_image = None
        self._drag_idx = None
        self._drag_start = None
        self._drag_start_original = None
        self._drag_started = False
        self._drag_on_left = False
        self._ring_start_candidate = None
        self._needs_redraw = True
        self._window_sized = False

    # ── Main loop ───────────────────────────────────────────────────────

    def run(self):
        self.images = sorted(collect_images(self.data_dir))
        if not self.images:
            print("No images found.", file=sys.stderr)
            return

        self._load_ground_truth()
        first = self._find_unannotated(-1, forward=True)
        if first == -1:
            print("All images already annotated. Starting from the beginning.")
            self.current_idx = 0
        else:
            self.current_idx = first
            rel = str(self.images[first].relative_to(self.data_dir))
            print(f"Resuming at image {first + 1}/{len(self.images)}: {rel}")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)

        while 0 <= self.current_idx < len(self.images):
            image_path = self.images[self.current_idx]
            rel_path = str(image_path.relative_to(self.data_dir))

            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Cannot read {image_path}", file=sys.stderr)
                self.current_idx += 1
                continue

            # Run pipeline
            self.stats = {}
            try:
                process_frame(image, CameraType.RGB, 0,
                              try_hard=self.try_hard, stats=self.stats)
            except Exception as e:
                print(f"Pipeline error on {rel_path}: {e}", file=sys.stderr)
                self.stats["rectified"] = None
                self.stats["aruco_id"] = None

            # Load existing annotation or create from pipeline
            if rel_path in self.ground_truth["images"]:
                self.annotation = ImageAnnotation.from_dict(
                    self.ground_truth["images"][rel_path])
            else:
                self.annotation = ImageAnnotation.from_stats(self.stats)
                # Use the pipeline homography for new annotations
                H = self.stats.get("homography")
                if H is not None:
                    self.annotation.homography = np.array(H, dtype=np.float64).tolist()

            # Ensure invisible corners have positions in original image coords
            if self.annotation.homography is not None:
                for i, c in enumerate(self.annotation.corners):
                    if not c["visible"]:
                        c["position"] = self.annotation.to_original(*CORNER_POS[i])
            else:
                # No homography — place invisible corners in a centered square
                # proportional to the board's corner layout.
                img_h, img_w = image.shape[:2]
                scale = 0.5 * min(img_w, img_h) / board_size
                ox = (img_w - board_size * scale) / 2
                oy = (img_h - board_size * scale) / 2
                for i, c in enumerate(self.annotation.corners):
                    if not c["visible"]:
                        bx, by = CORNER_POS[i]
                        c["position"] = [ox + bx * scale, oy + by * scale]

            # Re-warp the rectified image using the annotation's homography
            # so the displayed image is consistent with to_original/to_board.
            # This matters when loading a saved annotation whose homography
            # differs from the pipeline's (e.g. user corrected corners earlier).
            if all(c["visible"] for c in self.annotation.corners):
                self._recompute_homography(image)
            elif self.annotation.homography is not None:
                # Not all corners visible but we have a saved homography.
                # Warp using it so the displayed image matches to_original/to_board.
                H = np.array(self.annotation.homography, dtype=np.float64)
                mask = image[:, :, 2]
                self.stats["rectified"] = cv2.warpPerspective(
                    mask, H, (board_size, board_size))

            self.mode = Mode.IDLE
            self._ring_start_candidate = None
            self.status_msg = ""
            self._needs_redraw = True

            self._current_image = image
            action = self._image_loop(image, rel_path)

            if action == "accept":
                self.ground_truth["images"][rel_path] = self.annotation.to_dict()
                self._save_ground_truth()
                self.current_idx += 1
            elif action == "skip":
                self.current_idx += 1
            elif action == "back":
                self.current_idx = max(0, self.current_idx - 1)
            elif action == "next_unannotated":
                self.current_idx = self._find_unannotated(self.current_idx, forward=True)
            elif action == "prev_unannotated":
                self.current_idx = self._find_unannotated(self.current_idx, forward=False)
            elif action == "clear":
                self.ground_truth["images"].pop(rel_path, None)
                self._save_ground_truth()
                # stay on current image — re-runs pipeline on next iteration
            elif action == "quit":
                break

        cv2.destroyAllWindows()

    def _image_loop(self, image, rel_path):
        composite = None
        while True:
            if self._needs_redraw:
                composite = self._render(image, rel_path)
                if not self._window_sized:
                    h, w = composite.shape[:2]
                    cv2.resizeWindow(WINDOW_NAME, w, h)
                    self._window_sized = True
                cv2.imshow(WINDOW_NAME, composite)
                self._needs_redraw = False

            key = cv2.waitKey(30) & 0xFF

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return "quit"

            if key == 255:
                continue

            action = self._process_key(key)
            if action is not None:
                return action

    # ── Keyboard handling ───────────────────────────────────────────────

    def _process_key(self, key):
        if key in (13, 10, 32):  # Enter or Space
            return "accept"
        elif key == 83 or key == ord('d'):  # Right arrow
            return "skip"
        elif key == 81 or key == ord('a'):  # Left arrow
            return "back"
        elif key in (ord('n'), ord('N')):
            return "next_unannotated"
        elif key in (ord('b'), ord('B')):
            return "prev_unannotated"
        elif key in (ord('c'), ord('C'), 8, 127):  # C or Backspace
            return "clear"
        elif key in (ord('q'), ord('Q')):
            return "quit"
        elif key in (ord('h'), ord('H')):
            self.show_help = not self.show_help
            self._needs_redraw = True
        elif key == 27:  # Escape
            if self.mode == Mode.RING_AWAITING_END:
                self.mode = Mode.IDLE
                self._ring_start_candidate = None
                self.status_msg = ""
                self._needs_redraw = True
        return None

    # ── Mouse handling ──────────────────────────────────────────────────

    def _display_to_board(self, x, y):
        """Convert display pixel coords to board coords. Returns None if outside board panel."""
        if x < self.left_panel_w:
            return None, None
        panel_x = x - self.left_panel_w
        bx = int(panel_x / self.board_scale)
        by = int(y / self.board_scale)
        if bx < 0 or bx >= board_size or by < 0 or by >= board_size:
            return None, None
        return bx, by

    def _display_to_original(self, x, y):
        """Convert display pixel coords to original image coords. Returns None if outside left panel."""
        if x >= self.left_panel_w:
            return None, None
        ox = x / self.left_scale
        oy = y / self.left_scale
        return ox, oy

    def _hit_test(self, bx, by):
        """Returns (element_type, index) or (None, None)."""
        # 1. Corner LEDs
        for i, c in enumerate(self.annotation.corners):
            cx, cy = self.annotation.to_board(*c["position"])
            if (bx - cx) ** 2 + (by - cy) ** 2 <= HIT_CORNER ** 2:
                return "corner", i

        # 2. Counter LEDs (individual)
        for i, (cx, cy) in enumerate(COUNTER_POS):
            if (bx - cx) ** 2 + (by - cy) ** 2 <= HIT_COUNTER ** 2:
                return "counter", i

        # 3. Counter bounding box (outside LED circles → toggle visibility)
        x1, y1, x2, y2 = COUNTER_BBOX
        if x1 <= bx <= x2 and y1 <= by <= y2:
            return "counter_bbox", 0

        # 4. ArUco region
        if ARUCO_X1 <= bx <= ARUCO_X2 and ARUCO_Y1 <= by <= ARUCO_Y2:
            return "aruco", 0

        # 5. Ring LEDs (nearest within radius)
        best_i, best_d = 0, float('inf')
        for i, (rx, ry) in enumerate(RING_POS):
            d = (bx - rx) ** 2 + (by - ry) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        if best_d <= HIT_RING ** 2:
            return "ring", best_i

        return None, None

    def _hit_test_corner_original(self, ox, oy):
        """Hit-test corners in original image coordinates."""
        img_h, img_w = self._current_image.shape[:2]
        hit_r = 2 * int(max(0.01 * min(img_w, img_h), 2))
        for i, c in enumerate(self.annotation.corners):
            cx, cy = c["position"]
            if (ox - cx) ** 2 + (oy - cy) ** 2 <= hit_r ** 2:
                return i
        return None

    def _mouse_callback(self, event, x, y, flags, param):
        # Try left panel (original image) — corners only
        ox, oy = self._display_to_original(x, y)
        if ox is not None:
            if event == cv2.EVENT_LBUTTONDOWN:
                idx = self._hit_test_corner_original(ox, oy)
                if idx is not None:
                    self._drag_idx = idx
                    self._drag_start_original = (ox, oy)
                    self._drag_started = False
                    self._drag_on_left = True
            elif event == cv2.EVENT_MOUSEMOVE and self._drag_on_left:
                if self._drag_idx is not None and (flags & cv2.EVENT_FLAG_LBUTTON):
                    dx = ox - self._drag_start_original[0]
                    dy = oy - self._drag_start_original[1]
                    img_h, img_w = self._current_image.shape[:2]
                    threshold = int(max(0.01 * min(img_w, img_h), DRAG_THRESHOLD))
                    if not self._drag_started and (dx * dx + dy * dy) > threshold ** 2:
                        self._drag_started = True
                        self.mode = Mode.DRAGGING_CORNER
                    if self._drag_started:
                        self.annotation.corners[self._drag_idx]["position"] = [ox, oy]
                        self._needs_redraw = True
            elif event == cv2.EVENT_LBUTTONUP and self._drag_on_left:
                if self._drag_idx is not None:
                    if not self._drag_started:
                        self._cycle_corner_state(self._drag_idx)
                    if all(c["visible"] for c in self.annotation.corners):
                        self._recompute_homography(self._current_image)
                    self._drag_idx = None
                    self._drag_started = False
                    self._drag_on_left = False
                    self.mode = Mode.IDLE
                    self._needs_redraw = True
            return

        # Right panel (rectified board)
        bx, by = self._display_to_board(x, y)
        if bx is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag_on_left = False
            self._on_left_down(bx, by)
        elif event == cv2.EVENT_MOUSEMOVE:
            self._on_mouse_move(bx, by, flags)
        elif event == cv2.EVENT_LBUTTONUP:
            self._on_left_up(bx, by)

    def _on_left_down(self, bx, by):
        if self.mode == Mode.RING_AWAITING_END:
            elem, idx = self._hit_test(bx, by)
            if elem == "ring":
                self.annotation.ring_end = idx
                self.mode = Mode.IDLE
                self._ring_start_candidate = None
                self.status_msg = ""
                self._needs_redraw = True
            return

        elem, idx = self._hit_test(bx, by)
        if elem == "corner":
            self._drag_idx = idx
            self._drag_start = (bx, by)
            self._drag_started = False
        elif elem == "counter":
            self._toggle_counter_led(idx)
        elif elem == "counter_bbox":
            self._toggle_counter_visibility()
        elif elem == "aruco":
            self._cycle_aruco()
        elif elem == "ring":
            self._handle_ring_first_click(idx)

    def _on_mouse_move(self, bx, by, flags):
        if self._drag_idx is not None and (flags & cv2.EVENT_FLAG_LBUTTON):
            dx = bx - self._drag_start[0]
            dy = by - self._drag_start[1]
            if not self._drag_started and (dx * dx + dy * dy) > DRAG_THRESHOLD ** 2:
                self._drag_started = True
                self.mode = Mode.DRAGGING_CORNER
            if self._drag_started:
                cbx = max(0, min(board_size - 1, bx))
                cby = max(0, min(board_size - 1, by))
                self.annotation.corners[self._drag_idx]["position"] = \
                    self.annotation.to_original(cbx, cby)
                self._needs_redraw = True

    def _on_left_up(self, bx, by):
        if self._drag_idx is not None:
            if not self._drag_started:
                self._cycle_corner_state(self._drag_idx)
            if all(c["visible"] for c in self.annotation.corners):
                self._recompute_homography(self._current_image)
            self._drag_idx = None
            self._drag_started = False
            self.mode = Mode.IDLE
            self._needs_redraw = True

    def _cycle_corner_state(self, idx):
        c = self.annotation.corners[idx]
        c["visible"] = not c["visible"]
        self._needs_redraw = True

    def _toggle_counter_led(self, idx):
        if self.annotation.counter_visible:
            self.annotation.counter_leds[idx] = not self.annotation.counter_leds[idx]
            self.annotation.recompute_counter()
        else:
            self.annotation.counter_visible = True
        self._needs_redraw = True

    def _toggle_counter_visibility(self):
        self.annotation.counter_visible = not self.annotation.counter_visible
        self._needs_redraw = True

    def _cycle_aruco(self):
        ann = self.annotation
        if not ann.aruco_visible:
            ann.aruco_visible = True
            ann.aruco_id = ARUCO_IDS[0]
        else:
            try:
                idx = ARUCO_IDS.index(ann.aruco_id)
                if idx + 1 < len(ARUCO_IDS):
                    ann.aruco_id = ARUCO_IDS[idx + 1]
                else:
                    ann.aruco_visible = False
            except ValueError:
                ann.aruco_visible = False
        self._needs_redraw = True

    def _recompute_homography(self, original):
        """Recompute homography and re-warp rectified image from all 4 visible corners."""
        ann = self.annotation
        visible = [c for c in ann.corners if c["visible"]]
        if len(visible) < 4:
            return
        src = np.array([c["position"] for c in ann.corners], dtype=np.float32)
        H = cv2.getPerspectiveTransform(src, corner_dots)
        ann.homography = H.tolist()
        mask = original[:, :, 2]  # red channel
        self.stats["rectified"] = cv2.warpPerspective(mask, H, (board_size, board_size))

    def _handle_ring_first_click(self, idx):
        """First ring click: set the first ON LED, then wait for first OFF LED."""
        self.annotation.ring_start = idx
        self._ring_start_candidate = idx
        self.mode = Mode.RING_AWAITING_END
        self.status_msg = f"Ring: first ON={idx}, click first OFF LED..."
        self._needs_redraw = True

    # ── Rendering ───────────────────────────────────────────────────────

    def _render(self, original, rel_path):
        """Build the composite side-by-side display image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        ann = self.annotation

        # Left panel: original image with corner LEDs
        left = original.copy()
        left_h, left_w = left.shape[:2]
        corner_radius = int(max(0.01 * min(left_w, left_h), 2))
        overlay = left.copy()
        for i, c in enumerate(ann.corners):
            cx, cy = int(round(c["position"][0])), int(round(c["position"][1]))
            color = COLOR_ON if c["visible"] else COLOR_NOT_VIS
            cv2.circle(overlay, (cx, cy), corner_radius, (0, 255, 255), -1)
            cv2.circle(left, (cx, cy), corner_radius, color, 2)
            cv2.putText(left, str(i), (cx + corner_radius + 4, cy + 5),
                        font, 0.5, color, 1)
        cv2.addWeighted(overlay, 0.35, left, 0.65, 0, dst=left)

        # Right panel: rectified board with overlays
        rectified = self.stats.get("rectified")
        if rectified is not None:
            if len(rectified.shape) == 2:
                board_img = cv2.cvtColor(rectified, cv2.COLOR_GRAY2BGR)
            else:
                board_img = rectified.copy()
        else:
            board_img = np.zeros((board_size, board_size, 3), dtype=np.uint8)

        # ArUco overlay (always shown)
        if ann.aruco_visible:
            marker_size = ARUCO_X2 - ARUCO_X1
            marker = cv2.aruco.generateImageMarker(dictionary, ann.aruco_id, marker_size)
            marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
            board_img[ARUCO_Y1:ARUCO_Y2, ARUCO_X1:ARUCO_X2] = cv2.addWeighted(
                board_img[ARUCO_Y1:ARUCO_Y2, ARUCO_X1:ARUCO_X2], 0.5, marker_bgr, 0.5, 0)
        else:
            # Grey box with diagonal cross
            cv2.rectangle(board_img, (ARUCO_X1, ARUCO_Y1), (ARUCO_X2, ARUCO_Y2), COLOR_NOT_VIS, 2)
            cv2.line(board_img, (ARUCO_X1, ARUCO_Y1), (ARUCO_X2, ARUCO_Y2), COLOR_NOT_VIS, 2)
            cv2.line(board_img, (ARUCO_X2, ARUCO_Y1), (ARUCO_X1, ARUCO_Y2), COLOR_NOT_VIS, 2)

        # ArUco label
        if ann.aruco_visible:
            aruco_label = f"ArUco ID {ann.aruco_id}"
        else:
            aruco_label = "ArUco: none"
        cv2.putText(board_img, aruco_label,(ARUCO_X1, ARUCO_Y1 - 8), font, 0.5, COLOR_BOARD_TEXT, 1)

        # Corner LEDs (positions stored in original image space, transform to board)
        for i, c in enumerate(ann.corners):
            cx, cy = ann.to_board(*c["position"])
            color = COLOR_ON if c["visible"] else COLOR_NOT_VIS
            cv2.circle(board_img, (cx, cy), led_size + 4, color, 2)
            cv2.putText(board_img, str(i), (cx + 14, cy + 5), font, 0.4, color, 1)

        # Counter bounding box
        bx1, by1, bx2, by2 = COUNTER_BBOX
        bbox_color = COLOR_BOARD_TEXT if ann.counter_visible else COLOR_NOT_VIS
        cv2.rectangle(board_img, (bx1, by1), (bx2, by2), bbox_color, 1)

        # Counter LEDs
        for i, (cx, cy) in enumerate(COUNTER_POS):
            if not ann.counter_visible:
                color = COLOR_NOT_VIS
            elif ann.counter_leds[i]:
                color = COLOR_ON
            else:
                color = COLOR_OFF
            cv2.circle(board_img, (cx, cy), led_size, color, 1)

        # Counter value text
        if ann.counter_visible:
            counter_text = f"Counter: {ann.counter_value}"
        else:
            counter_text = "Counter: n/a"
        cv2.putText(board_img, counter_text, (bx1, by1 - 8), font, 0.5, COLOR_BOARD_TEXT, 1)

        # Ring LEDs
        for i, (rx, ry) in enumerate(RING_POS):
            if ann.ring_start == ann.ring_end:
                color = COLOR_NOT_VIS
            else:
                in_arc = self._led_in_ring_arc(i, ann.ring_start, ann.ring_end)
                color = COLOR_ON if in_arc else COLOR_OFF
            if self.mode == Mode.RING_AWAITING_END and i == self._ring_start_candidate:
                color = COLOR_RING_SEL
            cv2.circle(board_img, (rx, ry), led_size, color, 1)

        # Annotation status label in top-right corner
        is_annotated = rel_path in self.ground_truth["images"]
        annotation_label = "Annotated" if is_annotated else "UNANNOTATED"
        font_scale, thickness = 0.6, (1 if is_annotated else 2)
        (tw, th), _ = cv2.getTextSize(annotation_label, font, font_scale, thickness)
        margin = 10
        tx = board_size - tw - margin
        ty = th + margin
        annotation_color = (0, 255, 0) if is_annotated else (0, 0, 255)
        cv2.putText(board_img, annotation_label, (tx, ty), font, font_scale, annotation_color, thickness)

        # Scale both to common height
        h = TARGET_HEIGHT
        left_h, left_w = left.shape[:2]
        left_scaled = cv2.resize(left, (int(left_w * h / left_h), h))
        board_scaled = cv2.resize(board_img, (int(board_size * h / board_size), h))

        self.left_panel_w = left_scaled.shape[1]
        self.left_scale = h / left_h
        self.board_scale = h / board_size

        # Status bar (3 rows: info, shortcuts, legend)
        row_h = 22
        status_h = row_h * 3 + 8
        total_w = left_scaled.shape[1] + board_scaled.shape[1]
        status_bar = np.full((status_h, total_w, 3), COLOR_STATUS_BG, dtype=np.uint8)

        # Row 1: progress + file path
        n_annotated = len(self.ground_truth["images"])
        progress = f"[{self.current_idx + 1}/{len(self.images)}] ({n_annotated} annotated)"
        info = f"{progress}  {rel_path}"
        cv2.putText(status_bar, info, (8, row_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1)

        if self.status_msg:
            cv2.putText(status_bar, self.status_msg, (total_w - 380, row_h - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 128, 0), 1)

        # Row 2: keyboard shortcuts
        shortcuts = "Enter=Accept  Left/Right=Prev/Next  N/B=Skip to unannotated  C=Clear  Q=Quit  H=Help"
        cv2.putText(status_bar, shortcuts, (8, row_h * 2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

        # Row 3: color legend
        legend_y = row_h * 3 - 4
        legend_items = [
            (COLOR_ON, "ON"),
            (COLOR_OFF, "OFF"),
            (COLOR_NOT_VIS, "Hidden"),
            (COLOR_RING_SEL, "Selection"),
        ]
        lx = 8
        for color, label in legend_items:
            cv2.circle(status_bar, (lx + 6, legend_y - 4), 5, color, -1)
            cv2.putText(status_bar, label, (lx + 16, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            lx += 16 + len(label) * 10 + 15

        composite = np.vstack([
            np.hstack([left_scaled, board_scaled]),
            status_bar,
        ])

        if self.show_help:
            self._draw_help(composite)

        return composite

    @staticmethod
    def _led_in_ring_arc(idx, start, end):
        """Check if LED is ON in the clockwise arc from start (first ON) to end (first OFF).

        The arc goes clockwise (decreasing index): start, start-1, ..., end+1.
        LED at index `end` is the first OFF and is excluded.
        """
        if start == end:
            return False  # undecodable
        if start > end:
            # Non-wrapping: ON indices are end+1, end+2, ..., start
            return end < idx <= start
        else:
            # Wrapping past 0: indices ..., 1, 0, 99, ...
            return idx <= start or idx > end

    @staticmethod
    def _draw_help(img):
        """Draw semi-transparent help overlay."""
        overlay = img.copy()
        h, w = img.shape[:2]
        pad = 20
        box_w, box_h = 380, len(HELP_TEXT) * 22 + 2 * pad
        x0 = (w - box_w) // 2
        y0 = (h - box_h) // 2
        cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, dst=img)
        for i, line in enumerate(HELP_TEXT):
            cv2.putText(img, line, (x0 + pad, y0 + pad + 18 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # ── Persistence ─────────────────────────────────────────────────────

    def _load_ground_truth(self):
        if self.output_path.exists():
            with open(self.output_path) as f:
                self.ground_truth = json.load(f)
            if "images" not in self.ground_truth:
                self.ground_truth = {"images": {}}
            n = len(self.ground_truth["images"])
            print(f"Loaded {n} existing annotations from {self.output_path}")

    def _save_ground_truth(self):
        with open(self.output_path, 'w') as f:
            json.dump(self.ground_truth, f, indent=2)

    def _find_unannotated(self, from_idx, forward=True):
        """Find the next/previous unannotated image index. Returns from_idx if none found."""
        step = 1 if forward else -1
        idx = from_idx + step
        while 0 <= idx < len(self.images):
            rel = str(self.images[idx].relative_to(self.data_dir))
            if rel not in self.ground_truth["images"]:
                return idx
            idx += step
        return from_idx



# ── CLI entry point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Annotate and verify RocSync benchmark images")
    parser.add_argument("data_dir", nargs="?", default="validation_data",
                        help="Path to validation data directory (default: validation_data)")
    parser.add_argument("--try-hard", action="store_true",
                        help="Enable try_hard mode for pipeline")
    parser.add_argument("-o", "--output", default=None,
                        help="Output ground truth JSON (default: <data_dir>/ground_truth.json)")
    args = parser.parse_args()

    tool = AnnotationTool(args.data_dir, output_path=args.output, try_hard=args.try_hard)
    tool.run()


if __name__ == "__main__":
    main()
