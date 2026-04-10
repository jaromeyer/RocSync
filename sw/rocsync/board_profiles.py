from dataclasses import dataclass

import numpy as np

from rocsync.vision import CameraType

def _mm2px(mm, board_mm, board_size):
    return mm / board_mm * board_size


@dataclass(frozen=True)
class BoardProfile:
    name: str
    board_size: int                   # rectified image dimension (px)
    board_mm: float                   # physical board size (mm)
    aruco_marker_id: int
    period: int
    visible_radius: int
    ir_radius: int
    corner_dots: np.ndarray           # RGB corner LED positions in rectified image (px)
    ir_corners: np.ndarray            # IR corner positions for perspective transform (px, always 4)
    aruco_corners_coords: np.ndarray  # ArUco destination corners in rectified image (px, always 4)
    perspective_corner_slice: slice   # which corner_dots subset for the 4-point transform
    counter_led_coords: dict          # {CameraType: np.ndarray Nx2 int} LED positions (px)
    counter_bg_y: dict                # {CameraType: int} background sample y-coordinate (px)
    counter_bits: int

    def __hash__(self):
        return hash(self.name)


def _build_v1(board_size=640, board_mm=250):
    mm2px = lambda mm: _mm2px(mm, board_mm, board_size)

    corner_dots = np.array([
        [51, 51],
        [board_size - 52, 51],
        [board_size - 52, board_size - 52],
        [51, board_size - 52],
    ], dtype=np.float32)

    ir_corners = np.array([
        [13, 13],
        [board_size - 14, 13],
        [board_size - 14, board_size - 13],
        [13, board_size - 14],
    ], dtype=np.float32)

    aruco_corners_coords = np.array([
        [202, 202],
        [board_size - 203, 202],
        [board_size - 203, board_size - 203],
        [202, board_size - 203],
    ], dtype=np.float32)

    # Counter: 16 LEDs in a single row, x = (65 + i*8) mm
    n_leds = 16
    x_coords = np.array([int(mm2px(65 + i * 8)) for i in range(n_leds)])
    rgb_y = int(mm2px(53))
    ir_y = int(mm2px(48))

    rgb_leds = np.stack([x_coords, np.full(n_leds, rgb_y)], axis=1)
    ir_leds = np.stack([x_coords, np.full(n_leds, ir_y)], axis=1)

    # Background sampled at y - 25 px in original code
    # TODO fix bg location
    rgb_bg_y = rgb_y - 25
    ir_bg_y = ir_y - 25

    return BoardProfile(
        name="v1",
        board_size=board_size,
        board_mm=board_mm,
        aruco_marker_id=0,
        period=100,
        visible_radius=294,
        ir_radius=280,
        corner_dots=corner_dots,
        ir_corners=ir_corners,
        aruco_corners_coords=aruco_corners_coords,
        perspective_corner_slice=slice(None),
        counter_led_coords={CameraType.RGB: rgb_leds, CameraType.INFRARED: ir_leds},
        counter_bg_y={CameraType.RGB: rgb_bg_y, CameraType.INFRARED: ir_bg_y},
        counter_bits=16,
    )


def _build_v2(board_size=640, board_mm=250):
    mm2px = lambda mm: _mm2px(mm, board_mm, board_size)

    corner_dots = np.array([
        [13, 51],   # 5th additional RGB LED
        [51, 51],
        [board_size - 52, 51],
        [board_size - 52, board_size - 52],
        [51, board_size - 52],
    ], dtype=np.float32)

    # 4 standard corners for IR perspective transform
    # (v2 has a 5th IR LED at [51, 13] but it is not used for the transform)
    ir_corners = np.array([
        [13, 13],
        [board_size - 14, 13],
        [board_size - 14, board_size - 13],
        [13, board_size - 14],
    ], dtype=np.float32)

    aruco_corners_coords = np.array([
        [202, 202],
        [board_size - 203, 202],
        [board_size - 203, board_size - 203],
        [202, board_size - 203],
    ], dtype=np.float32)

    # Counter: 20 LEDs in 2 rows x 10, x from 71mm to 179mm
    n_per_row = 10
    x_coords = np.linspace(mm2px(71), mm2px(179), n_per_row).astype(int)
    rgb_y1 = round(mm2px(41))
    rgb_y2 = round(mm2px(53))
    ir_y1 = round(mm2px(47))
    ir_y2 = round(mm2px(59))

    rgb_leds = np.stack([
        np.tile(x_coords, 2),
        np.concatenate([np.full(n_per_row, rgb_y1), np.full(n_per_row, rgb_y2)]),
    ], axis=1)
    ir_leds = np.stack([
        np.tile(x_coords, 2),
        np.concatenate([np.full(n_per_row, ir_y1), np.full(n_per_row, ir_y2)]),
    ], axis=1)

    # TODO fix bg location
    bg_y = round(mm2px(34))

    return BoardProfile(
        name="v2",
        board_size=board_size,
        board_mm=board_mm,
        aruco_marker_id=21,
        period=100,
        visible_radius=294,
        ir_radius=280,
        corner_dots=corner_dots,
        ir_corners=ir_corners,
        aruco_corners_coords=aruco_corners_coords,
        perspective_corner_slice=slice(1, None),
        counter_led_coords={CameraType.RGB: rgb_leds, CameraType.INFRARED: ir_leds},
        counter_bg_y={CameraType.RGB: bg_y, CameraType.INFRARED: bg_y},
        counter_bits=20,
    )


BOARD_V1 = _build_v1()
BOARD_V2 = _build_v2()
ALL_PROFILES = [BOARD_V1, BOARD_V2]
PROFILES_BY_NAME = {p.name: p for p in ALL_PROFILES}
PROFILES_BY_ARUCO = {p.aruco_marker_id: p for p in ALL_PROFILES}
