import math
from enum import Enum

import cv2
import numpy as np

from rocsync.printer import *


class CameraType(Enum):
    RGB = "rgb"
    INFRARED = "ir"


# Blob detector params
params = cv2.SimpleBlobDetector_Params()
params.filterByColor = True
params.blobColor = 255
params.filterByArea = False
params.minArea = 100
params.maxArea = 1000
params.filterByCircularity = True
params.minCircularity = 0.5
params.filterByInertia = True
params.minInertiaRatio = 0.5
params.filterByConvexity = True
params.minConvexity = 0.8
params.minDistBetweenBlobs = 0.01
blob_detector = cv2.SimpleBlobDetector_create(params)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
aruco_detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# Board layout
frquency = 1000
aruco_marker_id = 0
board_size = 640
period = 100
led_size = 8
ir_radius = 280
visible_radius = 294

ir_corners = np.array(
    [
        [13, 13],
        [board_size - 14, 13],
        [board_size - 14, board_size - 13],
        [13, board_size - 14],
    ],
    dtype=np.float32,
)

corner_dots = np.array(
    [
        [51, 51],
        [board_size - 52, 51],
        [board_size - 52, board_size - 52],
        [51, board_size - 52],
    ],
    dtype=np.float32,
)

aruco_corners_coords = np.array(
    [
        [202, 202],
        [board_size - 203, 202],
        [board_size - 203, board_size - 203],
        [202, board_size - 203],
    ],
    dtype=np.float32,
)


def draw_polygon(points, image, color):
    for i in range(len(points)):
        cv2.line(
            image,
            tuple(map(int, points[i])),
            tuple(map(int, points[(i + 1) % len(points)])),
            color,
            2,
        )


def read_ring(extracted_board, camera_type, draw_result=False):
    radius = visible_radius if camera_type == CameraType.RGB else ir_radius
    leds = np.zeros(period, dtype=bool)
    for i in range(period):
        angle = -(i / period + 0.25) * 2 * math.pi
        x = int(board_size / 2 + radius * math.cos(angle))
        y = int(board_size / 2 + radius * math.sin(angle))

        # Create mask and sample image
        led_mask = np.zeros((board_size, board_size), dtype=np.uint8)
        cv2.circle(led_mask, (x, y), led_size, (255), -1)
        mean_intensity = np.mean(extracted_board[led_mask > 0])
        leds[i] = mean_intensity > 20  # TODO make param

        if draw_result:
            color = (0, 0, 255) if leds[i] else (255, 0, 0)
            cv2.circle(extracted_board, (x, y), led_size, color, 1)

    potential_starts = []
    potential_ends = []
    for i in range(period):
        if not leds[(i - 1) % period] and leds[i]:
            potential_starts.append(i)
        if leds[i] and not leds[(i + 1) % period]:
            potential_ends.append(i)

    # Make sure that there is exactly ONE of enabled
    if len(potential_starts) == 1 and len(potential_ends) == 1:
        return (potential_starts[0], potential_ends[0])


def read_counter(extracted_board, camera_type, draw_result=False):
    y = int(53 / 250 * 640) if camera_type == CameraType.RGB else int(48 / 250 * 640)
    counter = 0
    for i in range(0, 16):
        x = int((65 + i * 8) / 250 * 640)

        # Create mask and sample image
        led_mask = np.zeros((board_size, board_size), dtype=np.uint8)
        cv2.circle(led_mask, (x, y), led_size, (255), -1)
        mean_intensity = np.mean(extracted_board[led_mask > 0])
        enabled = mean_intensity > 20  # TODO make param
        if enabled:
            counter += 2 ** (15 - i)
        if draw_result:
            color = (0, 0, 255) if enabled else (255, 0, 0)
            cv2.circle(extracted_board, (x, y), led_size, color, 1)
    return counter


def find_corners_convexhull(mask, frame_number, debug_dir=None):
    points = blob_detector.detect(mask)

    # Draw detected blobs as red circles
    debug_image = cv2.drawKeypoints(
        mask,
        points,
        np.array([]),
        (0, 0, 255),
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    points = [kp.pt for kp in points]

    # Find the convex hull and identify the corners
    if len(points) >= 4:
        hull = cv2.convexHull(np.array(points, dtype=np.float32))
        corners = hull.reshape(-1, 2)
        draw_polygon(corners, debug_image, (0, 255, 0))

        if len(hull) > 4:
            # Approximate to 4 points
            epsilon_factor = 0.02
            n_points = len(hull)
            approx_hull = hull
            while n_points > 4:
                epsilon = epsilon_factor * cv2.arcLength(hull, True)
                approx_hull = cv2.approxPolyDP(hull, epsilon, True)
                n_points = len(approx_hull)
                epsilon_factor += 0.02

            # Draw the approxcuimated convex hull
            corners = approx_hull.reshape(-1, 2)
            draw_polygon(corners, debug_image, (255, 0, 0))

    if debug_dir:
        cv2.imwrite(f"{debug_dir}/convexhull_{frame_number}.png", debug_image)
    if corners is not None and len(corners) == 4:
        return corners


def find_corners_dots(mask, frame_number, debug_dir=None):
    points = blob_detector.detect(mask)
    if not points:
        return
    closest_points = [
        min(points, key=lambda p: np.linalg.norm(p.pt - target)).pt for target in corner_dots
    ]
    max_distance = max([np.linalg.norm(act - exp) for act, exp in zip(closest_points, corner_dots)])
    if max_distance > 50:
        return  # Some corner is too far away from where it should be

    if debug_dir:
        debug_image = cv2.drawKeypoints(
            mask,
            points,
            np.array([]),
            (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        cv2.imwrite(f"{debug_dir}/corner_{frame_number}.png", debug_image)

    return np.array(closest_points, dtype=np.float32)


def find_corners_aruco(mask, frame_number, debug_dir=None):
    markers, marker_ids, _ = aruco_detector.detectMarkers(mask)
    if debug_dir:
        debug_image = mask.copy()
        cv2.aruco.drawDetectedMarkers(debug_image, markers, marker_ids)
        cv2.imwrite(f"{debug_dir}/aruco_{frame_number}.png", debug_image)

    if marker_ids is None:
        return
    marker_dict = {id[0]: marker for id, marker in zip(marker_ids, markers)}
    if aruco_marker_id in marker_dict.keys():
        return marker_dict[aruco_marker_id]


def process_frame(image, camera_type, frame_number, debug_dir=None):
    match camera_type:
        case CameraType.RGB:
            # First extract course PCB using ArUco marker
            aruco_corners = find_corners_aruco(image, frame_number, debug_dir)
            if aruco_corners is None:
                return False, None
            red_channel = image[:, :, 2]
            _, mask = cv2.threshold(red_channel, 220, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

            # Use course PCB to accurately extract corner
            rough_transformation_matrix = cv2.getPerspectiveTransform(
                aruco_corners, aruco_corners_coords
            )
            rough_pcb = cv2.warpPerspective(
                mask, rough_transformation_matrix, (board_size, board_size)
            )
            corners = find_corners_dots(rough_pcb, frame_number, debug_dir)
            if corners is None:
                return True, None
            transformation_matrix = np.dot(
                cv2.getPerspectiveTransform(corners, corner_dots),
                rough_transformation_matrix,
            )
            pcb = cv2.warpPerspective(mask, transformation_matrix, (board_size, board_size))
        case CameraType.INFRARED:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            corners = find_corners_convexhull(mask, frame_number, debug_dir)
            if corners is None:
                return False, None
            transformation_matrix = cv2.getPerspectiveTransform(corners, ir_corners)
            pcb = cv2.warpPerspective(mask, transformation_matrix, (board_size, board_size))

            # Find correct rotation
            for _ in range(4):
                if read_counter(pcb, CameraType.INFRARED) == 0:
                    pcb = cv2.rotate(pcb, cv2.ROTATE_90_CLOCKWISE)
            if read_counter(pcb, CameraType.INFRARED) == 0:
                return True, None  # Counter was actually 0, can't determine orientation

    if debug_dir:  # For RGB debug output
        pcb = cv2.cvtColor(pcb, cv2.COLOR_GRAY2BGR)

    counter = read_counter(pcb, camera_type, draw_result=True)
    ring = read_ring(pcb, camera_type, draw_result=True)

    if debug_dir:
        cv2.imwrite(f"{debug_dir}/leds_{frame_number}.png", pcb)

    if ring is not None:
        start, end = ring
        if start > end or min(start, period - start) <= 2 or min(period, 100 - end) <= 2:
            return True, None # Counter increment during exposure

        start += counter * period
        end += counter * period
        return True, (start, end)
