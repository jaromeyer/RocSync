import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import root_mean_squared_error
from tqdm import tqdm

# Board layout
frquency = 1000
board_size = 250
period = 100
led_size = 3
ir_radius = 110

maker_format = [
    "host_timestamp",
    "ftk_timestamp",
    "type",
    "marker_id",
    "x_position",
    "y_position",
    "z_position",
    "r00",
    "r01",
    "r02",
    "r10",
    "r11",
    "r12",
    "r20",
    "r21",
    "r22",
    "registration_error",
]

fiducial_format = [
    "host_timestamp",
    "ftk_timestamp",
    "type",
    "x_position",
    "y_position",
    "z_position",
    "triangulation_error",
]


def read_ring(
    fiducials: list[tuple[float, float]], draw_result: bool = False
) -> tuple[int, int] | None:
    if draw_result:
        plt.figure(figsize=(6, 6))
        fiducials_arr = np.array(fiducials)
        plt.scatter(fiducials_arr[:, 0], fiducials_arr[:, 1], color="green")

    radius = ir_radius
    leds = np.zeros(period, dtype=bool)
    for i in range(period):
        angle = -(i / period + 0.25) * 2 * math.pi
        x = board_size / 2 + radius * math.cos(angle)
        y = board_size / 2 + radius * math.sin(angle)

        # Check if any fiducial matches this LED
        for fiducial in fiducials:
            distance = np.linalg.norm(fiducial - np.array([x, y]))
            if distance < led_size:
                leds[i] = True
                break

        if draw_result:
            circle = plt.Circle((x, y), led_size, color="red", fill=False)
            plt.gca().add_patch(circle)
    if draw_result:
        plt.gca().invert_yaxis()
        plt.grid(True)
        plt.show()

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


def read_counter(fiducials: list[tuple[float, float]], draw_result: bool = False):
    y = 48
    counter = 0

    # plt.figure(figsize=(6, 6))
    # fiducials_arr = np.array(fiducials)
    # plt.scatter(fiducials_arr[:, 0], fiducials_arr[:, 1], color="green")

    for i in range(0, 16):
        x = 65 + i * 8

        enabled = False
        for fiducial in fiducials:
            distance = np.linalg.norm(fiducial - np.array([x, y]))
            if distance < led_size:
                enabled = True
                break
        if enabled:
            counter += 2 ** (15 - i)

        # circle = plt.Circle((x, y), led_size, color="red", fill=False)
        # plt.gca().add_patch(circle)

    # plt.gca().invert_yaxis()
    # plt.grid(True)
    # plt.show()
    return counter


def process_frame(
    position: np.ndarray, rotation_matrix: np.ndarray, fiducials: list[dict]
) -> tuple[int, int] | None:
    # Transform fiducials into local coordinate system
    transformed_fiducials = []
    inv_rotation = np.linalg.inv(rotation_matrix)
    for fiducial in fiducials:
        fid_pos_world = np.array(
            [
                float(fiducial["x_position"]),
                float(fiducial["y_position"]),
                float(fiducial["z_position"]),
                1.0,
            ]
        )
        fid_pos_marker = inv_rotation @ (fid_pos_world - position)
        fid_pos_marker[:2] += np.array([5, 5])  # Adjust for PCB origin

        # Filter fiducials within the PCB area
        if (
            abs(fid_pos_marker[2]) < 5
            and 0 < fid_pos_marker[0] < 250
            and 0 < fid_pos_marker[1] < 250
        ):
            transformed_fiducials.append(fid_pos_marker[:2])

    # Rotate until counter is readable
    counter = read_counter(transformed_fiducials)
    for _ in range(3):
        if counter == 0:
            # Rotate 90 degrees arround center
            rot90 = np.array([[0, -1], [1, 0]])
            rotated_fiducials = []
            center = np.array([125, 125])
            for f in transformed_fiducials:
                v = f - center
                rotated = rot90 @ v + center
                rotated_fiducials.append(rotated)
            transformed_fiducials = rotated_fiducials
            counter = read_counter(transformed_fiducials)

    if counter == 0:
        return None

    ring = read_ring(transformed_fiducials)
    if ring is not None:
        start, end = ring
        # Check if the values are valid
        if (
            start > end
            or min(start, period - start) <= 2
            or min(period, 100 - end) <= 2
        ):
            return None

        start += counter * period
        end += counter * period
        return start, end
    return None


def fit_ftk_timestamps(timestamps: dict[int, tuple[int, int]]) -> dict:
    # Assuming constant frame rate, fit robust linear model
    x = np.array(list(timestamps.keys())).reshape(-1, 1)
    y = np.array([start for start, _ in timestamps.values()])
    model = RANSACRegressor(
        residual_threshold=10,
        max_trials=1000,  # more trials for more consistent results
        random_state=0,  # deterministic results
    )
    # Convert x to numeric type before fitting
    model.fit(x, y)
    print(model.estimator_.coef_)
    print(model.estimator_.intercept_)

    # Remove outliers
    filtered_timestamps = {
        k: v for i, (k, v) in enumerate(timestamps.items()) if model.inlier_mask_[i]
    }

    filtered_x = np.array(list(filtered_timestamps.keys())).reshape(-1, 1)
    filtered_y = np.array([start for start, _ in filtered_timestamps.values()])

    rmse = root_mean_squared_error(filtered_y, model.predict(filtered_x))
    results = {
        "slope": model.estimator_.coef_[0],
        "intercept": model.estimator_.intercept_,
        "rmse_after": rmse,
        "n_considered_frames": len(filtered_timestamps),
        "n_rejected_frames": len(timestamps) - len(filtered_timestamps),
    }
    return results


def process_ftk_recording(filename: str) -> dict:
    with open(filename, "r") as file:
        total_lines = sum(1 for _ in file)

    timestamps = {}
    with open(filename, "r") as file:
        with tqdm(total=total_lines, desc="Processing lines") as pbar:
            while True:
                line = file.readline()
                if not line:
                    break
                pbar.update(1)
                fields = line.strip().split(",")
                if len(fields) >= 4 and fields[2] == "m" and fields[3] == "240":
                    marker_dict = dict(zip(maker_format, fields[: len(maker_format)]))

                    # Read and collect all related fiducials (type "f") immediately after this marker
                    fiducials = []
                    current_pos = file.tell()
                    while True:
                        fid_line = file.readline()
                        if not fid_line:
                            break
                        pbar.update(1)
                        fid_fields = fid_line.strip().split(",")
                        if fid_fields[2] != "f":
                            # Not a fiducial, stop collecting
                            file.seek(current_pos)
                            break
                        fiducial_dict = dict(
                            zip(fiducial_format, fid_fields[: len(fiducial_format)])
                        )
                        fiducials.append(fiducial_dict)
                        current_pos = file.tell()

                    position = np.array(
                        [
                            float(marker_dict["x_position"]),
                            float(marker_dict["y_position"]),
                            float(marker_dict["z_position"]),
                            1.0,
                        ]
                    )

                    # Make sure z-axis is pointing towards the camera
                    if float(marker_dict["r22"]) < 0:
                        rotation_matrix = np.array(
                            [
                                [
                                    float(marker_dict["r01"]),
                                    float(marker_dict["r00"]),
                                    float(marker_dict["r02"]),
                                    0.0,
                                ],
                                [
                                    float(marker_dict["r11"]),
                                    float(marker_dict["r10"]),
                                    float(marker_dict["r12"]),
                                    0.0,
                                ],
                                [
                                    float(marker_dict["r21"]),
                                    float(marker_dict["r20"]),
                                    float(marker_dict["r22"]),
                                    0.0,
                                ],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    else:
                        rotation_matrix = np.array(
                            [
                                [
                                    float(marker_dict["r00"]),
                                    float(marker_dict["r01"]),
                                    float(marker_dict["r02"]),
                                    0.0,
                                ],
                                [
                                    float(marker_dict["r10"]),
                                    float(marker_dict["r11"]),
                                    float(marker_dict["r12"]),
                                    0.0,
                                ],
                                [
                                    float(marker_dict["r20"]),
                                    float(marker_dict["r21"]),
                                    float(marker_dict["r22"]),
                                    0.0,
                                ],
                                [0.0, 0.0, 0.0, 1.0],
                            ]
                        )
                    result = process_frame(position, rotation_matrix, fiducials)
                    if result is not None:
                        timestamps[int(marker_dict["ftk_timestamp"])] = result
    if len(timestamps) > 0:
        statistics = fit_ftk_timestamps(timestamps)
        return statistics
    return None
