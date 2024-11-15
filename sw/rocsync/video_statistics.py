from dataclasses import asdict, dataclass


@dataclass
class VideoStatistics:
    n_frames: int
    n_considered_frames: int
    n_rejected_frames: int
    r2_before: float
    rmse_before: float
    r2_after: float
    rmse_after: float
    expected_duration: float
    measured_duration: float
    expected_fps: float
    measured_fps: float
    speed_factor: float
    first_frame: float
    last_frame: float
    mean_exposure_time: float
    min_exposure_time: float
    max_exposure_time: float
    std_exposure_time: float
    measured_timestamps: dict
    interpolated_timestamps: list

    def to_dict(self):
        return asdict(self)
