from dataclasses import asdict, dataclass


@dataclass
class VideoStatistics:
    # Frame counts
    n_frames: int
    n_considered_frames: int
    n_rejected_frames: int

    # Scores
    r2_before: float
    rmse_before: float
    r2_after: float
    rmse_after: float

    # Duration and FPS
    expected_duration: float
    measured_duration: float
    expected_fps: float
    measured_fps: float
    speed_factor: float

    # Start and end
    first_frame: float
    last_frame: float

    # Exposure
    mean_exposure_time: float
    min_exposure_time: float
    max_exposure_time: float
    std_exposure_time: float

    # Timestamps
    considered_timestamps: dict
    rejected_timestamps: dict
    # interpolated_timestamps: list

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        format_str = "{:<40} {:>30}"
        lines = []
        lines.append(71 * "-")
        lines.append(
            format_str.format(
                "Number of considered frames", f"{self.n_considered_frames}"
            )
        )
        lines.append(
            format_str.format(
                "Number of rejected outliers", f"{self.n_rejected_frames}"
            )
        )
        lines.append(
            format_str.format(
                "R2 (before/after outlier rejection)",
                f"{self.r2_before:.4f}/{self.r2_after:.4f}",
            )
        )
        lines.append(
            format_str.format(
                "RMSE (before/after outlier rejection)",
                f"{self.rmse_before:.2f}/{self.rmse_after:.2f} ms",
            )
        )
        lines.append(
            format_str.format("First frame:", f"{self.first_frame / 1000:.3f} s")
        )
        lines.append(
            format_str.format("Last frame:", f"{self.last_frame / 1000:.3f} s")
        )
        lines.append(
            format_str.format(
                "Framerate (expected/measured):",
                f"{self.expected_fps:.3f}/{self.measured_fps:.3f} fps ({self.speed_factor:.6f}x)",
            )
        )
        lines.append(
            format_str.format(
                "Duration (expected/measured):",
                f"{self.expected_duration / 1000:.3f}/{self.measured_duration / 1000:.3f} s (Δ={self.measured_duration - self.expected_duration:.2f} ms)",
            )
        )
        lines.append(
            format_str.format(
                "Exposure time (mean/min/max/std):",
                f"{self.mean_exposure_time:.2f}/{self.min_exposure_time:.2f}/{self.max_exposure_time:.2f}/{self.std_exposure_time:.2f} ms",
            )
        )
        lines.append(71 * "-")
        return "\n".join(lines)
