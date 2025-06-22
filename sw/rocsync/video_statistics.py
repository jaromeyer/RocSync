from dataclasses import asdict, dataclass


@dataclass
class VideoMetadata:
    path: str
    fps: float
    duration_ms: float
    n_frames: int


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
        data = [
            (
                "Number of considered frames",
                f"{self.n_considered_frames}",
                self.n_considered_frames < 10,
            ),
            (
                "Number of rejected outliers",
                f"{self.n_rejected_frames}",
                self.n_rejected_frames > 0.1 * self.n_frames,
            ),
            (
                "R2 (before/after outlier rejection)",
                f"{self.r2_before:.4f}/{self.r2_after:.4f}",
                self.r2_after < 0.99,
            ),
            (
                "RMSE (before/after outlier rejection)",
                f"{self.rmse_before:.2f}/{self.rmse_after:.2f} ms",
                self.rmse_after > 2,
            ),
            ("First frame", f"{self.first_frame / 1000:.3f} s", True),
            ("Last frame", f"{self.last_frame / 1000:.3f} s", False),
            (
                "Framerate (expected/measured)",
                f"{self.expected_fps:.3f}/{self.measured_fps:.3f} fps ({self.speed_factor:.6f}x)",
                False,
            ),
            (
                "Duration (expected/measured)",
                f"{self.expected_duration / 1000:.3f}/{self.measured_duration / 1000:.3f} s (Δ={self.measured_duration - self.expected_duration:.2f} ms)",
                False,
            ),
            (
                "Exposure time (mean/min/max/std)",
                f"{self.mean_exposure_time:.2f}/{self.min_exposure_time:.2f}/{self.max_exposure_time:.2f}/{self.std_exposure_time:.2f} ms",
                False,
            ),
        ]

        # Calculate max column widths
        label_width = max(len(label) for label, _, _ in data)
        value_width = max(len(value) for _, value, _ in data)
        format_str = f"{{:<{label_width}}} {{:>{value_width}}}"
        separator = "-" * (label_width + value_width + 1)

        lines = [separator]
        for label, value, highlight in data:
            line = format_str.format(label, value)
            if highlight:
                line = "\033[91m" + line + "\033[0m"
            lines.append(line)
        lines.append(separator)

        return "\n".join(lines)
