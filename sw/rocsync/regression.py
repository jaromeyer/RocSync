import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error as root_mean_squared_error

from rocsync.printer import errprint, warnprint
from rocsync.video_statistics import VideoMetadata, VideoStatistics


def fit_timestamps(
    video: VideoMetadata, timestamps: dict[int, tuple[int, int]]
) -> VideoStatistics:
    if len(timestamps) == 0:
        errprint("Error: Unable to timestamp any frames.")
        return

    # Assuming constant frame rate, fit robust linear model
    x = np.array(list(timestamps.keys())).reshape(-1, 1)
    y = np.array([start for start, _ in timestamps.values()])
    model = RANSACRegressor(
        residual_threshold=1000 / video.fps,  # max one frame deviation
        max_trials=1000,  # more trials for more consistent results
        random_state=0,  # deterministic results
    )
    model.fit(x, y)

    # Assert that we have at least 80% inliers
    if np.sum(model.inlier_mask_) < 0.8 * len(timestamps):
        warnprint(
            f"WARNING: Estimated model has fewer than 80% inliers ({np.sum(model.inlier_mask_) / len(timestamps) * 100}%)."
        )

    # Predict timestamps for all frames
    x_range = np.arange(0, video.n_frames).reshape(-1, 1)
    y_pred = model.predict(x_range)

    # Add error to timestamps
    errors = model.predict(x) - y
    timestamps = {
        frame_number: (start, end, error)
        for (frame_number, (start, end)), error in zip(timestamps.items(), errors)
    }

    # Remove outliers
    filtered_timestamps = {
        k: v for i, (k, v) in enumerate(timestamps.items()) if model.inlier_mask_[i]
    }
    rejected_timestamps = {
        k: v for i, (k, v) in enumerate(timestamps.items()) if not model.inlier_mask_[i]
    }
    filtered_x = np.array(list(filtered_timestamps.keys())).reshape(-1, 1)
    filtered_y = np.array([start for start, _, _ in filtered_timestamps.values()])

    # Calculate statistics
    exposure_times = [end - start for start, end, _ in filtered_timestamps.values()]
    measured_duration = y_pred[-1] - y_pred[0]
    statistics = VideoStatistics(
        n_frames=video.n_frames,
        n_considered_frames=len(filtered_timestamps),
        n_rejected_frames=len(timestamps) - len(filtered_timestamps),
        r2_before=model.score(x, y),
        rmse_before=root_mean_squared_error(y, model.predict(x)),
        r2_after=model.score(filtered_x, filtered_y),
        rmse_after=root_mean_squared_error(filtered_y, model.predict(filtered_x)),
        expected_duration=video.duration_ms,
        measured_duration=measured_duration,
        expected_fps=video.fps,
        measured_fps=(video.n_frames - 1) / measured_duration * 1000,
        speed_factor=measured_duration / video.duration_ms,
        first_frame=y_pred[0],
        last_frame=y_pred[-1],
        mean_exposure_time=np.mean(exposure_times),
        min_exposure_time=np.min(exposure_times),
        max_exposure_time=np.max(exposure_times),
        std_exposure_time=np.std(exposure_times),
        considered_timestamps=filtered_timestamps,
        rejected_timestamps=rejected_timestamps,
        # interpolated_timestamps=y_pred.tolist(),
    )

    # if debug_dir:
    #     statistics.plot_timechart(
    #         filtered_x,
    #         filtered_y,
    #         x_range,
    #         y_pred,
    #         exposure_times,
    #         expected_duration,
    #         debug_dir,
    #     )
    #     statistics.plot_exposure_histogram(exposure_times, debug_dir)

    return statistics
