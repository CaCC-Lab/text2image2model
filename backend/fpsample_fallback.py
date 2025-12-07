"""
Fallback implementation of fpsample for Windows compatibility.
Provides a pure NumPy implementation of Farthest Point Sampling (FPS).
"""
import numpy as np
from typing import Optional


def fps_sampling(points: np.ndarray, n_samples: int, start_idx: Optional[int] = None) -> np.ndarray:
    """
    Farthest Point Sampling (FPS) algorithm.

    Args:
        points: Input point cloud of shape (N, D) where N is number of points and D is dimension
        n_samples: Number of points to sample
        start_idx: Starting point index (random if None)

    Returns:
        Indices of sampled points of shape (n_samples,)
    """
    n_points = points.shape[0]

    if n_samples >= n_points:
        return np.arange(n_points)

    # Initialize
    if start_idx is None:
        start_idx = np.random.randint(0, n_points)

    selected_indices = np.zeros(n_samples, dtype=np.int64)
    selected_indices[0] = start_idx

    # Distance from each point to nearest selected point
    distances = np.full(n_points, np.inf)

    for i in range(1, n_samples):
        # Update distances based on last selected point
        last_selected = selected_indices[i - 1]
        dist_to_last = np.sum((points - points[last_selected]) ** 2, axis=1)
        distances = np.minimum(distances, dist_to_last)

        # Select the point with maximum distance
        selected_indices[i] = np.argmax(distances)

    return selected_indices


def fps_npdu_sampling(points: np.ndarray, n_samples: int, start_idx: Optional[int] = None) -> np.ndarray:
    """
    FPS sampling with NPDU (Non-Probabilistic Distance Uniform) variant.
    Falls back to standard FPS for compatibility.
    """
    return fps_sampling(points, n_samples, start_idx)


def bucket_fps_kdline_sampling(points: np.ndarray, n_samples: int, start_idx: Optional[int] = None) -> np.ndarray:
    """
    Bucket FPS with KD-line acceleration.
    Falls back to standard FPS for compatibility.
    """
    return fps_sampling(points, n_samples, start_idx)
