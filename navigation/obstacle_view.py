"""Floor rejection for the Go2 point clouds shown during navigation."""
import numpy as np


def obstacle_mask(heights):
    """Keep finite returns at least 10 cm above the calibrated floor, at any height."""
    heights = np.asarray(heights)
    return np.isfinite(heights) & (heights >= 0.10)
