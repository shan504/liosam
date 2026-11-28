"""The shared initial-plane and small-component policy for levelled displays."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'original_go2' / 'go2_3d'))
from ground import height_obstacle_mask
from point_noise import keep_connected


def obstacle_mask(heights):
    """Keep finite returns strictly above 15 cm over the saved initial plane."""
    return height_obstacle_mask(heights)

