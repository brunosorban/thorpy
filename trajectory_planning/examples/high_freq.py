import numpy as np

target_points = np.array(
    [
        [0, 0, 0],
        [20, 50, 0],
        [40, 20, 0],
        [60, 50, 0],
        [80, 20, 0],
        [100, 50, 0],
        [120, 0, 0],
    ]
)

states = {
    "x": target_points[:, 0],
    "y": target_points[:, 1],
    "z": target_points[:, 2],
}

total_time = 35
