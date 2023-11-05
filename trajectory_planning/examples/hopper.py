import numpy as np

target_points = np.array(
    [
        [0, 0, 0],
        [9, 0, 50],
        [45, 0, 50],
        [81, 0, 50],
        [90, 0, 0],
    ]
)

states = {
    "x": target_points[:, 0],
    "y": target_points[:, 1],
    "z": target_points[:, 2],
}

total_time = 35
