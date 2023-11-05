import numpy as np
target_points = np.array(
    [
        [0, 0, 0],
        [0, 0, 10],
    ]
)

states = {
    "x": target_points[:, 0],
    "y": target_points[:, 1],
    "z": target_points[:, 2],
}

total_time = 10