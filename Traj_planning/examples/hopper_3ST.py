import sys
import numpy as np
from Traj_planning.traj_3ST.traj_generator_3ST import *

sys.path.append("../")
sys.path.append("../Traj_planning")

from parameters import *

target_points = np.array(
    [
        [0, 0, 0],
        [20, 50, 0],
        [50, 50, 0],
        [80, 50, 0],
        [100, 0, 0],
    ]
)

time_points = 5 * np.arange(0, len(target_points), 1)  # time list

states = {
    "t": time_points,
    "x": target_points[:, 0],
    "y": target_points[:, 1],
    "z": target_points[:, 2],
}


def traj_hopper_3ST():
    traj_params = trajenerator_3ST(states, env_params, rocket_params, controller_params)
    return traj_params
