import sys
import numpy as np
from Traj_planning.traj_3ST.traj_generator_3ST import *

sys.path.append("../")
sys.path.append("../Traj_planning")

from parameters import *
from Traj_planning.examples.simple_circ import *

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

time_points = 7 * np.arange(len(target_points))  # time list

states = {
    "t": time_points,
    "x": target_points[:, 0],
    "y": target_points[:, 1],
    "z": target_points[:, 2],
}


def traj_M_3ST():
    traj_params = trajenerator_3ST(states, env_params, rocket_params, controller_params)
    return traj_params
