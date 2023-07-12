import sys
import numpy as np
from Traj_planning.traj_3ST.traj_generator_3ST import *

sys.path.append("../")
sys.path.append("../Traj_planning")

from parameters import *
from Traj_planning.examples.simple_circ import *

# constraints
max_v = 150
max_a = 2 * g

target_points = np.array(
    [
        [0, 0, 0],
        [0, 100, 0],
    ]
)

time_points = 12 * np.arange(len(target_points))  # time list

states = {
    "t": time_points,
    "x": target_points[:, 0],
    "y": target_points[:, 1],
    "z": target_points[:, 2],
}


def traj_go_up_3ST():
    traj_params = trajenerator_3ST(
        states, env_params, rocket_params, controller_params
    )
    return traj_params
