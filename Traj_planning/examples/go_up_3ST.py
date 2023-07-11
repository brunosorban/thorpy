import sys
import numpy as np
from Traj_planning.traj_3ST.traj_generator_3ST import *

sys.path.append("../")
sys.path.append("../Traj_planning")

from parameters import *
from Traj_planning.examples.simple_circ import *

v = 15
r = 50

# constraints
max_v = 150
max_a = 2 * g

target_points = np.array(
    [
        [0, 0, 0],
        [0, 100, 0],
    ]
)

gamma_points_deg = np.array([90, 90])
gamma_dot_points = np.array([0, 0])
target_velocities = np.array([[0, 0, 0], [0, 0, 0]])  # m/s
target_accelerations = np.array([[0, 0, 0], [0, 0, 0]])  # m/s^2
time_points = np.array([0, 15])  # time list
dt = 0.01  # time step

constraints = {
    "max_v": max_v,
    "max_a": max_a,
    "g": g,
}

states = {
    "t": time_points,
    "x": target_points[:, 0],
    "y": target_points[:, 1],
    "z": target_points[:, 2],
    "vx": target_velocities[:, 0],
    "vy": target_velocities[:, 1],
    "vz": target_velocities[:, 2],
    "ax": target_accelerations[:, 0],
    "ay": target_accelerations[:, 1],
    "az": target_accelerations[:, 2],
    "gamma_dot": gamma_dot_points,
}


def traj_go_up_3ST():
    traj_params = trajenerator_3ST(
        states, constraints, env_params, rocket_params, controller_params
    )
    return traj_params
