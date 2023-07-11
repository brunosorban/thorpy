import sys
import numpy as np
from Traj_planning.traj_3ST.traj_generator_3ST import *

sys.path.append("../")
sys.path.append("../Traj_planning")

from parameters import *

# constraints
max_vx = 15
max_vy = 15
max_vz = 15

min_vx = -max_vx
min_vy = -max_vy
min_vz = -max_vz

max_ax = 2 * g * np.sin(15 * np.pi / 180)
max_ay = g
max_az = 2 * g

min_ax = -max_ax
min_ay = (0.6 - 1) * g
min_az = -max_az

target_points = np.array(
    [
        [0, 0, 0],
        [20, 50, 0],
        [50, 50, 0],
        [80, 50, 0],
        [100, 0, 0],
    ]
)

# gamma_points_deg = np.array([90, 90, 90, 90])
gamma_dot_points = np.array([0, None, None, None, 0])
target_velocities = np.array(
    [[0, 0, 0], [None, None, None], [None, None, None], [None, None, None], [0, 0, 0]]
)  # m/s
target_accelerations = np.array(
    [[0, 0, 0], [None, None, None], [None, None, None], [None, None, None], [0, 0, 0]]
)  # m/s^2
time_points = 6 * np.array([0, 1, 2, 3, 4])  # time list
dt = 0.01  # time step

constraints = {
    "max_vx": max_vx,
    "max_vy": max_vy,
    "max_vz": max_vz,
    "min_vx": min_vx,
    "min_vy": min_vy,
    "min_vz": min_vz,
    "max_ax": max_ax,
    "max_ay": max_ay,
    "max_az": max_az,
    "min_ax": min_ax,
    "min_ay": min_ay,
    "min_az": min_az,
    "acceptable_offset": 5,
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


def traj_hopper_3ST():
    traj_params = trajenerator_3ST(
        states, constraints, env_params, rocket_params, controller_params
    )
    return traj_params
