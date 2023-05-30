import sys
import numpy as np

sys.path.append("../")
sys.path.append("../Traj_planning")
from Traj_planning.traj_generators.traj_planner import *

dt = 0.01
plot = False
t0 = 0  # beginning
dt1 = 23  # first jump
dt2 = 30  # second jump
dt3 = 3  # getting into inverted flight
dt4 = 10  # inverted flight upside down
dt5 = 15  # getting out of inverted flight
dt6 = 3  # returning to upside up
dt7 = 50  # slowing down
dt8 = 18  # landing

time_points = [
    t0,
    t0 + dt1,
    t0 + dt1 + dt2,
    t0 + dt1 + dt2 + dt3,
    t0 + dt1 + dt2 + dt3 + dt4,
    t0 + dt1 + dt2 + dt3 + dt4 + dt5,
    t0 + dt1 + dt2 + dt3 + dt4 + dt5 + dt6,
    t0 + dt1 + dt2 + dt3 + dt4 + dt5 + dt6 + dt7,
    t0 + dt1 + dt2 + dt3 + dt4 + dt5 + dt6 + dt7 + dt8,
]
g = 9.81  # gravity

# controller input bounds
max_acc_y = g  # maxium thrust rate
min_acc_y = (
    -2 * max_acc_y
)  # 0.35 * max_acc_y  - g # should be 30% to 40% of the max thrust
max_acc_x = max_acc_y
min_acc_x = -max_acc_x

points = np.array(
    [
        [0, 0, 0],  # beginning
        [100, 2000, 0],  # first jump
        [850, 9200, 0],  # second jump
        [900, 9500, 0],  # getting into inverted flight
        [1000, 10000, 0],  # inverted flight upside down
        [900, 9000, 0],  # getting out of inverted flight
        [850, 8650, 0],  # returning to upside up
        [500, 500, 0],  # slowing down
        [500, 0, 0],  # landing
    ]
)

gamma_points_deg = np.array(
    [
        90,  # beginning
        90,  # first jump
        90,  # second jump
        270,  # getting into inverted flight
        270,  # inverted flight upside down
        270,  # getting out of inverted flight
        90,  # returning to upside up
        90,  # slowing down
        90,  # landing
    ]
)

velocities = np.array(
    [
        [0, 0, None],  # beginning
        [None, None, None],  # first jump
        [None, 122, None],  # second jump
        [None, None, None],  # getting into inverted flight
        [0, 0, None],  # inverted flight upside down
        [None, None, None],  # getting out of inverted flight
        [None, None, None],  # returning to upside up
        [0, -30, None],  # slowing down
        [0, 0, 0],  # landing
    ]
)

accelerations = np.array(
    [
        [0, 0, None],  # beginning
        [None, None, None],  # first jump
        [None, None, None],  # second jump
        [None, None, None],  # getting into inverted flight
        [0, 0, None],  # inverted flight upside down
        [None, None, None],  # getting out of inverted flight
        [None, None, None],  # returning to upside up
        [0, None, None],  # slowing down
        [0, None, None],  # landing
    ]
)


states = {
    "x": points[:, 0],
    "y": points[:, 1],
    "z": points[:, 2],
    "vx": velocities[:, 0],
    "vy": velocities[:, 1],
    "vz": velocities[:, 2],
    "ax": accelerations[:, 0],
    "ay": accelerations[:, 1],
    "az": accelerations[:, 2],
    "gamma": np.deg2rad(gamma_points_deg),
    "gamma_dot": np.zeros(len(points[:, 0])),
    "gamma_ddot": np.zeros(len(points[:, 0])),
}

constraints = {
    "vx0": 0,
    "vy0": 0,
    "gamma_dot0": 0,
    "min_acc_x": min_acc_x,
    "min_acc_y": min_acc_y,
    "max_acc_y": max_acc_y,
    "max_acc_x": max_acc_x,
    "offset": 200,
    "angle_offset": np.deg2rad(45),
}


def traj_backflip():
    trajectory_params = get_traj_params(time_points, states, constraints)
    return trajectory_params


# trajectory_params = get_traj_params(time_points, states, constraints)

# Px_coefs, Py_coefs, Pgamma_coefs = calculate_trajectory(time_points, states, constraints)
# plot_trajectory(time_points, states, Px_coefs, Py_coefs, Pgamma_coefs)

# print(time_points)
