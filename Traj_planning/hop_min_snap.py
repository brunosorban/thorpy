import numpy as np
from Traj_planning.optimal_traj_planner import *

############################## Data ###########################################
# environment variables
g = 9.81  # gravity

# rocket variables
m = 100  # mass of the hopper
# mf = 50  # final mass of the hopper
h = 2  # height of the hopper
radius = 0.25  # radius of the hopper
l_tvc = 0.5  # distance from the center of mass to the TVC

# define trajectory parameters (x, y, z)
target_points = np.array(
    [
        [0, 0, 0],
        [0, 500, 0],
        [1000, 500, 0],
        [1000, 0, 0],
    ]
)

gamma_points_deg = np.array([90, 90, 90, 90])
target_velocities = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])  # m/s
target_accelerations = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]) # m/s^2
time_points = np.array([0, 30, 80, 120]) # time list
dt = 0.01 # time step

# controller input bounds
max_acc_y = 2 * g  # maxium thrust rate
min_acc_y = 0.35 * max_acc_y  - g # should be 30% to 40% of the max thrust
max_acc_x = max_acc_y * np.sin(np.deg2rad(30))
min_acc_x = -max_acc_x

states = {
    "x": target_points[:, 0],
    "y": target_points[:, 1],
    "z": target_points[:, 2],
    "vx": np.zeros(len(target_points[:, 0])),
    "vy": np.zeros(len(target_points[:, 0])),
    "vz": np.zeros(len(target_points[:, 0])),
    "ax": np.zeros(len(target_points[:, 0])),
    "ay": np.zeros(len(target_points[:, 0])),
    "az": np.zeros(len(target_points[:, 0])),
    "gamma": np.deg2rad(gamma_points_deg),
    "gamma_dot": np.zeros(len(target_points[:, 0])),
    "gamma_ddot": np.zeros(len(target_points[:, 0])),
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
    "angle_offset": np.deg2rad(30)
}

def hop_min_snap():
    trajectory_params = get_traj_params(time_points, states, constraints)
    return trajectory_params

def plot_traj():
    get_traj_params(time_points, states, constraints, plot=True)
    
print(gamma_points_deg)