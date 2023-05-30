import numpy as np
from Traj_planning.backflip.points_circle import *
from Traj_planning.traj_generators.traj_planner_SE3 import *


# Example usage
initial_point = (50, 0, 0)
height = 50
radius = 50
total_time = 50
num_points = 50

x, y, z = points_circle(initial_point, height, radius, total_time, num_points)

############################## Data ###########################################
# environment variables
g = 9.81  # gravity

# rocket variables
m = 100  # mass of the hopper
# mf = 50  # final mass of the hopper
h = 2  # height of the hopper
radius = 0.25  # radius of the hopper
l_tvc = 0.5  # distance from the center of mass to the TVC

target_points = np.array([x, y, z]).T
target_velocities = np.array(
    [[None, None, None] for i in range(len(target_points[:, 0]))]
)  # m/s
target_velocities[0] = [0, 1e-6, 0]
target_accelerations = np.array(
    [[None, None, None] for i in range(len(target_points[:, 0]))]
)  # m/s^2
time_points = np.linspace(0, total_time, len(target_points[:, 0]))  # time list
dt = 0.01  # time step

# controller input bounds
max_ax = g * np.sin(np.deg2rad(10))  # maxium thrust rate
min_ax = -g * np.sin(np.deg2rad(10))  # should be 30% to 40% of the max thrust

max_ay = g  # maxium thrust rate
min_ay = -3 * g  # should be 30% to 40% of the max thrust

states = {
    "x": target_points[:, 0],
    "y": target_points[:, 1],
    "z": target_points[:, 2],
    "vx": target_velocities[:, 0],
    "vy": target_velocities[:, 1],
    "vz": target_velocities[:, 2],
    "ax": target_accelerations[:, 0],
    "ay": target_accelerations[:, 1],
    "az": target_accelerations[:, 2],
}

constraints = {
    "min_ax": min_ax,
    "max_ax": max_ax,
    "min_ay": min_ay,
    "max_ay": max_ay,
    "g": g,
}


def traj_circle():
    trajectory_params = get_traj_params(time_points, states, constraints)
    return trajectory_params


def plot_traj():
    get_traj_params(time_points, states, constraints, plot=True)