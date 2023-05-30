import numpy as np
import sys
sys.path.append('../')
sys.path.append('../../')

from Traj_planning.backflip.points_circle import *
from Traj_planning.traj_generators.diff_flat_traj_gen import *

# Example usage
initial_point = (50, 0, 0)
height = 50
radius = 50
total_time = 18
num_points = 500

############################## Data ###########################################
# environment variables
g = 9.81  # gravity

# constraints
max_ax = 15
min_ax = -15
max_ay = 15
min_ay = -15 

x, y, z = points_circle(initial_point, height, radius, total_time, num_points)

constraints = {
    "min_ax": min_ax,
    "max_ax": max_ax,
    "min_ay": min_ay,
    "max_ay": max_ay,
    "g": g,
}


(
    t,
    x,
    x_dot,
    x_dot_dot,
    y,
    y_dot,
    y_dot_dot,
    z,
    z_dot,
    z_dot_dot,
    e1bx,
    e1by,
    e2bx,
    e2by,
    gamma_dot,
    gamma_dot_dot,
) = diff_flat_traj(x, y, z, total_time, constraints=constraints, optimize=True)

target_points = np.array([x, y, z]).T
target_velocities = np.array([x_dot, y_dot, z_dot]).T
target_accelerations = np.array([x_dot_dot, y_dot_dot, z_dot_dot]).T

# controller input bounds
# max_ax = g * np.sin(np.deg2rad(10))  # maxium thrust rate
# min_ax = -g * np.sin(np.deg2rad(10))  # should be 30% to 40% of the max thrust

# max_ay = g  # maxium thrust rate
# min_ay = -3 * g  # should be 30% to 40% of the max thrust

states = {
    "t": t,  # time
    "x": target_points[:, 0],
    "y": target_points[:, 1],
    "z": target_points[:, 2],
    "vx": target_velocities[:, 0],
    "vy": target_velocities[:, 1],
    "vz": target_velocities[:, 2],
    "ax": target_accelerations[:, 0],
    "ay": target_accelerations[:, 1],
    "az": target_accelerations[:, 2],
    "e1bx": e1bx,
    "e1by": e1by,
    "e2bx": e2bx,
    "e2by": e2by,
    "gamma_dot": gamma_dot,
    "gamma_dot_dot": gamma_dot_dot,
}


def traj_points_df_circ(plot=False):
    trajectory_params = get_traj_params(states, constraints, plot=plot)
    return trajectory_params


def plot_traj():
    get_traj_params(states, constraints, plot=True)