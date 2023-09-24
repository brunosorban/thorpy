import sys
import numpy as np
from Traj_planning.traj_generator_3ST import *

sys.path.append("../")
sys.path.append("../Traj_planning")

from parameters import *
from Traj_planning.examples.simple_circ import *


def generate_spiral(radius, height, num_turns, num_points):
    theta = np.linspace(0, 2 * np.pi * num_turns, num_points)  # Angle values
    z = np.linspace(0, height * num_turns, num_points)  # Height values

    # Generate x, y coordinates based on radius and angle
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Create a list of (x, y, z) points
    spiral_points = np.array(
        [[x_val, y_val, z_val] for x_val, y_val, z_val in zip(x, y, z)]
    )

    return spiral_points


radius = 50
height = 50  # between each spiral
num_turns = 3
num_points = 20

target_points = generate_spiral(radius, height, num_turns, num_points)

time_points = 4.3 * np.arange(len(target_points))  # time list

states = {
    "t": time_points,
    "x": target_points[:, 0],
    "y": target_points[:, 1],
    "z": target_points[:, 2],
}


def traj_spiral_3ST():
    traj_params = trajenerator_3ST(states, env_params, rocket_params, controller_params)
    return traj_params
