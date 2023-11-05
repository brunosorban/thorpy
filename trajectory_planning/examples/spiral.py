import numpy as np


def generate_spiral(radius, height, num_turns, num_points):
    theta = np.linspace(0, 2 * np.pi * num_turns, num_points)  # Angle values
    z = np.linspace(0, height * num_turns, num_points)  # Height values

    # Generate x, y coordinates based on radius and angle
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Create a list of (x, y, z) points
    spiral_points = np.array([[x_val, y_val, z_val] for x_val, y_val, z_val in zip(x, y, z)])

    return spiral_points


radius = 50
height = 50  # between each spiral
num_turns = 3
num_points = 20

target_points = generate_spiral(radius, height, num_turns, num_points)

states = {
    "x": target_points[:, 0],
    "y": target_points[:, 1],
    "z": target_points[:, 2],
}

total_time = 120
