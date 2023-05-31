import numpy as np
import matplotlib.pyplot as plt

import numpy as np


def points_circle(initial_point, height, radius, total_time, num_points):
    num_ascent_points = 3  # Number of points for ascent
    num_circular_points = num_points - 3  # Number of points for circular motion

    t_ascent = np.linspace(
        0, total_time / 6, num_ascent_points
    )  # Time points for ascent
    t_circular = np.linspace(
        total_time / 6, total_time, num_circular_points, endpoint=True
    )  # Time points for circular motion

    # Calculate x, y, and z coordinates for ascent
    x_ascent = np.full_like(t_ascent, initial_point[0]) + radius
    y_ascent = initial_point[1] + (height + radius) * t_ascent / (total_time / 6)
    z_ascent = np.zeros_like(y_ascent)

    # Calculate x, y, and z coordinates for circular motion
    theta = 2 * np.pi * (t_circular - total_time / 6) / (5 * total_time / 6)
    x_circular = x_ascent[-1] + radius * np.cos(theta) - radius
    y_circular = y_ascent[-1] + radius * np.sin(theta)
    z_circular = np.zeros_like(x_circular)

    # Concatenate the coordinates
    x = np.append(x_ascent, x_circular)
    y = np.append(y_ascent, y_circular)
    z = np.append(z_ascent, z_circular)

    return x, y, z


# # Example usage
# initial_point = (0, 0)
# height = 10
# radius = 5
# total_time = 10
# num_points = 100

# x, y, z = generate_trajectory(initial_point, height, radius, total_time, num_points)

# # Plotting
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x, y, z)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_aspect('equal')

# plt.show()
