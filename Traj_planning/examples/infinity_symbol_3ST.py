import sys
import numpy as np
from Traj_planning.traj_generator_3ST import *

sys.path.append("../")
sys.path.append("../Traj_planning")

from parameters import *
from Traj_planning.examples.simple_circ import *

def generate_infinity_symbol(length, height_variation, num_points):
    theta = np.linspace(-np.pi, np.pi, num_points, endpoint=True)  # Angle values
    x = length * np.sin(theta)
    y = length * np.sin(theta) * np.cos(theta)
    z = height_variation * (1 + np.cos(theta))
    
    infinity_symbol_points = np.array([[x_val, y_val, z_val] for x_val, y_val, z_val in zip(x, y, z)])
    
    return infinity_symbol_points

# Parameters
length = 50
height_variation = 20
num_points = 9

target_points = generate_infinity_symbol(length, height_variation, num_points)

time_points = 3 * np.arange(len(target_points))  # time list

states = {
    "t": time_points,
    "x": target_points[:, 0],
    "y": target_points[:, 1],
    "z": target_points[:, 2],
}


def traj_infinity_symbol_3ST():
    traj_params = trajenerator_3ST(states, env_params, rocket_params, controller_params)
    return traj_params

# import matplotlib.pyplot as plt

# # plot 3D trajectory
# fig_5 = plt.figure(figsize=(15, 10))
# ax = fig_5.add_subplot(111, projection='3d')
# ax.plot(x, y, z, label="trajectory")
# ax.set_xlabel("X position (m)")
# ax.set_ylabel("Y position (m)")
# ax.set_zlabel("Z position (m)")
# ax.set_title("3D trajectory")
# ax.set_box_aspect([1, 1, 1])
# ax.axis('equal')

# plt.tight_layout()
# plt.show()