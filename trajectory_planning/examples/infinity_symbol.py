import numpy as np

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

states = {
    "x": target_points[:, 0],
    "y": target_points[:, 1],
    "z": target_points[:, 2],
}

total_time = 36