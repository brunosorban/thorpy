import numpy as np
from Traj_planning.safe_traj import safe_traj

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
target_accelerations = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])  # m/s^2
time_points = np.array([0, 30, 80, 120])  # time list
dt = 0.01  # time step

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


def hopper_traj():
    trajectory_params = safe_traj(
        target_points,
        target_velocities,
        target_accelerations,
        time_points,
        dt,
        plot=False,
    )
    return trajectory_params


def plot_traj():
    safe_traj(
        target_points,
        target_velocities,
        target_accelerations,
        time_points,
        dt,
        plot=True,
    )
