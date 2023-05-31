import numpy as np
import matplotlib.pyplot as plt
from Traj_planning.traj_generators.traj_planner_SE3 import *


def curve(r, theta):
    return r * np.cos(theta), r * np.sin(theta)


def vel_direction(r, theta):
    return np.array([-np.sin(theta), np.cos(theta)])


def calculate_centripetal_acceleration(v, r, theta):
    # Calculate centripetal acceleration
    x, y = curve(r, theta)
    radial = -np.array([x, y]) / np.linalg.norm(np.array([x, y]))
    a_cp = (v**2) / r * radial

    return a_cp


def calculate_force(v, r, theta, m, g=9.8):
    # Calculate centripetal acceleration in the world frame
    a_cp = calculate_centripetal_acceleration(v, r, theta)
    mg = np.zeros_like(a_cp)
    mg[1] = m * g

    # Calculate the force in the world frame
    f = m * a_cp + mg

    return f


def calculate_traj_params(v, r, m, g=9.8):
    """
    Given the velocity, radius, and mass of the rocket, calculate the
    trajectory parameters. The rocket shall go h=2r meters above the ground, reaching
    the point (0, h) with velocity v. It shall do a circular motion of radius r
    from this point starting at 90 degrees and ending at 270 degrees.
    """

    x = []
    y = []
    vx = []
    vy = []
    t = []
    e1bx = []
    e1by = []

    theta = np.linspace(0, 2 * np.pi, 100)
    d_theta = theta[1] - theta[0]
    d_arc = r * d_theta
    dt = d_arc / v
    h = 2 * r

    x.append(2 * r)
    y.append(0)
    vx.append(0)
    vy.append(0.8 * v)
    t.append(0)
    e1bx.append(0)
    e1by.append(1)

    x.append(2 * r)
    y.append(h)
    vx.append(0)
    vy.append(v)
    t.append(h / v)
    e1bx.append(0)
    e1by.append(1)

    for i in range(len(theta)):
        x.append(curve(r, theta[i])[0] + r)
        y.append(curve(r, theta[i])[1] + h)

        vx.append(v * vel_direction(r, theta[i])[0])
        vy.append(v * vel_direction(r, theta[i])[1])

        t.append(t[-1] + dt)

        f = calculate_force(v, r, theta[i], m, g)
        e1bx.append(f[0] / np.linalg.norm(f))
        e1by.append(f[1] / np.linalg.norm(f))

    x = np.array(x)
    y = np.array(y)
    vx = np.array(vx)
    vy = np.array(vy)
    t = np.array(t)

    return x, y, vx, vy, t, e1bx, e1by


def plot_vel_vectors(x, y, vx, vy):
    plt.figure()
    plt.plot(x, y, label="Trajectory")
    plt.quiver(x, y, vx, vy, color="red", label="Velocity Vector", scale=2000)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Velocity Vector over Trajectory")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def get_traj_params(t, x, y, z, vx, vy, vz, e1bx, e1by):
    # initialize variables
    n = len(t)
    ax = np.zeros(n)
    ay = np.zeros(n)
    az = np.zeros(n)
    e2bx = np.zeros(n)
    e2by = np.zeros(n)
    gamma_dot = np.zeros(n)

    gamma = np.arctan2(e1by, e1bx)

    for i in range(n):
        if i == 0:
            ax[i] = (vx[i + 1] - vx[i]) / (t[i + 1] - t[i])
            ay[i] = (vy[i + 1] - vy[i]) / (t[i + 1] - t[i])
            az[i] = (vz[i + 1] - vz[i]) / (t[i + 1] - t[i])
            gamma_dot[i] = (gamma[i + 1] - gamma[i]) / (t[i + 1] - t[i])
        elif i == n - 1:
            ax[i] = (vx[i] - vx[i - 1]) / (t[i] - t[i - 1])
            ay[i] = (vy[i] - vy[i - 1]) / (t[i] - t[i - 1])
            az[i] = (vz[i] - vz[i - 1]) / (t[i] - t[i - 1])
            gamma_dot[i] = (gamma[i] - gamma[i - 1]) / (t[i] - t[i - 1])
        else:
            ax[i] = (vx[i + 1] - 2 * vx[i] + vx[i - 1]) / (
                (t[i + 1] - t[i]) * (t[i] - t[i - 1])
            )
            ay[i] = (vy[i + 1] - 2 * vy[i] + vy[i - 1]) / (
                (t[i + 1] - t[i]) * (t[i] - t[i - 1])
            )
            az[i] = (vz[i + 1] - 2 * vz[i] + vz[i - 1]) / (
                (t[i + 1] - t[i]) * (t[i] - t[i - 1])
            )
            gamma_dot[i] = (gamma[i + 1] - 2 * gamma[i] + gamma[i - 1]) / (
                (t[i + 1] - t[i]) * (t[i] - t[i - 1])
            )

        [e2bx[i], e2by[i], _] = np.cross(
            np.array([0, 0, 1.0]), np.array([e1bx[i], e1by[i], 0])
        )

    trajectory = {
        "t": t,
        "x": x,
        "y": y,
        "z": z,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "ax": ax,
        "ay": ay,
        "az": az,
        "e1bx": e1bx,
        "e1by": e1by,
        "e2bx": e2bx,
        "e2by": e2by,
        "gamma": gamma,
        "gamma_dot": gamma_dot,
    }

    return trajectory


# Example usage
v = 5  # m/s
r = 50  # meters
m = 100  # kg

# plot_vel_vectors(*calculate_traj_params(v, r, m)[0:4])


def traj_simple_circle():
    x, y, vx, vy, t, e1bx, e1by = calculate_traj_params(v, r, m)
    z = np.zeros_like(t)
    vz = np.zeros_like(t)
    trajectory_params = get_traj_params(t, x, y, z, vx, vy, vz, e1bx, e1by)
    return trajectory_params


# def plot_traj():
#     get_traj_params(time_points, states, constraints, plot=True)


############################## Data ###########################################
# # environment variables
# g = 9.81  # gravity

# # rocket variables
# m = 100  # mass of the hopper
# # mf = 50  # final mass of the hopper
# h = 2  # height of the hopper
# radius = 0.25  # radius of the hopper
# l_tvc = 0.5  # distance from the center of mass to the TVC

# target_points = np.array([x, y, z]).T
# target_velocities = np.array([vx, vy, vz]).T
# # target_velocities[0] += np.array([0, 1e-6, 0])
# target_accelerations = np.array([[None, None, None] for i in range(len(target_points[:, 0]))])  # m/s^2
# target_accelerations[0] = np.array([0, 0, 0])
# # target_accelerations[1] = np.array([0, 0, 0])

# # # controller input bounds
# # max_acc_x = g * np.sin(np.deg2rad(10))  # maxium thrust rate
# # min_acc_x = -g * np.sin(np.deg2rad(10))  # should be 30% to 40% of the max thrust

# # max_acc_y = g  # maxium thrust rate
# # min_acc_y = -3 * g  # should be 30% to 40% of the max thrust

# states = {
#     "x": target_points[:, 0],
#     "y": target_points[:, 1],
#     "z": target_points[:, 2],
#     "vx": target_velocities[:, 0],
#     "vy": target_velocities[:, 1],
#     "vz": target_velocities[:, 2],
#     "ax": target_accelerations[:, 0],
#     "ay": target_accelerations[:, 1],
#     "az": target_accelerations[:, 2],
# }

# constraints = {
#     # "min_acc_x": min_acc_x,
#     # "max_acc_x": max_acc_x,
#     # "min_acc_y": min_acc_y,
#     # "max_acc_y": max_acc_y,
#     "g": g,
# }


# def traj_simple_circle():
#     trajectory_params = get_traj_params(time_points, states, constraints)
#     return trajectory_params


# def plot_traj():
#     get_traj_params(time_points, states, constraints, plot=True)

# print("x: ", states["x"])
# print("y: ", states["y"])
# print("vx: ", states["vx"])
# print("vy: ", states["vy"])
# print("t: ", time_points)
