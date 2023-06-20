import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(states, trajectory_params, title="Trajectory"):
    t = trajectory_params["t"]
    x = trajectory_params["x"]
    vx = trajectory_params["vx"]
    ax = trajectory_params["ax"]
    y = trajectory_params["y"]
    vy = trajectory_params["vy"]
    ay = trajectory_params["ay"]
    z = trajectory_params["z"]
    vz = trajectory_params["vz"]
    az = trajectory_params["az"]
    e1bx = trajectory_params["e1bx"]
    e1by = trajectory_params["e1by"]
    e2bx = trajectory_params["e2bx"]
    e2by = trajectory_params["e2by"]
    gamma_dot = trajectory_params["gamma_dot"]
    gamma_dot_dot = trajectory_params["gamma_dot_dot"]

    gamma = np.arctan2(e1by, e1bx)
    last_t = t[0]

    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    axs[0, 0].plot(x, y, label="trajectory")
    axs[0, 0].scatter([-100, 100], [0, 0], s=1e-3)
    axs[0, 0].plot(states["x"], states["y"], "o", label="target points")
    axs[0, 0].set_xlabel("Horizontal Position (m)")
    axs[0, 0].set_ylabel("Vertical Position (m)")
    axs[0, 0].set_title("Trajectory")
    axs[0, 0].set_aspect("equal")
    axs[0, 0].legend()
    axs[0, 0].grid()

    # make the arrows
    for k in range(0, len(t) - 1):
        temp = t[t >= t[k]]

        for i in range(len(temp)):
            if (
                temp[i] - last_t >= 1 and temp[i] < t[k + 1]
            ):  # Plot frame every 10 seconds
                ind = np.searchsorted(t, temp[i])
                origin_x = x[ind]
                origin_y = y[ind]
                e1_x = e1bx[ind]
                e1_y = e1by[ind]
                e2_x = e2bx[ind]
                e2_y = e2by[ind]
                axs[0, 0].quiver(origin_x, origin_y, e1_x, e1_y, scale=10, color="red")
                axs[0, 0].quiver(origin_x, origin_y, e2_x, e2_y, scale=10, color="blue")
                last_t = temp[i]

    fig.suptitle(title, fontsize=16)

    axs[1, 0].plot(t, x, label="x")
    axs[1, 0].plot(t, y, label="y")
    # axs[1, 0].scatter(t, states["x"], "o", label="$x_{ref}$")
    # axs[1, 0].scatter(t, states["y"], "o", label="$y_{ref}$")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Position (m)")
    axs[1, 0].set_title("Position vs Time")
    axs[1, 0].legend()
    axs[1, 0].grid()

    axs[2, 0].plot(t, np.rad2deg(gamma), label="$\\gamma$")
    axs[2, 0].set_xlabel("Time (s)")
    axs[2, 0].set_ylabel("Angle (deg)")
    axs[2, 0].set_title("Angle vs Time")
    axs[2, 0].legend()
    axs[2, 0].grid()

    axs[0, 1].plot(t, vx, label="vx")
    axs[0, 1].plot(t, vy, label="vy")
    # axs[0, 1].plot(t, states["vx"], "o", label="$v_{x_{ref}}$")
    # axs[0, 1].plot(t, states["vy"], "o", label="$v_{y_{ref}}$")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Velocity (m/s)")
    axs[0, 1].set_title("Velocity vs Time")
    axs[0, 1].legend()
    axs[0, 1].grid()

    axs[1, 1].plot(t, ax, label="ax")
    axs[1, 1].plot(t, ay, label="ay")
    # axs[1, 1].plot(t, states["ax"], "o", label="$a_{x_{ref}}$")
    # axs[1, 1].plot(t, states["ay"], "o", label="$a_{y_{ref}}$")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Acceleration (m/s^2)")
    axs[1, 1].set_title("Acceleration vs Time")
    axs[1, 1].legend()
    axs[1, 1].grid()

    # axs[2, 1].plot(t, np.rad2deg(gamma_dot), label="$\\dot{\\gamma}$")
    axs[2, 1].plot(t, e1bx, label="$e_{1bx}$")
    axs[2, 1].plot(t, e1by, label="$e_{1by}$")
    axs[2, 1].plot(t, e2bx, label="$e_{2bx}$")
    axs[2, 1].plot(t, e2by, label="$e_{2by}$")
    axs[2, 1].set_xlabel("Time (s)")
    axs[2, 1].set_ylabel("Frame parameters (-)")
    axs[2, 1].set_title("Frame parameters vs Time")
    axs[2, 1].legend()
    axs[2, 1].grid()

    plt.show()
