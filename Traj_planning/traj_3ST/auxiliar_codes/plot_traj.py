import numpy as np
import matplotlib.pyplot as plt


def plot_trajectory(states, trajectory_params, controller_params, title="Trajectory"):
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
    gamma = trajectory_params["gamma"]
    gamma_dot = trajectory_params["gamma_dot"]
    gamma_dot_dot = trajectory_params["gamma_dot_dot"]
    
    f1 = trajectory_params["f1"]
    f2 = trajectory_params["f2"]
    f1_dot = trajectory_params["f1_dot"]
    f2_dot = trajectory_params["f2_dot"]
    f = trajectory_params["f"]
    f_dot = trajectory_params["f_dot"]
    delta_tvc = trajectory_params["delta_tvc"]
    delta_tvc_dot = trajectory_params["delta_tvc_dot"]

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
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Velocity (m/s)")
    axs[0, 1].set_title("Velocity vs Time")
    axs[0, 1].legend()
    axs[0, 1].grid()

    axs[1, 1].plot(t, ax, label="ax")
    axs[1, 1].plot(t, ay, label="ay")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Acceleration (m/s^2)")
    axs[1, 1].set_title("Acceleration vs Time")
    axs[1, 1].legend()
    axs[1, 1].grid()

    axs[2, 1].plot(t, e1bx, label="$e_{1bx}$")
    axs[2, 1].plot(t, e1by, label="$e_{1by}$")
    axs[2, 1].plot(t, e2bx, label="$e_{2bx}$")
    axs[2, 1].plot(t, e2by, label="$e_{2by}$")
    axs[2, 1].set_xlabel("Time (s)")
    axs[2, 1].set_ylabel("Frame parameters (-)")
    axs[2, 1].set_title("Frame parameters vs Time")
    axs[2, 1].legend()
    axs[2, 1].grid()
    
    
    ######################################
    # Plotting the estimated f1 and f2
    ######################################
    thrust_bounds = controller_params["thrust_bounds"]
    delta_tvc_bounds = controller_params["delta_tvc_bounds"]
    
    f1_bounds = np.cos(delta_tvc_bounds[1]) * np.array(thrust_bounds)
    f2_bounds = np.sin(delta_tvc_bounds[1]) * np.array([-thrust_bounds[1], thrust_bounds[1]])

    fig_2, (
        (ax1_2, ax2_2, ax3_2),
        (ax4_2, ax5_2, ax6_2),
        (ax7_2, ax8_2, ax9_2),
    ) = plt.subplots(3, 3, figsize=(15, 10))

    ax1_2.plot(t, f1, label="f1")
    ax1_2.plot(t, [f1_bounds[0]] * len(t), "--", color="black")
    ax1_2.plot(t, [f1_bounds[1]] * len(t), "--", color="black")
    ax1_2.grid()
    ax1_2.legend()
    ax1_2.set_title("Estimated f1")
    ax1_2.set_xlabel("t")
    ax1_2.set_ylabel("f1")

    ax2_2.plot(t, f2, label="f2")
    ax2_2.plot(t, np.sin(delta_tvc_bounds[1]) * f, "--", color="orange")
    ax2_2.plot(t, [f2_bounds[0]] * len(t), "--", color="black")
    ax2_2.plot(t, [f2_bounds[1]] * len(t), "--", color="black")
    ax2_2.plot(t, -np.sin(delta_tvc_bounds[1]) * f, "--", color="orange")
    ax2_2.grid()
    ax2_2.legend(["f2", "current", "max"])
    ax2_2.set_title("Estimated f2")
    ax2_2.set_xlabel("t")
    ax2_2.set_ylabel("f2")


    ax3_2.plot(t, f, label="f")
    ax3_2.plot(t, [thrust_bounds[0]] * len(t), "--", color="black")
    ax3_2.plot(t, [thrust_bounds[1]] * len(t), "--", color="black")
    ax3_2.grid()
    ax3_2.legend()
    ax3_2.set_title("Estimated f")
    ax3_2.set_xlabel("t")
    ax3_2.set_ylabel("f")

    ax4_2.plot(t, f1_dot, label="f1_dot")
    ax4_2.plot(t, [controller_params["thrust_dot_bounds"][0]] * len(t), "--", color="black")
    ax4_2.plot(t, [controller_params["thrust_dot_bounds"][1]] * len(t), "--", color="black")
    ax4_2.grid()
    ax4_2.legend()
    ax4_2.set_title("Estimated f1_dot")
    ax4_2.set_xlabel("t")
    ax4_2.set_ylabel("f1_dot")


    ax5_2.plot(t, f2_dot, label="f2_dot")
    ax5_2.plot(t, f * controller_params["delta_tvc_dot_bounds"][0], "--", color="orange")
    ax5_2.plot(t, f_dot * np.sin(delta_tvc) + f * np.cos(delta_tvc) * controller_params["delta_tvc_dot_bounds"][0], "--", color="black")
    ax5_2.plot(t, f * controller_params["delta_tvc_dot_bounds"][1], "--", color="orange")
    ax5_2.plot(t, f_dot * np.sin(delta_tvc) + f * np.cos(delta_tvc) * controller_params["delta_tvc_dot_bounds"][1], "--", color="black")
    ax5_2.grid()
    ax5_2.legend(["f2_dot", "current", "max"])
    ax5_2.set_title("Estimated f2_dot")
    ax5_2.set_xlabel("t")
    ax5_2.set_ylabel("f2_dot")


    ax6_2.plot(t, f_dot, label="f_dot")
    ax6_2.grid()
    ax6_2.legend()
    ax6_2.set_title("Estimated f_dot")
    ax6_2.set_xlabel("t")
    ax6_2.set_ylabel("f_dot")


    ax7_2.plot(
        t, np.rad2deg(delta_tvc), label="delta_tvc"
    )
    ax7_2.plot(t, [np.rad2deg(delta_tvc_bounds[0])] * len(t), "--", color="black")
    ax7_2.plot(t, [np.rad2deg(delta_tvc_bounds[1])] * len(t), "--", color="black")
    ax7_2.grid()
    ax7_2.legend()
    ax7_2.set_title("Estimated delta_tvc")
    ax7_2.set_xlabel("t")
    ax7_2.set_ylabel("delta_tvc (deg)")


    ax8_2.plot(
        t, delta_tvc_dot, label="delta_tvc_dot"
    )
    ax8_2.plot(t, [controller_params["delta_tvc_dot_bounds"][0]] * len(t), "--", color="black")
    ax8_2.plot(t, [controller_params["delta_tvc_dot_bounds"][1]] * len(t), "--", color="black")
    ax8_2.grid()
    ax8_2.legend()
    ax8_2.set_title("Estimated delta_tvc_dot")
    ax8_2.set_xlabel("t")
    ax8_2.set_ylabel("delta_tvc_dot (rad/s)")


    ax9_2.plot(t, np.rad2deg(gamma), label="gamma")
    ax9_2.plot(
    )
    ax9_2.grid()
    ax9_2.legend()
    ax9_2.set_title("Estimated gamma")
    ax9_2.set_xlabel("t")
    ax9_2.set_ylabel("gamma (deg)")
    
    plt.show()