import numpy as np
import matplotlib.pyplot as plt

############################ Auxiliar functions ################################
def compute_rotation_angles(e1b, e2b, e3t):
    # Angles
    p_e1b = np.dot(e3t, e1b)
    p_e2b = np.dot(e3t, e2b)
    theta_e1b = p_e2b  # Rotation around e1b
    theta_e2b = -p_e1b  # Rotation around e2b

    return theta_e1b, theta_e2b


########################## Plotting the trajectory #############################
def plot_simulation(t, x, u, trajectory_params, controller_params, epos_list):
    fig_1, ((ax1_1, ax2_1, ax3_1), (ax4_1, ax5_1, ax6_1)) = plt.subplots(2, 3, figsize=(15, 12))
    t = np.array(t)
    x = np.array(x)
    u = np.array(u)

    t_ref = np.array(trajectory_params["t"])
    x_ref = np.array(trajectory_params["x"])
    y_ref = np.array(trajectory_params["y"])
    z_ref = np.array(trajectory_params["z"])
    e1bx_ref = np.array(trajectory_params["e1bx"])
    e1by_ref = np.array(trajectory_params["e1by"])
    e1bz_ref = np.array(trajectory_params["e1bz"])
    e2bx_ref = np.array(trajectory_params["e2bx"])
    e2by_ref = np.array(trajectory_params["e2by"])
    e2bz_ref = np.array(trajectory_params["e2bz"])
    e3bx_ref = np.array(trajectory_params["e3bx"])
    e3by_ref = np.array(trajectory_params["e3by"])
    e3bz_ref = np.array(trajectory_params["e3bz"])

    e1b = np.array([x[:, 6], x[:, 7], x[:, 8]])
    e2b = np.array([x[:, 9], x[:, 10], x[:, 11]])
    e3b = np.array([x[:, 12], x[:, 13], x[:, 14]])
    e3t = np.array([x[:, 21], x[:, 22], x[:, 23]])

    omega_control = np.array([u[1, :], u[2, :], np.zeros_like(u[2, :])])  # The TVC does not rotate around e3b
    omega_control = np.concatenate((omega_control, np.zeros((3, 1))), axis=1)

    delta_tvc_x = np.zeros_like(x[:, 0])
    delta_tvc_y = np.zeros_like(x[:, 0])

    for i in range(len(delta_tvc_x)):
        delta_tvc_x[i], delta_tvc_y[i] = compute_rotation_angles(e1b[:, i], e2b[:, i], e3t[:, i])  # rad

    fig_1.suptitle("Simulation results (position and velocity))")

    # plot 1: x
    ax1_1.plot(t, x[:, 0], label="x")
    ax1_1.set_xlabel("Time (s)")
    ax1_1.set_ylabel("X position (m)")
    ax1_1.set_title("X position vs Time")
    ax1_1.legend()
    ax1_1.grid()

    # plot 1: y
    ax2_1.plot(t, x[:, 2], label="y")
    ax2_1.set_xlabel("Time (s)")
    ax2_1.set_ylabel("Y position (m)")
    ax2_1.set_title("Y position vs Time")
    ax2_1.legend()
    ax2_1.grid()

    # plot 3: z
    ax3_1.plot(t, x[:, 4], label="z")
    ax3_1.set_xlabel("Time (s)")
    ax3_1.set_ylabel("Z position (m)")
    ax3_1.set_title("Z position vs Time")
    ax3_1.legend()
    ax3_1.grid()

    # plot 4: vx
    ax4_1.plot(t, x[:, 1], label="vx")
    ax4_1.set_xlabel("Time (s)")
    ax4_1.set_ylabel("X velocity (m/s)")
    ax4_1.set_title("X velocity vs Time")
    ax4_1.legend()
    ax4_1.grid()

    # plot 5: vy
    ax5_1.plot(t, x[:, 3], label="vy")
    ax5_1.set_xlabel("Time (s)")
    ax5_1.set_ylabel("Y velocity (m/s)")
    ax5_1.set_title("Y velocity vs Time")
    ax5_1.legend()
    ax5_1.grid()

    # plot 6: vz
    ax6_1.plot(t, x[:, 5], label="vz")
    ax6_1.set_xlabel("Time (s)")
    ax6_1.set_ylabel("Z velocity (m/s)")
    ax6_1.set_title("Z velocity vs Time")
    ax6_1.legend()
    ax6_1.grid()

    fig_2, ((ax1_2, ax2_2, ax3_2), (ax4_2, ax5_2, ax6_2)) = plt.subplots(2, 3, figsize=(15, 12))
    fig_2.suptitle("Simulation results (attitude and angular velocity))")

    # plot 4: e1
    ax1_2.plot(t, e1b[0, :])
    ax1_2.plot(t, e1b[1, :])
    ax1_2.plot(t, e1b[2, :])
    ax1_2.set_xlabel("Time (s)")
    ax1_2.set_ylabel("e1b")
    ax1_2.set_title("e1b vs Time")
    ax1_2.grid()

    # plot 5: e2
    ax2_2.plot(t, e2b[0, :])
    ax2_2.plot(t, e2b[1, :])
    ax2_2.plot(t, e2b[2, :])
    ax2_2.set_xlabel("Time (s)")
    ax2_2.set_ylabel("e2b")
    ax2_2.set_title("e2b vs Time")
    ax2_2.grid()

    # plot 6: e3
    ax3_2.plot(t, e3b[0, :])
    ax3_2.plot(t, e3b[1, :])
    ax3_2.plot(t, e3b[2, :])
    ax3_2.set_xlabel("Time (s)")
    ax3_2.set_ylabel("e3b")
    ax3_2.set_title("e3b vs Time")
    ax3_2.grid()

    ax4_2.plot(t, x[:, 24], label="$\\omega_x$")
    ax4_2.set_xlabel("Time (s)")
    ax4_2.set_ylabel("$\\omega_x$ angular velocity (rad/s)")
    ax4_2.set_title("$\\omega_x$ angular velocity vs Time")
    ax4_2.legend()
    ax4_2.grid()

    ax5_2.plot(t, x[:, 25], label="$\\omega_y$")
    ax5_2.set_xlabel("Time (s)")
    ax5_2.set_ylabel("$\\omega_y$ angular velocity (rad/s)")
    ax5_2.set_title("$\\omega_y$ angular velocity vs Time")
    ax5_2.legend()
    ax5_2.grid()

    ax6_2.plot(t, x[:, 26], label="$\\omega_z$")
    ax6_2.set_xlabel("Time (s)")
    ax6_2.set_ylabel("$\\omega_z$ angular velocity (rad/s)")
    ax6_2.set_title("$\\omega_z$ angular velocity vs Time")
    ax6_2.legend()
    ax6_2.grid()

    # tvc_angle = np.angle(x[:, 8] + 1j * x[:, 9]) - gamma

    fig_3, ((ax1_3, ax2_3, ax3_3), (ax4_3, ax5_3, ax6_3)) = plt.subplots(2, 3, figsize=(15, 12))
    fig_3.suptitle("Simulation results (control inputs))")

    thrust_bounds = controller_params["thrust_bounds"]
    delta_tvc_bounds = controller_params["delta_tvc_bounds"]
    delta_tvc_bounds = controller_params["delta_tvc_bounds"]

    thrust_dot_bounds = controller_params["thrust_dot_bounds"]
    delta_tvc_dot_bounds = controller_params["delta_tvc_dot_bounds"]

    # plot 1: thrust
    ax1_3.plot(t, x[:, 27])
    ax1_3.plot(t, [thrust_bounds[0]] * len(t), "--", color="black")
    ax1_3.plot(t, [thrust_bounds[1]] * len(t), "--", color="black")
    ax1_3.legend(["$f$", "$f_{min}$", "$f_{max}$"])
    ax1_3.set_xlabel("Time (s)")
    ax1_3.set_ylabel("Thrust (N)")
    ax1_3.set_title("Thrust vs Time")
    ax1_3.grid()

    # plot 2: delta_tvc_x
    ax2_3.plot(t, np.rad2deg(delta_tvc_x), label="$\\delta_{tvcx}$")
    ax2_3.plot(t, [np.rad2deg(delta_tvc_bounds[0])] * len(t), "--", color="black")
    ax2_3.plot(t, [np.rad2deg(delta_tvc_bounds[1])] * len(t), "--", color="black")
    ax2_3.legend()
    ax2_3.set_xlabel("Time (s)")
    ax2_3.set_ylabel("$\\delta_{tvc}$ (degrees)")
    ax2_3.set_title("$\\delta_{tvc}$ vs Time")
    ax2_3.legend()
    ax2_3.grid()

    # plot 3: delta_tvc_y
    ax3_3.plot(t, np.rad2deg(delta_tvc_y), label="$\\delta_{tvcy}$")
    ax3_3.plot(t, [np.rad2deg(delta_tvc_bounds[0])] * len(t), "--", color="black")
    ax3_3.plot(t, [np.rad2deg(delta_tvc_bounds[1])] * len(t), "--", color="black")
    ax3_3.legend()
    ax3_3.set_xlabel("Time (s)")
    ax3_3.set_ylabel("$\\delta_{tvc}$ (degrees)")
    ax3_3.set_title("$\\delta_{tvc}$ vs Time")
    ax3_3.legend()
    ax3_3.grid()

    # plot 4: thrust derivative
    ax4_3.plot(t[0:-1], u[0, :])
    ax4_3.plot(t, [thrust_dot_bounds[0]] * len(t), "--", color="black")
    ax4_3.plot(t, [thrust_dot_bounds[1]] * len(t), "--", color="black")
    ax4_3.legend(["$\dot{f}$", "$\dot{f}_{min}$", "$\dot{f}_{max}$"])
    ax4_3.set_xlabel("Time (s)")
    ax4_3.set_ylabel("Thrust derivative (N)")
    ax4_3.set_title("Thrust derivative vs Time")
    ax4_3.grid()

    # plot 5: delta_tvc_x derivative
    ax5_3.plot(t[0:-1], u[0, :], label=r"$\dot{f}$")
    ax5_3.plot(t, [thrust_dot_bounds[0]] * len(t), "--", color="black")
    ax5_3.plot(t, [thrust_dot_bounds[1]] * len(t), "--", color="black")
    ax5_3.legend()
    ax5_3.set_xlabel("Time (s)")
    ax5_3.set_ylabel("Thrust derivative (N)")
    ax5_3.set_title("Thrust derivative vs Time")
    ax5_3.grid()

    # plot 6: delta_tvc_y derivative
    ax6_3.plot(t, omega_control[0, :], label="$\\dot{\\delta}_{tvcx}$")
    ax6_3.plot(t, omega_control[1, :], label="$\\dot{\\delta}_{tvcy}$")
    ax6_3.plot(t, [delta_tvc_dot_bounds[0]] * len(t), "--", color="black")
    ax6_3.plot(t, [delta_tvc_dot_bounds[1]] * len(t), "--", color="black")
    ax6_3.legend()
    ax6_3.set_xlabel("Time (s)")
    ax6_3.set_ylabel("$\\dot{\\delta}_{tvcx}$ (rad/s)")
    ax6_3.set_title("$\\dot{\\delta}_{tvcx}$ vs Time")
    ax6_3.grid()

    fig_4, (ax1_4, ax2_4, ax3_4) = plt.subplots(1, 3, figsize=(15, 12))
    fig_4.suptitle("Simulation results (tracking error))")

    epos_list = np.array(epos_list)

    # plot 1: x error
    ax1_4.plot(t, epos_list[:, 0])
    ax1_4.set_xlabel("Time (s)")
    ax1_4.set_ylabel("X error (m)")
    ax1_4.set_title("X error vs Time")
    ax1_4.grid()

    # plot 2: y error
    ax2_4.plot(t, epos_list[:, 1])
    ax2_4.set_xlabel("Time (s)")
    ax2_4.set_ylabel("Y error (m)")
    ax2_4.set_title("Y error vs Time")
    ax2_4.grid()

    # plot 3: z error
    ax3_4.plot(t, epos_list[:, 2])
    ax3_4.set_xlabel("Time (s)")
    ax3_4.set_ylabel("Z error (m)")
    ax3_4.set_title("Z error vs Time")
    ax3_4.grid()

    # plot 3D trajectory
    fig_5 = plt.figure(figsize=(15, 10))
    ax = fig_5.add_subplot(111, projection="3d")
    ax.plot(x[:, 0], x[:, 2], x[:, 4])
    ax.plot(x_ref, y_ref, z_ref, label="trajectory")

    # Determine indices to sample every 5 seconds
    arrow_length = 10
    dt = t_ref[1] - t_ref[0]  # Assuming t is uniformly spaced
    sample_interval = int(3 / dt)
    sampled_indices = range(0, len(t), sample_interval)

    for i in sampled_indices:
        ax.quiver(
            x_ref[i],
            y_ref[i],
            z_ref[i],
            e1bx_ref[i],
            e1by_ref[i],
            e1bz_ref[i],
            color="r",
            label="e1b" if i == sampled_indices[0] else "",
            length=arrow_length,
        )
        ax.quiver(
            x_ref[i],
            y_ref[i],
            z_ref[i],
            e2bx_ref[i],
            e2by_ref[i],
            e2bz_ref[i],
            color="g",
            label="e2b" if i == sampled_indices[0] else "",
            length=arrow_length,
        )
        ax.quiver(
            x_ref[i],
            y_ref[i],
            z_ref[i],
            e3bx_ref[i],
            e3by_ref[i],
            e3bz_ref[i],
            color="b",
            label="e3b" if i == sampled_indices[0] else "",
            length=arrow_length,
        )

    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Y position (m)")
    ax.set_zlabel("Z position (m)")
    ax.set_title("3D trajectory")
    ax.set_box_aspect([1, 1, 1])
    ax.axis("equal")

    plt.tight_layout()
    plt.show()
