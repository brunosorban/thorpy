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
    e1bz = trajectory_params["e1bz"]
    e2bx = trajectory_params["e2bx"]
    e2by = trajectory_params["e2by"]
    e2bz = trajectory_params["e2bz"]
    e3bx = trajectory_params["e3bx"]
    e3by = trajectory_params["e3by"]
    e3bz = trajectory_params["e3bz"]
    omega = trajectory_params["omega"]
    omega_dot = trajectory_params["omega_dot"]
    omega_dot_2 = trajectory_params["omega_dot_2"]

    f1 = trajectory_params["f1"]
    f2 = trajectory_params["f2"]
    f3 = trajectory_params["f3"]
    f1_dot = trajectory_params["f1_dot"]
    f2_dot = trajectory_params["f2_dot"]
    f3_dot = trajectory_params["f3_dot"]
    f = trajectory_params["f"]
    f_dot = trajectory_params["f_dot"]
    # delta_tvc = trajectory_params["delta_tvc"]
    # delta_tvc_dot = trajectory_params["delta_tvc_dot"]
    f1_2 = trajectory_params["f1_2"]
    f2_2 = trajectory_params["f2_2"]

    last_t = t[0]

    fig, axs = plt.subplots(3, 2, figsize=(15, 15))

    # axs[0, 0].plot(x, z, label="trajectory")
    # # axs[0, 0].scatter([-100, 100], [0, 0], s=1e-3)
    # axs[0, 0].plot(states["x"], states["z"], "o", markersize=1, label="target points")
    # axs[0, 0].set_xlabel("X Position (m)")
    # axs[0, 0].set_ylabel("Z Position (m)")
    # axs[0, 0].set_title("Trajectory")
    # axs[0, 0].set_aspect("equal")
    # axs[0, 0].legend()
    # axs[0, 0].grid()

    # # make the arrows
    # for k in range(0, len(t) - 1):
    #     temp = t[t >= t[k]]
    #     print("e1b: {:.2f} {:.2f}".format(e1bx[k], e1bz[k]))
    #     print("e3b: {:.2f} {:.2f}".format(e3bx[k], e3bz[k]))

    #     for i in range(len(temp)):
    #         if (
    #             temp[i] - last_t >= 1 and temp[i] < t[k + 1]
    #         ):  # Plot frame every 10 seconds
    #             ind = np.searchsorted(t, temp[i])
    #             origin_x = x[ind]
    #             origin_z = z[ind]
    #             e1_x = e1bx[ind]
    #             e1_z = e1bz[ind]
    #             e3_x = e3bx[ind]
    #             e3_z = e3bz[ind]
    #             axs[0, 0].quiver(origin_x, origin_z, e1_x, e1_z, scale=10, color="red")
    #             axs[0, 0].quiver(origin_x, origin_z, e3_x, e3_z, scale=10, color="blue")
    #             last_t = temp[i]

    fig.suptitle(title, fontsize=16)

    axs[0, 0].plot(t, x, label="x")
    axs[0, 0].plot(t, y, label="y")
    axs[0, 0].plot(t, z, label="z")
    axs[0, 0].set_xlabel("Time (s)")
    axs[0, 0].set_ylabel("Position (m)")
    axs[0, 0].set_title("Position vs Time")
    axs[0, 0].legend()
    axs[0, 0].grid()

    axs[1, 0].plot(t, vx, label="vx")
    axs[1, 0].plot(t, vy, label="vy")
    axs[1, 0].plot(t, vz, label="vz")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Velocity (m/s)")
    axs[1, 0].set_title("Velocity vs Time")
    axs[1, 0].legend()
    axs[1, 0].grid()

    axs[2, 0].plot(t, ax, label="ax")
    axs[2, 0].plot(t, ay, label="ay")
    axs[2, 0].plot(t, az, label="az")
    axs[2, 0].set_xlabel("Time (s)")
    axs[2, 0].set_ylabel("Acceleration (m/s^2)")
    axs[2, 0].set_title("Acceleration vs Time")
    axs[2, 0].legend()
    axs[2, 0].grid()

    axs[0, 1].plot(t, e1bx, label="$e_{1bx}$")
    axs[0, 1].plot(t, e1by, label="$e_{1by}$")
    axs[0, 1].plot(t, e1bz, label="$e_{1bz}$")
    axs[0, 1].plot(t, np.ones_like(t), "--", color="black")
    axs[0, 1].plot(t, -np.ones_like(t), "--", color="black")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Frame parameters (-)")
    axs[0, 1].set_title("Frame parameters vs Time")
    axs[0, 1].legend()
    axs[0, 1].grid()

    axs[1, 1].plot(t, e2bx, label="$e_{2bx}$")
    axs[1, 1].plot(t, e2by, label="$e_{2by}$")
    axs[1, 1].plot(t, e2bz, label="$e_{2bz}$")
    axs[1, 1].plot(t, np.ones_like(t), "--", color="black")
    axs[1, 1].plot(t, -np.ones_like(t), "--", color="black")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Frame parameters (-)")
    axs[1, 1].set_title("Frame parameters vs Time")
    axs[1, 1].legend()
    axs[1, 1].grid()

    axs[2, 1].plot(t, e3bx, label="$e_{3bx}$")
    axs[2, 1].plot(t, e3by, label="$e_{3by}$")
    axs[2, 1].plot(t, e3bz, label="$e_{3bz}$")
    axs[2, 1].plot(t, np.ones_like(t), "--", color="black")
    axs[2, 1].plot(t, -np.ones_like(t), "--", color="black")
    axs[2, 1].set_xlabel("Time (s)")
    axs[2, 1].set_ylabel("Frame parameters (-)")
    axs[2, 1].set_title("Frame parameters vs Time")
    axs[2, 1].legend()
    axs[2, 1].grid()

    ######################################
    # Plotting the estimated f1 and f2
    ######################################
    thrust_bounds = controller_params["thrust_bounds"]
    delta_tvc_y_bounds = controller_params["delta_tvc_bounds"]
    delta_tvc_z_bounds = controller_params["delta_tvc_bounds"]

    f1_bounds = np.cos(delta_tvc_y_bounds[1]) * np.array(thrust_bounds)
    f2_bounds = np.sin(delta_tvc_y_bounds[1]) * np.array(
        [-thrust_bounds[1], thrust_bounds[1]]
    )

    fig_2, ((ax1_2, ax2_2, ax3_2), (ax4_2, ax5_2, ax6_2)) = plt.subplots(
        2, 3, figsize=(15, 10)
    )
    ax1_2.plot(t, omega[0, :], label="$\omega_x$")
    ax1_2.set_xlabel("Time (s)")
    ax1_2.set_ylabel("Angular velocity in x (rad/s)")
    ax1_2.set_title("Angular velocity vs Time")
    ax1_2.legend()
    ax1_2.grid()

    ax2_2.plot(t, omega[1, :], label="$\omega_y$")
    ax2_2.set_xlabel("Time (s)")
    ax2_2.set_ylabel("Angular velocity in y (rad/s)")
    ax2_2.set_title("Angular velocity vs Time")
    ax2_2.legend()
    ax2_2.grid()

    ax3_2.plot(t, omega[2, :], label="$\omega_z$")
    ax3_2.set_xlabel("Time (s)")
    ax3_2.set_ylabel("Angular velocity in z (rad/s)")
    ax3_2.set_title("Angular velocity vs Time")
    ax3_2.legend()
    ax3_2.grid()

    ax4_2.plot(t, omega_dot[0, :], label="$\omega_{dot_x}$")
    ax4_2.set_xlabel("Time (s)")
    ax4_2.set_ylabel("Angular acceleration in x (rad/s^2)")
    ax4_2.set_title("Angular acceleration vs Time")
    ax4_2.legend()
    ax4_2.grid()

    ax5_2.plot(t, omega_dot[1, :], label="$\omega_{dot_y}$")
    ax5_2.set_xlabel("Time (s)")
    ax5_2.set_ylabel("Angular acceleration in y (rad/s^2)")
    ax5_2.set_title("Angular acceleration vs Time")
    ax5_2.legend()
    ax5_2.grid()

    ax6_2.plot(t, omega_dot[2, :], label="$\omega_{dot_z}$")
    ax6_2.set_xlabel("Time (s)")
    ax6_2.set_ylabel("Angular acceleration in z (rad/s^2)")
    ax6_2.set_title("Angular acceleration vs Time")
    ax6_2.legend()
    ax6_2.grid()

    fig_2, (
        (ax1_2, ax2_2, ax3_2),
        (ax4_2, ax5_2, ax6_2),
    ) = plt.subplots(2, 3, figsize=(15, 10))

    ax1_2.plot(t, f1, "o", markersize=1, label="f1")
    ax1_2.plot(t, f1_2, "o", markersize=1, label="f1_2")
    ax1_2.plot(t, [f1_bounds[0]] * len(t), "--", color="black")
    ax1_2.plot(t, [f1_bounds[1]] * len(t), "--", color="black")
    ax1_2.grid()
    ax1_2.legend()
    ax1_2.set_title("Estimated f1")
    ax1_2.set_xlabel("t")
    ax1_2.set_ylabel("f1")

    ax2_2.plot(t, np.sqrt(f2**2 + f3**2), "o", markersize=1, label="f2")
    # ax2_2.plot(t, f3, "o", markersize=1, label="f3")
    ax2_2.plot(t, f2_2, "o", markersize=1, label="f2_2")
    ax2_2.plot(
        t, np.sin(delta_tvc_y_bounds[1]) * f, "--", color="orange", label="constraint"
    )
    ax2_2.plot(t, [f2_bounds[0]] * len(t), "--", color="black", label="max")
    ax2_2.plot(t, [f2_bounds[1]] * len(t), "--", color="black")
    ax2_2.plot(t, -np.sin(delta_tvc_y_bounds[1]) * f, "--", color="orange")
    ax2_2.grid()
    ax2_2.legend()
    ax2_2.set_title("Estimated f2")
    ax2_2.set_xlabel("t")
    ax2_2.set_ylabel("f2")

    ax3_2.plot(t, f, "o", markersize=1, label="f")
    ax3_2.plot(t, [thrust_bounds[0]] * len(t), "--", color="black")
    ax3_2.plot(t, [thrust_bounds[1]] * len(t), "--", color="black")
    ax3_2.grid()
    ax3_2.legend()
    ax3_2.set_title("Estimated f")
    ax3_2.set_xlabel("t")
    ax3_2.set_ylabel("f")

    ax4_2.plot(t, f1_dot, "o", markersize=1, label="f1_dot")
    ax4_2.plot(
        t, [controller_params["thrust_dot_bounds"][0]] * len(t), "--", color="black"
    )
    ax4_2.plot(
        t, [controller_params["thrust_dot_bounds"][1]] * len(t), "--", color="black"
    )
    ax4_2.grid()
    ax4_2.legend()
    ax4_2.set_title("Estimated f1_dot")
    ax4_2.set_xlabel("t")
    ax4_2.set_ylabel("f1_dot")

    ax5_2.plot(t, f2_dot, "o", markersize=1, label="f2_dot")
    ax5_2.plot(t, f3_dot, "o", markersize=1, label="f3_dot")
    # ax5_2.plot(
    #     t, f * controller_params["delta_tvc_dot_bounds"][0], "--", color="orange", label="constraint"
    # )
    # ax5_2.plot(
    #     t,
    #     f_dot * np.sin(delta_tvc_y_bounds)
    #     + f * np.cos(delta_tvc_y_bounds) * controller_params["delta_tvc_dot_bounds"][0],
    #     "--",
    #     color="black",
    # )
    # ax5_2.plot(
    #     t, f * controller_params["delta_tvc_dot_bounds"][1], "--", color="orange"
    # )
    # ax5_2.plot(
    #     t,
    #     f_dot * np.sin(delta_tvc_y_bounds)
    #     + f * np.cos(delta_tvc_y_bounds) * controller_params["delta_tvc_dot_bounds"][1],
    #     "--",
    #     color="black",
    # )
    ax5_2.grid()
    ax5_2.legend()
    ax5_2.set_title("Estimated f2_dot")
    ax5_2.set_xlabel("t")
    ax5_2.set_ylabel("f2_dot")

    ax6_2.plot(t, f_dot, "o", markersize=1, label="f_dot")
    ax6_2.grid()
    ax6_2.legend()
    ax6_2.set_title("Estimated f_dot")
    ax6_2.set_xlabel("t")
    ax6_2.set_ylabel("f_dot")

    fig_3 = plt.figure(figsize=(15, 15))
    axs_3 = fig_3.add_subplot(111, projection="3d")
    axs_3.plot(x, y, z, label="trajectory")
    axs_3.plot(
        states["x"], states["y"], states["z"], "o", markersize=5, label="target points"
    )
    

    # Determine indices to sample every 5 seconds
    dt = t[1] - t[0]  # Assuming t is uniformly spaced
    sample_interval = int(3 / dt)
    sampled_indices = range(0, len(t), sample_interval)
    arrow_length = 10
    
    # Plot e1b, e2b, e3b vectors
    for i in sampled_indices:
        axs_3.quiver(x[i], y[i], z[i],
                  e1bx[i], e1by[i], e1bz[i], color='r', label='e1b' if i == sampled_indices[0] else "", length=arrow_length)
        axs_3.quiver(x[i], y[i], z[i],
                  e2bx[i], e2by[i], e2bz[i], color='g', label='e2b' if i == sampled_indices[0] else "", length=arrow_length)
        axs_3.quiver(x[i], y[i], z[i],
                  e3bx[i], e3by[i], e3bz[i], color='b', label='e3b' if i == sampled_indices[0] else "", length=arrow_length)
        
    axs_3.set_xlabel("X Position (m)")
    axs_3.set_ylabel("Y Position (m)")
    axs_3.set_zlabel("Z Position (m)")
    axs_3.set_title("Trajectory")
    axs_3.set_aspect("equal")
    axs_3.legend()
    axs_3.grid()