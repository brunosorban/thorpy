import numpy as np
import matplotlib.pyplot as plt


def safe_traj(
    target_points, target_velocities, target_accelerations, t_list, dt, plot=False
):
    # initialize trajectory variables
    t = np.arange(t_list[0], t_list[-1], dt)
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    z = np.zeros(len(t))
    vx = np.zeros(len(t))
    vy = np.zeros(len(t))
    vz = np.zeros(len(t))
    ax = np.zeros(len(t))
    ay = np.zeros(len(t))
    az = np.zeros(len(t))

    # calculate trajectory minimum jerk
    t_index = 0
    for i in range(1, len(t_list)):
        t0 = t_list[i - 1]
        tf = t_list[i]
        A = np.array(
            [
                [t0**5, t0**4, t0**3, t0**2, t0, 1],
                [tf**5, tf**4, tf**3, tf**2, tf, 1],
                [5 * t0**4, 4 * t0**3, 3 * t0**2, 2 * t0, 1, 0],
                [5 * tf**4, 4 * tf**3, 3 * tf**2, 2 * tf, 1, 0],
                [20 * t0**3, 12 * t0**2, 6 * t0, 2, 0, 0],
                [20 * tf**3, 12 * tf**2, 6 * tf, 2, 0, 0],
            ]
        )

        bx = np.array(
            [
                target_points[i - 1, 0],
                target_points[i, 0],
                target_velocities[i - 1, 0],
                target_velocities[i, 0],
                target_accelerations[i - 1, 0],
                target_accelerations[i, 0],
            ]
        )
        by = np.array(
            [
                target_points[i - 1, 1],
                target_points[i, 1],
                target_velocities[i - 1, 1],
                target_velocities[i, 1],
                target_accelerations[i - 1, 1],
                target_accelerations[i, 1],
            ]
        )
        bz = np.array(
            [
                target_points[i - 1, 2],
                target_points[i, 2],
                target_velocities[i - 1, 2],
                target_velocities[i, 2],
                target_accelerations[i - 1, 2],
                target_accelerations[i, 2],
            ]
        )

        x_coeff = np.linalg.solve(A, bx)
        y_coeff = np.linalg.solve(A, by)
        z_coeff = np.linalg.solve(A, bz)

        while t_index < len(t) and t[t_index] < tf:
            x[t_index] = (
                x_coeff[0] * t[t_index] ** 5
                + x_coeff[1] * t[t_index] ** 4
                + x_coeff[2] * t[t_index] ** 3
                + x_coeff[3] * t[t_index] ** 2
                + x_coeff[4] * t[t_index]
                + x_coeff[5]
            )
            y[t_index] = (
                y_coeff[0] * t[t_index] ** 5
                + y_coeff[1] * t[t_index] ** 4
                + y_coeff[2] * t[t_index] ** 3
                + y_coeff[3] * t[t_index] ** 2
                + y_coeff[4] * t[t_index]
                + y_coeff[5]
            )
            z[t_index] = (
                z_coeff[0] * t[t_index] ** 5
                + z_coeff[1] * t[t_index] ** 4
                + z_coeff[2] * t[t_index] ** 3
                + z_coeff[3] * t[t_index] ** 2
                + z_coeff[4] * t[t_index]
                + z_coeff[5]
            )
            vx[t_index] = (
                5 * x_coeff[0] * t[t_index] ** 4
                + 4 * x_coeff[1] * t[t_index] ** 3
                + 3 * x_coeff[2] * t[t_index] ** 2
                + 2 * x_coeff[3] * t[t_index]
                + x_coeff[4]
            )
            vy[t_index] = (
                5 * y_coeff[0] * t[t_index] ** 4
                + 4 * y_coeff[1] * t[t_index] ** 3
                + 3 * y_coeff[2] * t[t_index] ** 2
                + 2 * y_coeff[3] * t[t_index]
                + y_coeff[4]
            )
            vz[t_index] = (
                5 * z_coeff[0] * t[t_index] ** 4
                + 4 * z_coeff[1] * t[t_index] ** 3
                + 3 * z_coeff[2] * t[t_index] ** 2
                + 2 * z_coeff[3] * t[t_index]
                + z_coeff[4]
            )
            ax[t_index] = (
                20 * x_coeff[0] * t[t_index] ** 3
                + 12 * x_coeff[1] * t[t_index] ** 2
                + 6 * x_coeff[2] * t[t_index]
                + 2 * x_coeff[3]
            )
            ay[t_index] = (
                20 * y_coeff[0] * t[t_index] ** 3
                + 12 * y_coeff[1] * t[t_index] ** 2
                + 6 * y_coeff[2] * t[t_index]
                + 2 * y_coeff[3]
            )
            az[t_index] = (
                20 * z_coeff[0] * t[t_index] ** 3
                + 12 * z_coeff[1] * t[t_index] ** 2
                + 6 * z_coeff[2] * t[t_index]
                + 2 * z_coeff[3]
            )
            t_index += 1

    if plot:
        # Plot the trajectory and target points in 3D
        fig = plt.figure()
        axes = fig.add_subplot(111, projection="3d")
        axes.plot(x, y, z, label="Trajectory")
        axes.scatter(
            target_points[:, 0],
            target_points[:, 1],
            target_points[:, 2],
            label="Target Points",
        )
        axes.legend()
        axes.set_xlabel("X")
        axes.set_ylabel("Y")
        axes.set_zlabel("Z")
        plt.show()

        # plot x, y, and z vs time
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))

        # Plot the position components
        ax1.plot(t, x, label="X")
        ax1.plot(t, y, label="Y")
        ax1.plot(t, z, label="Z")
        ax1.set_ylabel("Position")
        ax1.legend()

        # Plot the velocity components
        ax2.plot(t, vx, label="Vx")
        ax2.plot(t, vy, label="Vy")
        ax2.plot(t, vz, label="Vz")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Velocity (m/s)")
        ax2.legend()

        # Plot acceleration components
        ax3.plot(t, ax, label="Ax")
        ax3.plot(t, ay, label="Ay")
        ax3.plot(t, az, label="Az")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Acceleration (m/s^2)")
        ax3.legend()

        # Add gridlines to both subplots
        ax1.grid()
        ax2.grid()
        ax3.grid()

        fig.show()

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
        "gamma": np.zeros(len(t)) + np.deg2rad(90),
        "gamma_dot": np.zeros(len(t)),
        "gamma_ddot": np.zeros(len(t)),
    }

    return trajectory
