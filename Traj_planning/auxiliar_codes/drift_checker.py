import os
import sys
import numpy as np
import matplotlib.pyplot as plt

current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)

sys.path.append(parent_path)
sys.path.append("../Traj_planning")

from RK4 import RK4
from parameters import *


######################### Creating the EOM ##############################
def ode(last_sol, u):
    [
        x,
        x_dot,
        y,
        y_dot,
        e1bx,
        e1by,
        e2bx,
        e2by,
        e1tx,
        e1ty,
        e2tx,
        e2ty,
        omega_z,
        thrust,
    ] = last_sol

    sol = np.array(
        [
            x_dot,  # x
            thrust / m * e1tx,  # v_x
            y_dot,  # y
            thrust / m * e1ty - g,  # v_y
            omega_z * e2bx,  # e1bx
            omega_z * e2by,  # e1by
            -omega_z * e1bx,  # e2bx
            -omega_z * e1by,  # e2by
            (u[1] + omega_z) * e2tx,  # e1tx
            (u[1] + omega_z) * e2ty,  # e1ty
            -(u[1] + omega_z) * e1tx,  # e2tx
            -(u[1] + omega_z) * e1ty,  # e2ty
            -thrust * l_tvc * (e1tx * e2bx + e1ty * e2by) / J_z,  # omega
            u[0],  # thrust
        ]
    )

    return sol


def plot_psd(time_points, x):
    # Calculate the Power Spectral Density (PSD)
    sampling_rate = 1 / (time_points[1] - time_points[0])  # Calculate the sampling rate
    frequencies, psd = (
        np.fft.rfftfreq(len(time_points), d=1 / sampling_rate),
        np.abs(np.fft.rfft(x)) ** 2,
    )

    # Plot the PSD
    plt.figure()
    plt.plot(frequencies, psd)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.title("Power Spectral Density")
    # plt.show()


def plot_states(
    time, sim_states, trajectory_params, f, delta_tvc, f_dot, delta_tvc_dot
):
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(
        3, 3, figsize=(15, 10)
    )

    ax1.plot(sim_states[0, :], sim_states[2, :])
    ax1.plot(trajectory_params["x"], trajectory_params["y"])
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Trajectory")
    ax1.grid()
    ax1.legend([r"$RK4$", "Analytical"])
    ax1.axis("equal")

    ax2.plot(time, sim_states[1, :])
    ax2.plot(trajectory_params["t"], trajectory_params["vx"])
    ax2.set_xlabel("t")
    ax2.set_ylabel("vx")
    ax2.set_title("Velocity in x")
    ax2.grid()
    ax2.legend([r"$RK4$", "Analytical"])

    ax3.plot(time, sim_states[3, :])
    ax3.plot(trajectory_params["t"], trajectory_params["vy"])
    ax3.set_xlabel("t")
    ax3.set_ylabel("vy")
    ax3.set_title("Velocity in y")
    ax3.grid()
    ax3.legend([r"$RK4$", "Analytical"])

    gamma = np.arctan2(sim_states[5, :], sim_states[4, :])
    ax4.plot(time, np.rad2deg(gamma))
    ax4.plot(trajectory_params["t"], np.rad2deg(trajectory_params["gamma"]))
    ax4.set_xlabel("t")
    ax4.set_ylabel("gamma")
    ax4.set_title("Gamma")
    ax4.grid()
    ax4.legend([r"$RK4$", "Analytical"])

    ax5.plot(time, sim_states[13, :])
    ax5.plot(trajectory_params["t"], trajectory_params["f"])
    ax5.set_xlabel("t")
    ax5.set_ylabel("thrust")
    ax5.set_title("Thrust")
    ax5.grid()
    ax5.legend([r"$RK4$", "Analytical"])

    ax6.plot(time, np.rad2deg(delta_tvc))
    ax6.plot(trajectory_params["t"], np.rad2deg(trajectory_params["delta_tvc"]))
    ax6.set_xlabel("t")
    ax6.set_ylabel("delta_tvc")
    ax6.set_title("Delta_tvc")
    ax6.grid()
    ax6.legend([r"$RK4$", "Analytical"])

    ax7.plot(time, np.rad2deg(sim_states[12, :]))
    ax7.plot(trajectory_params["t"], np.rad2deg(trajectory_params["gamma_dot"]))
    ax7.set_xlabel("t")
    ax7.set_ylabel("$\dot{\gamma}$")
    ax7.set_title("Gamma_dot")
    ax7.grid()
    ax7.legend([r"$RK4$", "Analytical"])

    ax8.plot(time, f_dot)
    ax8.plot(trajectory_params["t"], trajectory_params["f_dot"])
    ax8.set_xlabel("t")
    ax8.set_ylabel("thrust_dot")
    ax8.set_title("Thrust_dot")
    ax8.grid()
    ax8.legend([r"$RK4$", "Analytical"])

    ax9.plot(time, delta_tvc_dot)
    ax9.plot(trajectory_params["t"], trajectory_params["delta_tvc_dot"])
    ax9.set_xlabel("t")
    ax9.set_ylabel("delta_tvc_dot (rad/s)")
    ax9.set_title("Delta_tvc_dot")
    ax9.grid()
    ax9.legend([r"$RK4$", "Analytical"])

    plot_psd(time, delta_tvc_dot)

    plt.show()


########################## Numerical integration ###############################
def drift_checker(env_params, trajectory_params, plot=False):
    print("Checking drift...")
    # retrieve control inputs
    f_dot = trajectory_params["f_dot"]
    delta_tvc_dot = trajectory_params["delta_tvc_dot"]

    # retrieve total time
    total_time = trajectory_params["t"][-1]

    time = np.linspace(0, total_time, len(f_dot), endpoint=True)
    dt = time[1] - time[0]

    sim_states = np.zeros(
        (len(initial_state), len(time))
    )  # the states are in the columns
    sim_states[:, 0] = initial_state

    for i in range(len(time) - 1):
        sim_states[:, i + 1] = RK4(
            ode, sim_states[:, i], [f_dot[i], delta_tvc_dot[i]], dt
        )

        # normalization
        sim_states[4:6, i + 1] /= np.linalg.norm(sim_states[4:6, i + 1])
        sim_states[6:8, i + 1] /= np.linalg.norm(sim_states[6:8, i + 1])
        sim_states[8:10, i + 1] /= np.linalg.norm(sim_states[8:10, i + 1])
        sim_states[10:12, i + 1] /= np.linalg.norm(sim_states[10:12, i + 1])

    gamma = np.arctan2(sim_states[5, :], sim_states[4, :])
    delta_tvc = np.arctan2(sim_states[9, :], sim_states[8, :]) - gamma
    f = sim_states[13, :]

    pos_drift = np.max(
        np.abs(
            np.array([sim_states[0], sim_states[2]])
            - np.array([trajectory_params["x"], trajectory_params["y"]])
        )
    )
    pos_drift = np.linalg.norm(pos_drift)
    max_pos = np.linalg.norm(
        [np.max(np.abs(trajectory_params["x"])), np.abs(np.max(trajectory_params["y"]))]
    )
    angular_drift = np.max(np.abs(gamma - trajectory_params["gamma"]))

    if plot:
        plot_states(
            time, sim_states, trajectory_params, f, delta_tvc, f_dot, delta_tvc_dot
        )

    if (
        pos_drift / max_pos < env_params["max_drift"]
        and np.abs(angular_drift) < env_params["max_angular_drift"]
    ):
        print("Drift check passed!")

    elif pos_drift / max_pos >= env_params["max_drift"]:
        raise ValueError(
            "Position drift too high!\nThe trajectory is not feasible from the numerical integration point of view. Please, check the trajectory and controller parameters."
        )

    else:
        raise ValueError(
            "Angular drift too high!\nThe trajectory is not feasible from the numerical integration point of view. Please, check the trajectory and controller parameters."
        )
