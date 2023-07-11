import numpy as np
import matplotlib.pyplot as plt

import sys
import os

# Get the parent folder path
parent_folder = os.path.dirname(os.path.abspath(__file__))

# Add the parent folder to sys.path
sys.path.append(os.path.join(parent_folder, ".."))

from Traj_planning.old.pol_interpolation import *
from Traj_planning.traj_3ST.diff_flat import *
from Traj_planning.old.MPC_traj import *
from Traj_planning.traj_3ST.auxiliar_codes.plot_traj import *
from performance_estimation.estimate_control import estimate_control
from performance_estimation.new_est_ctrl import new_estimate_control
from Traj_planning.examples.simple_circ import *
from parameters import *


# environment variables
g = 9.81  # gravity

# rocket variables
m = 100  # mass of the hopper
# mf = 50  # final mass of the hopper
h = 2  # height of the hopper
radius = 0.25  # radius of the hopper
l_tvc = 0.5  # distance from the center of mass to the TVC
J = 1 / 12 * m * (3 * radius**2 + h**2)  # moment of inertia of the hopper


def trajenerator_3ST(states, constraints, env_params, rocket_params, controller_params):
    # interpolate polinomials for minimum snap trajectory
    Px_coeffs, Py_coeffs, Pz_coeffs, t = min_snap_traj(
        states, constraints, rocket_params, controller_params
    )

    # calculate gamma using differential flatness
    trajectory_params = diff_flat_traj(
        Px_coeffs, Py_coeffs, Pz_coeffs, t, env_params, rocket_params, controller_params
    )

    plot_trajectory(states, trajectory_params, "Diff flat trajectory")

    # get state estimates using analytical solution
    estimated_states = estimate_control(
        Px_coeffs, Py_coeffs, Pz_coeffs, t, env_params, rocket_params, controller_params
    )
    new_estimated_states = new_estimate_control(
        Px_coeffs, Py_coeffs, Pz_coeffs, t, env_params, rocket_params, controller_params
    )

    return trajectory_params, estimated_states, new_estimated_states


def RK4(fun, x, u, dt):
    k1 = fun(x, u)
    k2 = fun(x + dt / 2 * k1, u)
    k3 = fun(x + dt / 2 * k2, u)
    k4 = fun(x + dt * k3, u)
    x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_next


def f_ode(states, u):
    x = states[0]
    x_dot = states[1]
    y = states[2]
    y_dot = states[3]
    e1bx = states[4]
    e1by = states[5]
    e2bx = states[6]
    e2by = states[7]
    e1tx = states[8]
    e1ty = states[9]
    e2tx = states[10]
    e2ty = states[11]
    omega_z = states[12]
    thrust = states[13]

    thrust_dot = u[0]
    delta_tvc_dot = u[1]

    f_dot = np.array(
        [
            x_dot,  # x
            thrust / m * e1tx,  # v_x
            y_dot,  # y
            thrust / m * e1ty - g,  # v_y
            omega_z * e2bx,  # e1bx
            omega_z * e2by,  # e1by
            -omega_z * e1bx,  # e2bx
            -omega_z * e1by,  # e2by
            (delta_tvc_dot + omega_z) * e2tx,  # e1tx
            (delta_tvc_dot + omega_z) * e2ty,  # e1ty
            -(delta_tvc_dot + omega_z) * e1tx,  # e2tx
            -(delta_tvc_dot + omega_z) * e1ty,  # e2ty
            -thrust * l_tvc * (e1tx * e2bx + e1ty * e2by) / J,  # omega
            thrust_dot,  # thrust
        ]
    )

    return f_dot


v = 15
r = 50

# constraints
max_vx = 150
min_vx = -150
max_vy = 150
min_vy = -150
max_vz = 150
min_vz = -150

max_ax = 2 * g
min_ax = -2 * g
max_ay = 2 * g
min_ay = -2 * g
max_az = 2 * g
min_az = -2 * g

############################################################################################

# x, y, vx, vy, t, e1bx, e1by = calculate_traj_params(v, r, m, g)
# z = np.zeros_like(x)

# none_vec = np.array([None] * len(x))
# # none_vec[0] = 0

# states = {
#     "t": t,
#     "x": x,
#     "y": y,
#     "z": z,
#     "vx": none_vec,
#     "vy": none_vec,
#     "vz": none_vec,
#     "ax": none_vec,
#     "ay": none_vec,
#     "az": none_vec,
# }

############################################################################################

target_points = np.array(
    [
        [0, 0, 0],
        [20, 50, 0],
        [50, 50, 0],
        [80, 50, 0],
        [100, 0, 0],
    ]
)

# gamma_points_deg = np.array([90, 90, 90, 90])
gamma_dot_points = np.array([0, None, None, None, 0])
target_velocities = np.array(
    [[0, 0, 0], [None, None, None], [None, None, None], [None, None, None], [0, 0, 0]]
)  # m/s
target_accelerations = np.array(
    [[0, 0, 0], [None, None, None], [None, None, None], [None, None, None], [0, 0, 0]]
)  # m/s^2
time_points = 1 * np.array([0, 10, 15, 20, 30])  # time list

states = {
    "t": time_points,
    "x": target_points[:, 0],
    "y": target_points[:, 1],
    "z": target_points[:, 2],
    "vx": target_velocities[:, 0],
    "vy": target_velocities[:, 1],
    "vz": target_velocities[:, 2],
    "ax": target_accelerations[:, 0],
    "ay": target_accelerations[:, 1],
    "az": target_accelerations[:, 2],
    "gamma_dot": gamma_dot_points,
}
##########################################################################################
# target_points = np.array(
#     [
#         [0, 0, 0],
#         [0, 100, 0],
#     ]
# )

# gamma_points_deg = np.array([90, 90])
# target_velocities = np.array([[0, 0, 0], [0, 0, 0]])  # m/s
# target_accelerations = np.array([[0, 0, 0], [0, 0, 0]])  # m/s^2
# time_points = np.array([0, 10])  # time list

# states = {
#     "t": time_points,
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

############################################################################################

constraints = {
    "min_vx": min_vx,
    "max_vx": max_vx,
    "min_vy": min_vy,
    "max_vy": max_vy,
    "min_vz": min_vz,
    "max_vz": max_vz,
    "min_ax": min_ax,
    "max_ax": max_ax,
    "min_ay": min_ay,
    "max_ay": max_ay,
    "min_az": min_az,
    "max_az": max_az,
    "g": g,
}

trajectory_params, estimated_states, new_estimated_states = trajenerator_3ST(
    states, constraints, env_params, rocket_params, controller_params
)


########### Estimation begin ############
# dt = controller_params["dt"]
dt = 1e-3
T = estimated_states["t"][-1]  # total time
N = int(T / dt)
t = np.linspace(0, T, N)

initial_state = np.array(
    [
        estimated_states["x"][0],
        estimated_states["vx"][0],
        estimated_states["y"][0],
        estimated_states["vy"][0],
        estimated_states["e1bx"][0],
        estimated_states["e1by"][0],
        estimated_states["e2bx"][0],
        estimated_states["e2by"][0],
        estimated_states["e1tx"][0],
        estimated_states["e1ty"][0],
        estimated_states["e2tx"][0],
        estimated_states["e2ty"][0],
        estimated_states["gamma_dot"][0],
        estimated_states["f"][0],
    ]
)

new_initial_state = np.array(
    [
        new_estimated_states["x"][0],
        new_estimated_states["vx"][0],
        new_estimated_states["y"][0],
        new_estimated_states["vy"][0],
        new_estimated_states["e1bx"][0],
        new_estimated_states["e1by"][0],
        new_estimated_states["e2bx"][0],
        new_estimated_states["e2by"][0],
        new_estimated_states["e1tx"][0],
        new_estimated_states["e1ty"][0],
        new_estimated_states["e2tx"][0],
        new_estimated_states["e2ty"][0],
        new_estimated_states["gamma_dot"][0],
        new_estimated_states["f"][0],
    ]
)

sim_states = np.zeros((N, len(initial_state)))
new_sim_states = np.zeros((N, len(initial_state)))

sim_states[0] = initial_state
new_sim_states[0] = new_initial_state

for i in range(N - 1):
    sim_states[i + 1] = RK4(
        f_ode,
        sim_states[i],
        [estimated_states["f_dot"][i], estimated_states["delta_tvc_dot"][i]],
        dt,
    )

    # normalization
    sim_states[i + 1, 4:6] /= np.linalg.norm(sim_states[i + 1, 4:6])
    sim_states[i + 1, 6:8] /= np.linalg.norm(sim_states[i + 1, 6:8])
    sim_states[i + 1, 8:10] /= np.linalg.norm(sim_states[i + 1, 8:10])
    sim_states[i + 1, 10:12] /= np.linalg.norm(sim_states[i + 1, 10:12])

    # apply for new estimated states
    new_sim_states[i + 1] = RK4(
        f_ode,
        new_sim_states[i],
        [new_estimated_states["f_dot"][i], new_estimated_states["delta_tvc_dot"][i]],
        dt,
    )

    # normalization
    new_sim_states[i + 1, 4:6] /= np.linalg.norm(new_sim_states[i + 1, 4:6])
    new_sim_states[i + 1, 6:8] /= np.linalg.norm(new_sim_states[i + 1, 6:8])
    new_sim_states[i + 1, 8:10] /= np.linalg.norm(new_sim_states[i + 1, 8:10])
    new_sim_states[i + 1, 10:12] /= np.linalg.norm(new_sim_states[i + 1, 10:12])

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(
    3, 3, figsize=(15, 10)
)

ax1.plot(sim_states[:, 0], sim_states[:, 2])
ax1.plot(new_sim_states[:, 0], new_sim_states[:, 2])
ax1.plot(estimated_states["x"], estimated_states["y"])
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title("Trajectory")
ax1.grid()
ax1.legend([r"$RK4_{old}$", r"$RK4_{new}$", "Analytical"])
ax1.axis("equal")

ax2.plot(t, sim_states[:, 1])
ax2.plot(t, new_sim_states[:, 1])
ax2.plot(estimated_states["t"], estimated_states["vx"])
ax2.set_xlabel("t")
ax2.set_ylabel("vx")
ax2.set_title("Velocity in x")
ax2.grid()
ax2.legend([r"$RK4_{old}$", r"$RK4_{new}$", "Analytical"])

ax3.plot(t, sim_states[:, 3])
ax3.plot(t, new_sim_states[:, 3])
ax3.plot(estimated_states["t"], estimated_states["vy"])
ax3.set_xlabel("t")
ax3.set_ylabel("vy")
ax3.set_title("Velocity in y")
ax3.grid()
ax3.legend([r"$RK4_{old}$", r"$RK4_{new}$", "Analytical"])

gamma = np.arctan2(sim_states[:, 5], sim_states[:, 4])
gamma_new = np.arctan2(new_sim_states[:, 5], new_sim_states[:, 4])
ax4.plot(t, np.rad2deg(gamma))
ax4.plot(t, np.rad2deg(gamma_new))
ax4.plot(estimated_states["t"], np.rad2deg(estimated_states["gamma"]))
ax4.set_xlabel("t")
ax4.set_ylabel("gamma")
ax4.set_title("Gamma")
ax4.grid()
ax4.legend([r"$RK4_{old}$", r"$RK4_{new}$", "Analytical"])

# ax4.plot(t, sim_states[:, 4], label="e1bx")
# ax4.plot(t, sim_states[:, 5], label="e1by")
# ax4.plot(t, sim_states[:, 6], label="e2bx")
# ax4.plot(t, sim_states[:, 7], label="e2by")
# ax4.set_xlabel("t")
# ax4.set_ylabel("e1bx, e1by, e2bx, e2by")
# ax4.grid()
# ax4.legend()

ax5.plot(t, sim_states[:, 13])
ax5.plot(t, new_sim_states[:, 13])
ax5.plot(estimated_states["t"], estimated_states["f"])
ax5.set_xlabel("t")
ax5.set_ylabel("thrust")
ax5.set_title("Thrust")
ax5.grid()
ax5.legend([r"$RK4_{old}$", r"$RK4_{new}$", "Analytical"])

# delta_tvc = np.zeros(len(t))
# for i in range(len(delta_tvc)):
#     delta_tvc[i] = np.arccos(np.dot(states[i, 8:9], states[i, 4:5]) / (np.linalg.norm(states[i, 8:9]) * np.linalg.norm(states[i, 4:5])))

delta_tvc = np.arctan2(sim_states[:, 9], sim_states[:, 8]) - gamma
delta_tvc_new = np.arctan2(new_sim_states[:, 9], new_sim_states[:, 8]) - gamma_new
ax6.plot(t, np.rad2deg(delta_tvc))
ax6.plot(t, np.rad2deg(delta_tvc_new))
ax6.plot(estimated_states["t"], np.rad2deg(estimated_states["delta_tvc"]))
ax6.set_xlabel("t")
ax6.set_ylabel("delta_tvc")
ax6.set_title("Delta_tvc")
ax6.grid()
ax6.legend([r"$RK4_{old}$", r"$RK4_{new}$", "Analytical"])

# ax6.plot(t, sim_states[:, 8], label="e1tx")
# ax6.plot(t, sim_states[:, 9], label="e1ty")
# ax6.plot(t, sim_states[:, 10], label="e2tx")
# ax6.plot(t, sim_states[:, 11], label="e2ty")
# ax6.set_xlabel("t")
# ax6.set_ylabel("e1tx, e1ty, e2tx, e2ty")
# ax6.grid()
# ax6.legend()

ax7.plot(t, np.rad2deg(sim_states[:, 12]))
ax7.plot(t, np.rad2deg(new_sim_states[:, 12]))
ax7.plot(estimated_states["t"], np.rad2deg(estimated_states["gamma_dot"]))
ax7.set_xlabel("t")
ax7.set_ylabel("$\dot{\gamma}$")
ax7.set_title("Gamma_dot")
ax7.grid()
ax7.legend([r"$RK4_{old}$", r"$RK4_{new}$", "Analytical"])

ax8.plot(t, get_derivative(t, sim_states[:, 13]))
ax8.plot(t, get_derivative(t, new_sim_states[:, 13]))
ax8.plot(estimated_states["t"], estimated_states["f_dot"])
ax8.set_xlabel("t")
ax8.set_ylabel("thrust_dot")
ax8.set_title("Thrust_dot")
ax8.grid()
ax8.legend([r"$RK4_{old}$", r"$RK4_{new}$", "Analytical"])

ax9.plot(t, np.rad2deg(get_derivative(t, delta_tvc)))
ax9.plot(t, np.rad2deg(get_derivative(t, delta_tvc_new)))
ax9.plot(estimated_states["t"], np.rad2deg(estimated_states["delta_tvc_dot"]))
ax9.set_xlabel("t")
ax9.set_ylabel("delta_tvc_dot")
ax9.set_title("Delta_tvc_dot")
ax9.grid()
ax9.legend([r"$RK4_{old}$", r"$RK4_{new}$", "Analytical"])


########################## Estimation end ##########################


fig_2, (
    (ax1_2, ax2_2, ax3_2),
    (ax4_2, ax5_2, ax6_2),
    (ax7_2, ax8_2, ax9_2),
) = plt.subplots(3, 3, figsize=(15, 10))

ax1_2.plot(estimated_states["t"], estimated_states["f1"], label="f1")
ax1_2.plot(new_estimated_states["t"], new_estimated_states["f1"], label="f1_new")
ax1_2.grid()
ax1_2.legend()
ax1_2.set_title("Estimated f1")
ax1_2.set_xlabel("t")
ax1_2.set_ylabel("f1")

ax2_2.plot(estimated_states["t"], estimated_states["f2"], label="f2")
ax2_2.plot(new_estimated_states["t"], new_estimated_states["f2"], label="f2_new")
ax2_2.grid()
ax2_2.legend()
ax2_2.set_title("Estimated f2")
ax2_2.set_xlabel("t")
ax2_2.set_ylabel("f2")


ax3_2.plot(estimated_states["t"], estimated_states["f"], label="f")
ax3_2.plot(new_estimated_states["t"], new_estimated_states["f"], label="f_new")
ax3_2.grid()
ax3_2.legend()
ax3_2.set_title("Estimated f")
ax3_2.set_xlabel("t")
ax3_2.set_ylabel("f")

ax4_2.plot(estimated_states["t"], estimated_states["f1_dot"], label="f1_dot")
ax4_2.plot(
    new_estimated_states["t"], new_estimated_states["f1_dot"], label="f1_dot_new"
)
ax4_2.grid()
ax4_2.legend()
ax4_2.set_title("Estimated f1_dot")
ax4_2.set_xlabel("t")
ax4_2.set_ylabel("f1_dot")


ax5_2.plot(estimated_states["t"], estimated_states["f2_dot"], label="f2_dot")
ax5_2.plot(
    new_estimated_states["t"], new_estimated_states["f2_dot"], label="f2_dot_new"
)
ax5_2.grid()
ax5_2.legend()
ax5_2.set_title("Estimated f2_dot")
ax5_2.set_xlabel("t")
ax5_2.set_ylabel("f2_dot")


ax6_2.plot(estimated_states["t"], estimated_states["f_dot"], label="f_dot")
ax6_2.plot(new_estimated_states["t"], new_estimated_states["f_dot"], label="f_dot_new")
ax6_2.grid()
ax6_2.legend()
ax6_2.set_title("Estimated f_dot")
ax6_2.set_xlabel("t")
ax6_2.set_ylabel("f_dot")


ax7_2.plot(
    estimated_states["t"], np.rad2deg(estimated_states["delta_tvc"]), label="delta_tvc"
)
ax7_2.plot(
    new_estimated_states["t"],
    np.rad2deg(new_estimated_states["delta_tvc"]),
    label="delta_tvc_new",
)
ax7_2.grid()
ax7_2.legend()
ax7_2.set_title("Estimated delta_tvc")
ax7_2.set_xlabel("t")
ax7_2.set_ylabel("delta_tvc (deg)")


ax8_2.plot(
    estimated_states["t"], estimated_states["delta_tvc_dot"], label="delta_tvc_dot"
)
ax8_2.plot(
    new_estimated_states["t"],
    new_estimated_states["delta_tvc_dot"],
    label="delta_tvc_dot_new",
)
ax8_2.grid()
ax8_2.legend()
ax8_2.set_title("Estimated delta_tvc_dot")
ax8_2.set_xlabel("t")
ax8_2.set_ylabel("delta_tvc_dot (rad/s)")


ax9_2.plot(estimated_states["t"], np.rad2deg(estimated_states["gamma"]), label="gamma")
ax9_2.plot(
    new_estimated_states["t"], np.rad2deg(new_estimated_states["gamma"]), label="gamma"
)
ax9_2.grid()
ax9_2.legend()
ax9_2.set_title("Estimated gamma")
ax9_2.set_xlabel("t")
ax9_2.set_ylabel("gamma (deg)")


# experiment
fig_3, (
    ax1_3,
    ax2_3,
    ax3_3,
) = plt.subplots(1, 3, figsize=(15, 5))

ax1_3.plot(new_estimated_states["t"], new_estimated_states["error_ax"], label="e_ax")
ax1_3.set_xlabel("Time (s)")
ax1_3.set_ylabel("Error (N)")
ax1_3.grid()
ax1_3.legend()

ax2_3.plot(new_estimated_states["t"], new_estimated_states["error_ay"], label="e_ay")
ax2_3.set_xlabel("Time (s)")
ax2_3.set_ylabel("Error (N)")
ax2_3.grid()
ax2_3.legend()

ax3_3.plot(
    new_estimated_states["t"],
    new_estimated_states["error_gamma_ddot"],
    label="e_agamma",
)
ax3_3.set_xlabel("Time (s)")
ax3_3.set_ylabel("Error (Nm)")
ax3_3.grid()
ax3_3.legend()

plt.show()
