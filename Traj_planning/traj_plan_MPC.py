import sys
import os

# Get the path of the parent folder
parent_folder_path = os.path.abspath(os.path.join(os.getcwd(), ".."))

# Add the parent folder path to the system path
sys.path.append(parent_folder_path)

sys.path.append("../")
sys.path.append("../Traj_planning")

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from RK4 import RK4

# from animate import *

# from Traj_planning.RK4 import RK4

# System model (example: simple 1D vehicle dynamics)
def system_model(state, control_input, dt):
    # Update the state based on the system dynamics
    updated_state = state + control_input * dt
    return updated_state


# MPC trajectory planning
def mpc_trajectory_planning(
    env_params, rocket_params, controller_params, states, constraints
):
    # rocket and environment parameters
    g = env_params["g"]
    m = rocket_params["m"]
    C = rocket_params["C"]
    K_tvc = rocket_params["K_tvc"]
    T_tvc = rocket_params["T_tvc"]
    l_tvc = rocket_params["l_tvc"]
    K_thrust = rocket_params["K_thrust"]
    T_thrust = rocket_params["T_thrust"]

    # controller parameters
    dt = controller_params["dt"]
    thrust_bounds = controller_params["thrust_bounds"]
    delta_tvc_bounds = controller_params["delta_tvc_bounds"]
    u_bounds = controller_params["u_bounds"]

    # retrieve the trajectory parameters
    desired_times = states["t"]
    total_time = desired_times[-1]
    x_ref = states["x"]
    y_ref = states["y"]
    z_ref = states["z"]
    vx_ref = states["vx"]
    vy_ref = states["vy"]
    vz_ref = states["vz"]
    ax_ref = states["ax"]
    ay_ref = states["ay"]
    az_ref = states["az"]
    gamma_ref = states["gamma"]
    gamma_dot_ref = states["gamma_dot"]
    gamma_ddot_ref = states["gamma_ddot"]

    # check if the desired points, velocities and times have the same length
    if (
        len(x_ref) != len(vx_ref)
        or len(x_ref) != len(desired_times)
        or len(gamma_ref) != len(desired_times)
    ):
        raise ValueError(
            "The length of desired_points, desired_velocities and desired_times must be the same"
        )

    num_steps = int(total_time / dt)
    num_targets = len(x_ref)

    # state parameters
    t0_val = controller_params["t0"]
    x0_val = controller_params["x0"]

    Q = controller_params["Q"]
    # Qf = controller_params["Qf"] # not needed, I'll use constraints
    R = controller_params["R"]

    # initializing the state varibles
    x = ca.SX.sym("x")  # states
    x_dot = ca.SX.sym("x_dot")  # states
    y = ca.SX.sym("y")  # states
    y_dot = ca.SX.sym("y_dot")  # states
    gamma = ca.SX.sym("gamma")  # states
    gamma_dot = ca.SX.sym("gamma_dot")  # states
    thrust = ca.SX.sym("thrust")  # states
    delta_tvc = ca.SX.sym("delta_tvc")  # states

    # create a vector with variable names
    variables = ca.vertcat(x, x_dot, y, y_dot, gamma, gamma_dot, thrust, delta_tvc)

    # initializing the control varibles
    thrust_dot = ca.SX.sym("thrust_dot")  # controls
    delta_tvc_dot = ca.SX.sym("delta_tvc_dot")  # controls

    # create a vector with variable names
    u = ca.vertcat(thrust_dot, delta_tvc_dot)

    # define the nonlinear ode
    ode = ca.vertcat(
        x_dot,
        thrust * ca.cos(gamma - delta_tvc) / m,
        y_dot,
        thrust * ca.sin(gamma - delta_tvc) / m - g,
        gamma_dot,
        -ca.power(gamma_dot, 2) - thrust * l_tvc * ca.sin(delta_tvc) / C,
        thrust_dot,
        delta_tvc_dot,
    )

    # creating the ploblem statement function
    f = ca.Function(
        "f",
        [variables, u],
        [ode],
        [
            "[x, x_dot, y, y_dot, gamma, gamma_dot, delta_tvc, thrust]",
            "[thrust, delta_tvc_ref]",
        ],
        ["ode"],
    )

    # integrator
    x_next = RK4(f, variables, u, dt)

    F = ca.Function(
        "F",
        [variables, u],
        [x_next],
        [
            "[x, x_dot, y, y_dot, gamma, gamma_dot, delta_tvc, thrust]",
            "[thrust, delta_tvc_ref]",
        ],
        ["x_next"],
    )

    # building the optimal control problem
    opti = ca.Opti()

    # create the state, control and initial state varibles
    x = opti.variable(8, num_steps + 1)
    u = opti.variable(2, num_steps)
    x_target = opti.parameter(8, num_targets)

    # Formulate the optimization problem
    cost = 0
    last_k = 0
    print("desired_times: ", desired_times)
    print("Number of targets: ", num_targets)
    for i in range(num_steps):
        t = i * dt
        k = np.searchsorted(desired_times, t)

        if k >= num_targets:
            k = num_targets - 1

        if k > last_k or t == desired_times[k]:
            # add constraints
            opti.subject_to(x[0, i] == x_ref[k])  # x
            opti.subject_to(x[2, i] == y_ref[k])  # y
            # opti.subject_to(x[1, i]) == desired_velocities[k, 0] # x_dot
            # opti.subject_to(x[3, i]) == desired_velocities[k, 1] # y_dot
            last_k += 1

            print("k: ", k)
            print("Constraints added at t = ", t)
            print("x_ref[{:.0f}]: {:.1f}".format(k, x_ref[k]))
            print("y_ref[{:.0f}]: {:.1f}".format(k, y_ref[k]))

        # add final cost
        tracking_error = x[:, i] - x_target[:, k]
        cost += ca.mtimes([tracking_error.T, Q, tracking_error]) + ca.mtimes(
            [u[:, i].T, R, u[:, i]]
        )

    # check if all constraints were added
    if t < total_time:
        opti.subject_to(x[0, i] == x_ref[k])  # x
        opti.subject_to(x[2, i] == y_ref[k])  # y

        print("k: ", k)
        print("Constraints added at t = ", t)
        print("x_ref[{:.0f}]: {:.1f}".format(k, x_ref[k]))
        print("y_ref[{:.0f}]: {:.1f}".format(k, y_ref[k]))

    # Add final cost
    tracking_error = x[:, num_steps] - x_target[:, k]
    cost += ca.mtimes([tracking_error.T, Q, tracking_error])

    # Set the cost to be minimized
    opti.minimize(cost)

    # set the initial position
    opti.subject_to(x[:, 0] == x0_val)

    # set the constraints (dynamics)
    for k in range(num_steps):
        # apply the dynamics as constraints
        opti.subject_to(x[:, k + 1] == F(x[:, k], u[:, k]))

        # apply the control input constraints
        opti.subject_to(u[0, k] >= u_bounds[0][0])
        opti.subject_to(u[0, k] <= u_bounds[0][1])
        opti.subject_to(u[1, k] >= u_bounds[1][0])
        opti.subject_to(u[1, k] <= u_bounds[1][1])

        # thrust = x[6, k]
        opti.subject_to(x[6, k] >= thrust_bounds[0])
        opti.subject_to(x[6, k] <= thrust_bounds[1])

        # delta_tvc = x[7, k]
        opti.subject_to(x[7, k] >= delta_tvc_bounds[0])
        opti.subject_to(x[7, k] <= delta_tvc_bounds[1])

    # set the target positions
    for i in range(num_targets):
        states_ref = ca.vertcat(
            x_ref[i],
            y_ref[i],
            vx_ref[i] if vx_ref[i] is not None else 0,
            vy_ref[i] if vy_ref[i] is not None else 0,
            gamma_ref[i] if gamma_ref[i] is not None else 0,
            gamma_dot_ref[i] if gamma_dot_ref[i] is not None else 0,
            0,
            0,
        )

        opti.set_value(x_target[:, i], states_ref)

    # select the desired solver
    opti.solver("ipopt")

    sol = opti.solve()

    return sol.value(x), sol.value(u)


def plot_simulation(x, u, dt, controller_params, trajectory_params):
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 12))

    t = np.linspace(0, len(x[0]) * dt, len(x[0]))
    # plot 1: x, and y
    ax1.plot(t, x[0, :])
    ax1.plot(t, x[1, :])
    ax1.plot(t, x[2, :])
    ax1.plot(t, x[3, :])
    ax1.plot(trajectory_params["t"], trajectory_params["x"], "x")
    ax1.plot(trajectory_params["t"], trajectory_params["y"], "o")
    ax1.legend(["$x$", "$v_x$", "$y$", "$v_y$"])
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Position (m)")
    ax1.set_title("Position vs Time")
    ax1.grid()

    thrust_deriv_list = u[0]
    delta_tvc_ref_deriv_list = u[1]
    u_bounds = controller_params["u_bounds"]
    # gamma_bounds = controller_params["gamma_bounds"]
    thrust_bounds = controller_params["thrust_bounds"]
    delta_tvc_bounds = controller_params["delta_tvc_bounds"]

    # plot 2: thrust
    ax2.plot(t, x[6, :])
    ax2.plot(t, [thrust_bounds[0]] * len(t), "--", color="black")
    ax2.plot(t, [thrust_bounds[1]] * len(t), "--", color="black")
    ax2.legend(["$f$", "$f_{min}$", "$f_{max}$"])
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Thrust (N)")
    ax2.set_title("Thrust vs Time")
    ax2.grid()

    # plot 3: thrust derivative
    ax3.plot(t[0 : len(t) - 1], thrust_deriv_list)
    ax3.plot(t, [u_bounds[0][0]] * len(t), "--", color="black")
    ax3.plot(t, [u_bounds[0][1]] * len(t), "--", color="black")
    ax3.legend(["$\dot{f}$", "$\dot{f}_{ref}$", "$\dot{f}_{max}$"])
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Thrust derivative (N/s)")
    ax3.set_title("Thrust derivative vs Time")
    ax3.grid()

    # plot 4: gamma
    ax4.plot(t, np.rad2deg(x[4, :]), label="$\\gamma$")
    # ax4.plot(t, [np.rad2deg(gamma_bounds[0])] * len(t), "--", color="black", label="$\\gamma_{min}$")
    # ax4.plot(t, [np.rad2deg(gamma_bounds[1])] * len(t), "--", color="black", label="$\\gamma_{max}$")
    ax4.legend()
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Angle (degrees)")
    ax4.set_title("Angle vs Time")
    ax4.grid()

    # plot 5: delta_tvc
    ax5.plot(t, np.rad2deg(x[7, :]))
    ax5.plot(t, [np.rad2deg(delta_tvc_bounds[0])] * len(t), "--", color="black")
    ax5.plot(t, [np.rad2deg(delta_tvc_bounds[1])] * len(t), "--", color="black")
    ax5.legend(
        ["$\\delta_{tvc}$", "$\\delta_{tvc_{min}}$", "$\\delta_{tvc_{max}}$"],
        loc="upper right",
    )
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("$\\delta_{tvc}$ (degrees)")
    ax5.set_title("$\\delta_{tvc}$ vs Time")
    ax5.grid()

    # plot 5: delta_tvc
    ax6.plot(t[0 : len(t) - 1], np.rad2deg(delta_tvc_ref_deriv_list))
    ax6.plot(t, [np.rad2deg(u_bounds[1][0])] * len(t), "--", color="black")
    ax6.plot(t, [np.rad2deg(u_bounds[1][1])] * len(t), "--", color="black")
    ax6.legend(
        [
            "$\\dot{\\delta}_{tvc}$",
            "$\\dot{\\delta}_{tvc_{min}}$",
            "$\\dot{\\delta}_{tvc_{max}}$",
        ],
        loc="upper right",
    )
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("$\\dot{\\delta}_{tvc}$ (degrees)")
    ax6.set_title("$\\dot{\\delta}_{tvc}$ vs Time")
    ax6.grid()

    plt.tight_layout()
    fig.show()
