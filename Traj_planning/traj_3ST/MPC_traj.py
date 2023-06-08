import casadi as ca
import sys

sys.path.append(r"D:\Code\SA2023\rocket-control-framework")
from RK4 import RK4
import numpy as np


def get_derivative(t, f):
    """Calculate the derivative of the array f."""
    df = np.zeros_like(f)

    h = t[1] - t[0]
    df[0] = (-f[2] + 4 * f[1] - 3 * f[0]) / (2 * h)

    # Treat right border
    h = t[-1] - t[-2]
    df[-1] = (3 * f[-1] - 4 * f[-2] + f[-3]) / (2 * h)

    # center of the array
    for i in range(1, len(f) - 1):
        df[i] = (f[i + 1] - f[i - 1]) / (2 * h)

    return df


def MPC_traj(
    env_params,
    rocket_params,
    controller_params,
    trajectory_params,
    estimated_states,
):
    """
    Function to implement the MPC trajectory generator.
    """

    print("MPC trajectory generator is running...")

    # rocket and environment parameters
    g = env_params["g"]
    m = rocket_params["m"]
    l_tvc = rocket_params["l_tvc"]
    C = rocket_params["C"]

    # controller parameters
    dt = controller_params["dt"]  # is equal to the sampling time of the controller
    T = trajectory_params["t"][-1]  # total time
    N = int(T / dt)  # number of time steps
    u_bounds = controller_params["u_bounds"]
    gamma_bounds = controller_params["gamma_bounds"]
    thrust_bounds = controller_params["thrust_bounds"]
    delta_tvc_bounds = controller_params["delta_tvc_bounds"]

    # state parameters
    x0_val = controller_params["x0"]

    Q = controller_params["Q"]
    Qf = controller_params["Qf"]
    R = controller_params["R"]
    Q = Q
    Qf = Qf
    R = R

    # initializing the state varibles
    x = ca.SX.sym("x")  # states
    x_dot = ca.SX.sym("x_dot")  # states
    y = ca.SX.sym("y")  # states
    y_dot = ca.SX.sym("y_dot")  # states
    e1bx = ca.SX.sym("e1bx")  # states
    e1by = ca.SX.sym("e1by")  # states
    e2bx = ca.SX.sym("e2bx")  # states
    e2by = ca.SX.sym("e2by")  # states
    e1tx = ca.SX.sym("e1tx")  # states
    e1ty = ca.SX.sym("e1ty")  # states
    e2tx = ca.SX.sym("e2tx")  # states
    e2ty = ca.SX.sym("e2ty")  # states
    omega_z = ca.SX.sym("omega")  # states
    thrust = ca.SX.sym("thrust")  # states

    # create a vector with variable names
    variables = ca.vertcat(
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
    )

    # initializing the control varibles
    thrust_dot = ca.SX.sym("thrust_dot")  # controls
    delta_tvc_dot = ca.SX.sym("delta_tvc_dot")  # controls

    # create a vector with variable names
    u = ca.vertcat(thrust_dot, delta_tvc_dot)

    # define the nonlinear ode
    ode = ca.vertcat(
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
        thrust * l_tvc * (e1tx * e2bx + e1ty * e2by) / C,  # omega
        thrust_dot,  # thrust
    )

    # creating the ploblem statement function
    f = ca.Function(
        "f",
        [variables, u],
        [ode],
        [
            "[x, x_dot, y, y_dot, e1bx, e1by, e2bx, e2by, e1tx, e1ty, e2tx, e2ty, omega, thrust]",
            "[thrust, delta_tvc_dot]",
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
            "[x, x_dot, y, y_dot, e1bx, e1by, e2bx, e2by, e1tx, e1ty, e2tx, e2ty, omega, thrust]",
            "[thrust, delta_tvc_dot]",
        ],
        ["x_next"],
    )

    # building the optimal control problem
    opti = ca.Opti()

    # create the state, control and initial state varibles
    x_var = opti.variable(14, N + 1)
    u_var = opti.variable(2, N)
    x_target_var = opti.parameter(14, N + 1)

    # define the cost function
    obj = 0
    for k in range(N):
        # position and velocity tracking error
        tracking_error = x_var[0:4, k] - x_target_var[0:4, k]
        obj += ca.mtimes([tracking_error.T, Q[0:4, 0:4], tracking_error])

        # rotation error
        er = (
            0.5 * x_var[4, k] * x_target_var[6, k]
            + 0.5 * x_var[5, k] * x_target_var[7, k]
            - 0.5 * x_target_var[4, k] * x_var[6, k]
            - 0.5 * x_target_var[5, k] * x_var[7, k]
        )

        obj += ca.mtimes([er, Q[4:5, 4:5], er])

        # angular velocity error
        obj += ca.mtimes([x_var[12, k], Q[5, 5], x_var[12, k]])

        # thrust error (almost 0, accounted on the control effort)
        obj += ca.mtimes([x_var[13, k], Q[6, 6], x_var[13, k]])

        # control effort
        obj += ca.mtimes([u_var[:, k].T, R, u_var[:, k]])

    # position and velocity tracking error
    tracking_error = x_var[0:4, N] - x_target_var[0:4, N]
    obj += ca.mtimes([tracking_error.T, Q[0:4, 0:4], tracking_error])

    # rotation error
    er = (
        0.5 * x_var[4, k] * x_target_var[6, k]
        + 0.5 * x_var[5, k] * x_target_var[7, k]
        - 0.5 * x_target_var[4, k] * x_var[6, k]
        - 0.5 * x_target_var[5, k] * x_var[7, k]
    )
    obj += ca.mtimes([er, Q[4:5, 4:5], er])

    # angular velocity error
    obj += ca.mtimes([x_var[12, k], Q[5, 5], x_var[12, k]])

    # thrust error (almost 0, accounted on the control effort)
    obj += ca.mtimes([x_var[13, k], Q[6, 6], x_var[13, k]])

    opti.minimize(obj)

    # apply the constraint that the state is defined by my system
    # set the initial position
    opti.subject_to(x_var[:, 0] == x0_val)

    # set the dynamics
    F = F
    for k in range(0, N + 1):
        if k < N:
            # apply the dynamics as constraints
            opti.subject_to(x_var[:, k + 1] == F(x_var[:, k], u_var[:, k]))

            # apply bounds to the inputs
            # opti.subject_to(u_var[0, k] >= u_bounds[0][0])
            # opti.subject_to(u_var[0, k] <= u_bounds[0][1])
            # opti.subject_to(u_var[1, k] >= u_bounds[1][0])
            # opti.subject_to(u_var[1, k] <= u_bounds[1][1])

        # # apply bounds to the states
        # gamma = x_var[4, k]
        # opti.subject_to(x_var[4, k] >= gamma_bounds[0])
        # opti.subject_to(x_var[4, k] <= gamma_bounds[1])

        # thrust = x_var[6, k]
        # opti.subject_to(x_var[13, k] >= thrust_bounds[0])
        # opti.subject_to(x_var[13, k] <= thrust_bounds[1])

        # delta_tvc = x_var[7, k]
        # opti.subject_to(x_var[7, k] >= delta_tvc_bounds[0])
        # opti.subject_to(x_var[7, k] <= delta_tvc_bounds[1])

    # set target
    for k in range(len(trajectory_params["x"])):
        x_targ = trajectory_params["x"][k]
        y_targ = trajectory_params["y"][k]
        vx_targ = trajectory_params["vx"][k]
        vy_targ = trajectory_params["vy"][k]
        e1bx_targ = trajectory_params["e1bx"][k]
        e1by_targ = trajectory_params["e1by"][k]
        e2bx_targ = trajectory_params["e2bx"][k]
        e2by_targ = trajectory_params["e2by"][k]
        omega_targ = trajectory_params["gamma_dot"][k]
        thrust_targ = m * g

        current_target = ca.vertcat(
            x_targ,
            vx_targ,
            y_targ,
            vy_targ,
            e1bx_targ,
            e1by_targ,
            e2bx_targ,
            e2by_targ,
            e1bx_targ,  # TVC alligned with the rocket axis
            e1by_targ,  # TVC alligned with the rocket axis
            e2bx_targ,  # TVC alligned with the rocket axis
            e2by_targ,  # TVC alligned with the rocket axis
            omega_targ,
            thrust_targ,
        )

        opti.set_value(x_target_var[:, k], current_target)
        # opti.set_initial(x, current_target)
        # opti.set_initial(u, last_sol.value(u))

    # select the desired solver
    opti.solver("ipopt")

    # print("x =", x)
    # print("x_est =", estimated_states["x"])

    # initial guess
    for i in range(N + 1):
        # print("i", i, "of", N+1, "initial guess")

        x_est = ca.vertcat(
            estimated_states["x"][i],
            estimated_states["vx"][i],
            estimated_states["y"][i],
            estimated_states["vy"][i],
            estimated_states["e1bx"][i],
            estimated_states["e1by"][i],
            estimated_states["e2bx"][i],
            estimated_states["e2by"][i],
            estimated_states["e1tx"][i],
            estimated_states["e1ty"][i],
            estimated_states["e2tx"][i],
            estimated_states["e2ty"][i],
            estimated_states["gamma_dot"][i],
            estimated_states["thrust"][i],
        )

        opti.set_initial(x_var[:, i], x_est)
        if i < N:
            u_est = ca.vertcat(
                estimated_states["delta_tvc_dot"][i],
                estimated_states["thrust_dot"][i],
            )
            opti.set_initial(u_var[:, i], u_est)

    # solve the optimal control problem
    sol = opti.solve()

    # retrieve the solution
    u = sol.value(u_var)
    horizon = sol.value(x_var)

    new_trajectory_params = {
        "t": trajectory_params["t"],
        "x": horizon[0, :],
        "vx": horizon[1, :],
        "ax": get_derivative(trajectory_params["t"], horizon[0, :]),
        "y": horizon[2, :],
        "vy": horizon[3, :],
        "ay": get_derivative(trajectory_params["t"], horizon[2, :]),
        "z": trajectory_params["z"],
        "vz": trajectory_params["vz"],
        "az": trajectory_params["az"],
        "e1bx": horizon[4, :],
        "e1by": horizon[5, :],
        "e2bx": horizon[6, :],
        "e2by": horizon[7, :],
        "gamma_dot": horizon[11, :],
        "gamma_dot_dot": get_derivative(trajectory_params["t"], horizon[11, :]),
        "thrust": horizon[12, :],
    }

    print("Trajectory optimization done!")
    return new_trajectory_params
