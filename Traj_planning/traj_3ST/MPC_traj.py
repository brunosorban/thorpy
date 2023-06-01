import casadi as ca
import sys
sys.path.append(r"D:\Code\SA2023\rocket-control-framework")
from RK4 import RK4

def MPC_traj(
    env_params,
    rocket_params,
    controller_params,
    trajectory_params,
):
    """
    Function to implement the MPC trajectory generator.
    """

    # rocket and environment parameters
    g = env_params["g"]
    m = rocket_params["m"]
    l_tvc = rocket_params["l_tvc"]

    # controller parameters
    dt = controller_params["dt"] # is equal to the sampling time of the controller
    T = trajectory_params["t"][-1] # total time
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
    omega = ca.SX.sym("omega")  # states
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
        omega,
        thrust,
    )

    # initializing the control varibles
    thrust_dot = ca.SX.sym("thrust_dot")  # controls
    omega_control = ca.SX.sym("omega_control")  # controls

    # create a vector with variable names
    u = ca.vertcat(thrust_dot, omega_control)

    # define the nonlinear ode
    ode = ca.vertcat(
        x_dot,  # x
        thrust / m * e1tx,  # v_x
        y_dot,  # y
        thrust / m * e1ty - g,  # v_y
        e2bx * omega,  # e1bx
        -e1bx * omega,  # e1by
        e2by * omega,  # e2bx
        -e1by * omega,  # e2by
        e2tx * (omega_control + omega),  # e1tx
        -e1tx * (omega_control + omega),  # e1ty
        e2ty * (omega_control + omega),  # e2tx
        -e1ty * (omega_control + omega),  # e2ty
        thrust * l_tvc * (e1tx * e2bx + e1ty * e2by),  # omega
        thrust_dot,  # thrust
    )

    # creating the ploblem statement function
    f = ca.Function(
        "f",
        [variables, u],
        [ode],
        [
            "[x, x_dot, y, y_dot, e1bx, e1by, e2bx, e2by, e1tx, e1ty, e2tx, e2ty, omega, thrust]",
            "[thrust, omega_control]",
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
            "[thrust, omega_control]",
        ],
        ["x_next"],
    )

    # building the optimal control problem
    opti = ca.Opti()

    # create the state, control and initial state varibles
    x = opti.variable(14, N + 1)
    u = opti.variable(2, N)
    x_target = opti.parameter(14, N + 1)

    # define the cost function
    obj = 0
    for k in range(N):
        # position and velocity tracking error
        tracking_error = x[0:4, k] - x_target[0:4, k]
        obj += ca.mtimes([tracking_error.T, Q[0:4, 0:4], tracking_error])

        # # rotation error
        er = (
            0.5 * x[4, k] * x_target[6, k]
            + 0.5 * x[5, k] * x_target[7, k]
            - 0.5 * x_target[4, k] * x[6, k]
            - 0.5 * x_target[5, k] * x[7, k]
        )

        obj += ca.mtimes([er, Q[4:5, 4:5], er])

        # angular velocity error
        obj += ca.mtimes([x[12, k], Q[5, 5], x[12, k]])

        # thrust error (almost 0, accounted on the control effort)
        obj += ca.mtimes([x[13, k], Q[6, 6], x[13, k]])

        # control effort
        obj += ca.mtimes([u[:, k].T, R, u[:, k]])

    # position and velocity tracking error
    tracking_error = x[0:4, N] - x_target[0:4, N]
    obj += ca.mtimes([tracking_error.T, Q[0:4, 0:4], tracking_error])

    # rotation error
    er = (
        0.5 * x[4, k] * x_target[6, k]
        + 0.5 * x[5, k] * x_target[7, k]
        - 0.5 * x_target[4, k] * x[6, k]
        - 0.5 * x_target[5, k] * x[7, k]
    )
    obj += ca.mtimes([er, Q[4:5, 4:5], er])

    # angular velocity error
    obj += ca.mtimes([x[12, k], Q[5, 5], x[12, k]])

    # thrust error (almost 0, accounted on the control effort)
    obj += ca.mtimes([x[13, k], Q[6, 6], x[13, k]])

    opti.minimize(obj)

    # apply the constraint that the state is defined by my system
    # set the initial position
    opti.subject_to(x[:, 0] == x0_val)

    # set the dynamics
    F = F
    for k in range(0, N + 1):
        if k < N:
            # apply the dynamics as constraints
            opti.subject_to(x[:, k + 1] == F(x[:, k], u[:, k]))

            # apply bounds to the inputs
            # opti.subject_to(u[0, k] >= u_bounds[0][0])
            # opti.subject_to(u[0, k] <= u_bounds[0][1])
            # opti.subject_to(u[1, k] >= u_bounds[1][0])
            # opti.subject_to(u[1, k] <= u_bounds[1][1])

        # # apply bounds to the states
        # gamma = x[4, k]
        # opti.subject_to(x[4, k] >= gamma_bounds[0])
        # opti.subject_to(x[4, k] <= gamma_bounds[1])

        # thrust = x[6, k]
        # opti.subject_to(x[13, k] >= thrust_bounds[0])
        # opti.subject_to(x[13, k] <= thrust_bounds[1])

        # delta_tvc = x[7, k]
        # opti.subject_to(x[7, k] >= delta_tvc_bounds[0])
        # opti.subject_to(x[7, k] <= delta_tvc_bounds[1])

    # set target
    for k in range(len(trajectory_params["x"])):
        x = trajectory_params["x"][k]
        y = trajectory_params["y"][k]
        vx = trajectory_params["vx"][k]
        vy = trajectory_params["vy"][k]
        e1bx = trajectory_params["e1bx"][k]
        e1by = trajectory_params["e1by"][k]
        e2bx = trajectory_params["e2bx"][k]
        e2by = trajectory_params["e2by"][k]
        omega = trajectory_params["gamma_dot"][k]
        thrust = m * g
        
        current_target = ca.vertcat(
            x,
            vx,
            y,
            vy,
            e1bx,
            e1by,
            e2bx,
            e2by,
            e1bx, # TVC alligned with the rocket axis
            e1by, # TVC alligned with the rocket axis
            e2bx, # TVC alligned with the rocket axis
            e2by, # TVC alligned with the rocket axis
            omega,
            thrust,
        )
        
        opti.set_value(x_target[:, k], current_target)
        # opti.set_initial(x, current_target)
        # opti.set_initial(u, last_sol.value(u))

    # select the desired solver
    opti.solver("ipopt")
    
    # solve the optimal control problem
    sol = opti.solve()
    
    # retrieve the solution
    u = sol.value(u)
    horizon = sol.value(x)    

    new_trajectory_params = {
        "t": trajectory_params["t"],
        "x": horizon[0, :],
        "vx": horizon[1, :],
        "y": horizon[2, :],
        "vy": horizon[3, :],
        "e1bx": horizon[4, :],
        "e1by": horizon[5, :],
        "e2bx": horizon[6, :],
        "e2by": horizon[7, :],
        "gamma_dot": horizon[11, :],
        "thrust": horizon[12, :],
    }
        
    return new_trajectory_params    