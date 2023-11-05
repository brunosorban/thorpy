import casadi as ca
import numpy as np
import sys
import os
from copy import deepcopy

# Get the parent folder path
parent_folder = os.path.dirname(os.path.abspath(__file__))

# Add the parent folder to sys.path
sys.path.append(os.path.join(parent_folder, ".."))

from trajectory_planning.auxiliar_codes.unc_pol_interpolation import *
from trajectory_planning.auxiliar_codes.pol_processor import *
from trajectory_planning.auxiliar_codes.compute_f1f2f3 import *
from trajectory_planning.auxiliar_codes.coeffs2derivatives import *


def coupled_pol_interpolation(states, rocket_params, controller_params, env_params, num_intervals=100):
    """
    This function interpolates the polynomials between the points. X, Y and Z directions are coupled through the control
    input constraints. There is a given position, velocity and acceleration at the initial and final points, but the
    intermediate points are free since the derivatives up to crackle are continuous and control input constraints are
    enforced. The order of the polynomials is 12 and the otimization is minimizing the jerk.

    Args:
        states (dict): Dictionary containing the desired states. The path points shall be in x, y, z lists, and the time
            points shall be in t list. The lists shall have the same length and the time points shall be equally spaced.
            It is assumed that y is the vertical direction. (To be updated)
        rocket_params (dict): Dictionary containing the rocket parameters. The used parameters are:
            - m (float): mass of the rocket [kg]
            - J_z (float): moment of inertia of the rocket [kg*m^2]
            - l_tvc (float): distance from the rocket's center of mass to the TVC's pivot point [m]
        controller_params (dict): Dictionary containing the controller parameters. The used parameters are:
            - thrust_bounds (list): list containing the lower and upper bounds of the thrust [N]
            - delta_tvc_bounds (list): list containing the lower and upper bounds of the TVC deflection angle [rad]
            - thrust_dot_bounds (list): list containing the lower and upper bounds of the thrust derivative [N/s]
            - delta_tvc_dot_bounds (list): list containing the lower and upper bounds of the TVC angular rate [rad/s]
        env_params (dict): Dictionary containing the environment parameters. The used parameters are:
            - g (float): gravitational acceleration [m/s^2]
        num_intervals (int): Number of points per polynomial where the cost function will be evaluated. The higher the
            number of points, the higher the accuracy of the interpolation, but the higher the computational cost.

    Returns:
        pol_coeffs_x (np.ndarray): 2-D array containing the coefficients of the polynomials for x direction. Each column is a polynomial.
        pol_coeffs_y (np.ndarray): 2-D array containing the coefficients of the polynomials for y direction. Each column is a polynomial.
        pol_coeffs_z (np.ndarray): 2-D array containing the coefficients of the polynomials for z direction. Each column is a polynomial.
        time_points (np.ndarray): List containing the time points of the trajectory.
    """

    #######################################
    ############ retrieve data ############
    #######################################
    time_points = states["t"]
    x_points = states["x"]
    y_points = states["y"]
    z_points = states["z"]

    # check if the time points and the position points have the same length
    if len(time_points) != len(x_points) or len(time_points) != len(y_points) or len(time_points) != len(z_points):
        raise ValueError("The time points and the position points shall have the same length.")

    params = deepcopy(rocket_params)
    params["g"] = env_params["g"]

    pol_order = 12  # order of the polynom (pol_order+1 coefficients)
    num_intervals = int(num_intervals)  # number of points per polynomial where the cost function will be evaluated

    safety_factor_num_int = controller_params["safety_factor_num_int"]
    #######################################
    ###### build auxiliar fuctions ########
    #######################################

    # calculate the polynom
    p0 = ca.MX.sym("p0")
    p1 = ca.MX.sym("p1")
    p2 = ca.MX.sym("p2")
    p3 = ca.MX.sym("p3")
    p4 = ca.MX.sym("p4")
    p5 = ca.MX.sym("p5")
    p6 = ca.MX.sym("p6")
    p7 = ca.MX.sym("p7")
    p8 = ca.MX.sym("p8")
    p9 = ca.MX.sym("p9")
    p10 = ca.MX.sym("p10")
    p11 = ca.MX.sym("p11")
    p12 = ca.MX.sym("p12")
    t = ca.MX.sym("t")

    coefs = ca.vertcat(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12)
    pos = get_pos(coefs, t)
    vel = get_vel(coefs, t)
    acc = get_acc(coefs, t)
    jerk = get_jerk(coefs, t)
    snap = get_snap(coefs, t)
    crackle = get_crackle(coefs, t)

    F_pos = ca.Function("F_pos", [coefs, t], [pos], ["[p0:p3]", "[t]"], ["position"])
    F_vel = ca.Function("F_vel", [coefs, t], [vel], ["[p0:p3]", "[t]"], ["velocity"])
    F_acc = ca.Function("F_acc", [coefs, t], [acc], ["[p0:p3]", "[t]"], ["acceleration"])
    F_jerk = ca.Function("F_jerk", [coefs, t], [jerk], ["[p0:p3]", "[t]"], ["jerk"])
    F_snap = ca.Function("F_snap", [coefs, t], [snap], ["[p0:p3]", "[t]"], ["snap"])
    F_crackle = ca.Function("F_crackle", [coefs, t], [crackle], ["[p0:p3]", "[t]"], ["crackle"])

    #######################################
    ##### interpolate the polinomial ######
    #######################################
    number_of_points = len(time_points)

    # initialize the optimization problem
    opti = ca.Opti()
    pol_coeffs_x = opti.variable(pol_order + 1, number_of_points - 1)
    pol_coeffs_y = opti.variable(pol_order + 1, number_of_points - 1)
    pol_coeffs_z = opti.variable(pol_order + 1, number_of_points - 1)

    ############## Treat edges ############
    # add postion constraints
    opti.subject_to(F_pos(pol_coeffs_x[:, 0], 0) == x_points[0])
    opti.subject_to(F_pos(pol_coeffs_x[:, -1], 1) == x_points[-1])

    opti.subject_to(F_pos(pol_coeffs_y[:, 0], 0) == y_points[0])
    opti.subject_to(F_pos(pol_coeffs_y[:, -1], 1) == y_points[-1])

    opti.subject_to(F_pos(pol_coeffs_z[:, 0], 0) == z_points[0])
    opti.subject_to(F_pos(pol_coeffs_z[:, -1], 1) == z_points[-1])

    # add velocity constraints (initial and final velocity are zero)
    opti.subject_to(F_vel(pol_coeffs_x[:, 0], 0) == 0)
    opti.subject_to(F_vel(pol_coeffs_x[:, -1], 1) == 0)

    opti.subject_to(F_vel(pol_coeffs_y[:, 0], 0) == 0)
    opti.subject_to(F_vel(pol_coeffs_y[:, -1], 1) == 0)

    opti.subject_to(F_vel(pol_coeffs_z[:, 0], 0) == 0)
    opti.subject_to(F_vel(pol_coeffs_z[:, -1], 1) == 0)

    # add acceleration constraints (initial and final acceleration are zero)
    opti.subject_to(F_acc(pol_coeffs_x[:, 0], 0) == 0)
    opti.subject_to(F_acc(pol_coeffs_x[:, -1], 1) == 0)

    opti.subject_to(F_acc(pol_coeffs_y[:, 0], 0) == 0)
    opti.subject_to(F_acc(pol_coeffs_y[:, -1], 1) == 0)

    opti.subject_to(F_acc(pol_coeffs_z[:, 0], 0) == 0)
    opti.subject_to(F_acc(pol_coeffs_z[:, -1], 1) == 0)

    # add jerk constraints - gamma_dot is zero in the beginning and end
    opti.subject_to(F_jerk(pol_coeffs_x[:, 0], 0) == 0)
    opti.subject_to(F_jerk(pol_coeffs_x[:, -1], 1) == 0)

    opti.subject_to(F_jerk(pol_coeffs_y[:, 0], 0) == 0)
    opti.subject_to(F_jerk(pol_coeffs_y[:, -1], 1) == 0)

    opti.subject_to(F_jerk(pol_coeffs_z[:, 0], 0) == 0)
    opti.subject_to(F_jerk(pol_coeffs_z[:, -1], 1) == 0)

    # add snap constraints - gamma_dot_dot is zero in the beginning and end, thus,
    # tvc_angle is zero in the beginning and end
    opti.subject_to(F_snap(pol_coeffs_x[:, 0], 0) == 0)
    opti.subject_to(F_snap(pol_coeffs_x[:, -1], 1) == 0)

    opti.subject_to(F_snap(pol_coeffs_y[:, 0], 0) == 0)
    opti.subject_to(F_snap(pol_coeffs_y[:, -1], 1) == 0)

    opti.subject_to(F_snap(pol_coeffs_z[:, 0], 0) == 0)
    opti.subject_to(F_snap(pol_coeffs_z[:, -1], 1) == 0)

    # add crackle constraints - tvc_angle derivative is zero in the beginning and end
    opti.subject_to(F_crackle(pol_coeffs_x[:, 0], 0) == 0)
    opti.subject_to(F_crackle(pol_coeffs_x[:, -1], 1) == 0)

    opti.subject_to(F_crackle(pol_coeffs_y[:, 0], 0) == 0)
    opti.subject_to(F_crackle(pol_coeffs_y[:, -1], 1) == 0)

    opti.subject_to(F_crackle(pol_coeffs_z[:, 0], 0) == 0)
    opti.subject_to(F_crackle(pol_coeffs_z[:, -1], 1) == 0)

    ############## Treat middle ############
    for i in range(1, number_of_points - 1):
        # add postion constraints
        opti.subject_to(F_pos(pol_coeffs_x[:, i], 0) == x_points[i])
        opti.subject_to(F_pos(pol_coeffs_x[:, i - 1], 1) == x_points[i])

        opti.subject_to(F_pos(pol_coeffs_y[:, i], 0) == y_points[i])
        opti.subject_to(F_pos(pol_coeffs_y[:, i - 1], 1) == y_points[i])

        opti.subject_to(F_pos(pol_coeffs_z[:, i], 0) == z_points[i])
        opti.subject_to(F_pos(pol_coeffs_z[:, i - 1], 1) == z_points[i])

        # add velocity constraints
        opti.subject_to(F_vel(pol_coeffs_x[:, i - 1], 1) == F_vel(pol_coeffs_x[:, i], 0))
        opti.subject_to(F_vel(pol_coeffs_y[:, i - 1], 1) == F_vel(pol_coeffs_y[:, i], 0))
        opti.subject_to(F_vel(pol_coeffs_z[:, i - 1], 1) == F_vel(pol_coeffs_z[:, i], 0))

        # add acceleration constraints
        opti.subject_to(F_acc(pol_coeffs_x[:, i - 1], 1) == F_acc(pol_coeffs_x[:, i], 0))
        opti.subject_to(F_acc(pol_coeffs_y[:, i - 1], 1) == F_acc(pol_coeffs_y[:, i], 0))
        opti.subject_to(F_acc(pol_coeffs_z[:, i - 1], 1) == F_acc(pol_coeffs_z[:, i], 0))

        # add jerk constraints
        opti.subject_to(F_jerk(pol_coeffs_x[:, i - 1], 1) == F_jerk(pol_coeffs_x[:, i], 0))
        opti.subject_to(F_jerk(pol_coeffs_y[:, i - 1], 1) == F_jerk(pol_coeffs_y[:, i], 0))
        opti.subject_to(F_jerk(pol_coeffs_z[:, i - 1], 1) == F_jerk(pol_coeffs_z[:, i], 0))

        # add snap constraints
        opti.subject_to(F_snap(pol_coeffs_x[:, i - 1], 1) == F_snap(pol_coeffs_x[:, i], 0))
        opti.subject_to(F_snap(pol_coeffs_y[:, i - 1], 1) == F_snap(pol_coeffs_y[:, i], 0))
        opti.subject_to(F_snap(pol_coeffs_z[:, i - 1], 1) == F_snap(pol_coeffs_z[:, i], 0))

        # add crackle constraints
        opti.subject_to(F_crackle(pol_coeffs_x[:, i - 1], 1) == F_crackle(pol_coeffs_x[:, i], 0))
        opti.subject_to(F_crackle(pol_coeffs_y[:, i - 1], 1) == F_crackle(pol_coeffs_y[:, i], 0))
        opti.subject_to(F_crackle(pol_coeffs_z[:, i - 1], 1) == F_crackle(pol_coeffs_z[:, i], 0))

    # define cost function
    obj = 0

    for i in range(number_of_points - 1):  # for each polinomial
        dt_interval = time_points[i + 1] - time_points[i]
        dt = 1 / num_intervals

        for j in range(num_intervals):
            # update the cost function
            obj += F_jerk(pol_coeffs_x[:, i], j * dt) ** 2
            obj += F_jerk(pol_coeffs_y[:, i], j * dt) ** 2
            obj += F_jerk(pol_coeffs_z[:, i], j * dt) ** 2

            vel = ca.vertcat(
                F_vel(pol_coeffs_x[:, i], j * dt) / dt_interval,
                F_vel(pol_coeffs_y[:, i], j * dt) / dt_interval,
                F_vel(pol_coeffs_z[:, i], j * dt) / dt_interval,
            )

            acc = ca.vertcat(
                F_acc(pol_coeffs_x[:, i], j * dt) / dt_interval**2,
                F_acc(pol_coeffs_y[:, i], j * dt) / dt_interval**2,
                F_acc(pol_coeffs_z[:, i], j * dt) / dt_interval**2,
            )

            jerk = ca.vertcat(
                F_jerk(pol_coeffs_x[:, i], j * dt) / dt_interval**3,
                F_jerk(pol_coeffs_y[:, i], j * dt) / dt_interval**3,
                F_jerk(pol_coeffs_z[:, i], j * dt) / dt_interval**3,
            )

            snap = ca.vertcat(
                F_snap(pol_coeffs_x[:, i], j * dt) / dt_interval**4,
                F_snap(pol_coeffs_y[:, i], j * dt) / dt_interval**4,
                F_snap(pol_coeffs_z[:, i], j * dt) / dt_interval**4,
            )

            crackle = ca.vertcat(
                F_crackle(pol_coeffs_x[:, i], j * dt) / dt_interval**5,
                F_crackle(pol_coeffs_y[:, i], j * dt) / dt_interval**5,
                F_crackle(pol_coeffs_z[:, i], j * dt) / dt_interval**5,
            )

            f1, f2, f3, f1_dot, f2_dot, f3_dot = compute_f1f2f3(
                acc[0],
                acc[1],
                acc[2],
                jerk[0],
                jerk[1],
                jerk[2],
                snap[0],
                snap[1],
                snap[2],
                crackle[0],
                crackle[1],
                crackle[2],
                params,
            )

            f_squared = f1**2 + f2**2 + f3**2

            # it is assumed that f1 > 0 - which holds because the rocket has no reverse thrust
            opti.subject_to(
                f1**2 + f2**2
                <= (f3 * ca.tan(controller_params["delta_tvc_bounds"][1]) / safety_factor_num_int) ** 2
            )

            opti.subject_to(
                f_squared >= (controller_params["thrust_bounds"][0] * safety_factor_num_int) ** 2
            )  # in this case is "*" because the thrust is always positive
            opti.subject_to(f_squared <= (controller_params["thrust_bounds"][1] / safety_factor_num_int) ** 2)

            opti.subject_to(f3_dot >= controller_params["thrust_dot_bounds"][0] / safety_factor_num_int)

            opti.subject_to(f3_dot <= controller_params["thrust_dot_bounds"][1] / safety_factor_num_int)

            opti.subject_to(
                f1_dot**2 + f2_dot**2
                <= f_squared * (controller_params["delta_tvc_dot_bounds"][1] / safety_factor_num_int) ** 2
            )

    # add final cost
    obj += F_jerk(pol_coeffs_x[:, i], 1) ** 2
    obj += F_jerk(pol_coeffs_y[:, i], 1) ** 2
    obj += F_jerk(pol_coeffs_z[:, i], 1) ** 2

    # define the optimization problem
    opti.minimize(obj)

    # initialize the polinomial coefficients
    states_x = deepcopy(states)
    states_x["pos"] = states["x"]
    initial_guess_x, _ = unconstrained_pol_interpolation(states_x)

    states_y = deepcopy(states)
    states_y["pos"] = states["y"]
    initial_guess_y, _ = unconstrained_pol_interpolation(states_y)

    states_z = deepcopy(states)
    states_z["pos"] = states["z"]
    initial_guess_z, _ = unconstrained_pol_interpolation(states_z)

    opti.set_initial(pol_coeffs_x, initial_guess_x)
    opti.set_initial(pol_coeffs_y, initial_guess_y)
    opti.set_initial(pol_coeffs_z, initial_guess_z)

    # configure the solver
    ipopt_options = {
        "verbose": False,
        "ipopt.tol": 1e-6,
        "ipopt.acceptable_tol": 1e-6,
        "ipopt.max_iter": 500,
        "ipopt.warm_start_init_point": "yes",
        "ipopt.print_level": 1,
        "print_time": False,
    }

    # select the desired solver
    opti.solver("ipopt", ipopt_options)

    print("Interpolating trajectory...")
    sol = opti.solve()
    print("Interpolation done!")

    if number_of_points == 2:  # tweak for the 2 points case so that the output is still a 2D array
        return (
            np.array([sol.value(pol_coeffs_x)]).T,
            np.array([sol.value(pol_coeffs_y)]).T,
            np.array([sol.value(pol_coeffs_z)]).T,
            np.array(time_points),
        )
    else:
        return (
            np.array(sol.value(pol_coeffs_x)),
            np.array(sol.value(pol_coeffs_y)),
            np.array(sol.value(pol_coeffs_z)),
            np.array(time_points),
        )
