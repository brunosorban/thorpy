import casadi as ca
import sys
import os


# Get the parent folder path
parent_folder = os.path.dirname(os.path.abspath(__file__))

# Add the parent folder to sys.path
sys.path.append(os.path.join(parent_folder, ".."))

from Traj_planning.auxiliar_codes.coeffs2derivatives import *
from Traj_planning.auxiliar_codes.pol_processor import *
from Traj_planning.unc_pol_interpolation import *


def pol_interpolation(states, contraints):
    """
    This function interpolates the polynomials between the points. There is a given position; velocity and acceleration
    at the initial and final points are 0, but the intermediate points are free since the derivatives up to crackle are
    continuous and constraints on maximum velocity and acceleration are enforced. The order of the polynomials is 12 and
    the otimization is minimizing the jerk.

    Args:
        states (dict): Dictionary containing the desired states. The path points shall be in pos list, and the time
            points shall be in t list. The lists shall have the same length and the time points shall be equally spaced.
        contraints (dict): Dictionary containing the contraints. The maximum and minimum velocity shall be in max_v and
            min_v, respectively. The maximum and minimum acceleration shall be in max_a and min_a, respectively.
        num_intervals (int): Number of points per polynomial where the cost function will be evaluated. The higher the
            number of points, the higher the accuracy of the interpolation, but the higher the computational cost.

    Returns:
        pol_coeffs (np.ndarray): 2-D array containing the coefficients of the polynomials. Each column is a polynomial.
        time_points (np.ndarray): List containing the time points of the trajectory.
    """

    #######################################
    ############ retrieve data ############
    #######################################
    time_points = states["t"]
    position_points = states["pos"]

    max_v = contraints["max_v"]
    min_v = contraints["min_v"]
    max_a = contraints["max_a"]
    min_a = contraints["min_a"]

    pol_order = 12  # order of the polynom (pol_order+1 coefficients)

    #######################################
    ###### build auxiliar fuctions ########
    #######################################

    # calculate the polynom
    p0 = ca.SX.sym("p0")
    p1 = ca.SX.sym("p1")
    p2 = ca.SX.sym("p2")
    p3 = ca.SX.sym("p3")
    p4 = ca.SX.sym("p4")
    p5 = ca.SX.sym("p5")
    p6 = ca.SX.sym("p6")
    p7 = ca.SX.sym("p7")
    p8 = ca.SX.sym("p8")
    p9 = ca.SX.sym("p9")
    p10 = ca.SX.sym("p10")
    p11 = ca.SX.sym("p11")
    p12 = ca.SX.sym("p12")
    t = ca.SX.sym("t")

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
    pol_coeffs = opti.variable(pol_order + 1, number_of_points - 1)

    ############## Treat edges ############
    # add postion constraints
    opti.subject_to(F_pos(pol_coeffs[:, 0], 0) == position_points[0])
    opti.subject_to(F_pos(pol_coeffs[:, -1], 1) == position_points[-1])

    # add velocity constraints (initial and final velocity are zero)
    opti.subject_to(F_vel(pol_coeffs[:, 0], 0) == 0)
    opti.subject_to(F_vel(pol_coeffs[:, -1], 1) == 0)

    # add acceleration constraints (initial and final acceleration are zero)
    opti.subject_to(F_acc(pol_coeffs[:, 0], 0) == 0)
    opti.subject_to(F_acc(pol_coeffs[:, -1], 1) == 0)

    # add jerk constraints - gamma_dot is zero in the beginning and end
    opti.subject_to(F_jerk(pol_coeffs[:, 0], 0) == 0)
    opti.subject_to(F_jerk(pol_coeffs[:, -1], 1) == 0)

    # add snap constraints - gamma_dot_dot is zero in the beginning and end, thus,
    # tvc_angle is zero in the beginning and end
    opti.subject_to(F_snap(pol_coeffs[:, 0], 0) == 0)
    opti.subject_to(F_snap(pol_coeffs[:, -1], 1) == 0)

    # add crackle constraints - gamma_dot_dot is zero in the beginning and end, thus,
    # tvc_angle derivative is zero in the beginning and end
    opti.subject_to(F_crackle(pol_coeffs[:, 0], 0) == 0)
    opti.subject_to(F_crackle(pol_coeffs[:, -1], 1) == 0)

    ############## Treat middle ############
    for i in range(1, number_of_points - 1):
        # add postion constraints
        opti.subject_to(F_pos(pol_coeffs[:, i], 0) == position_points[i])
        opti.subject_to(F_pos(pol_coeffs[:, i - 1], 1) == position_points[i])

        # add velocity constraints
        opti.subject_to(F_vel(pol_coeffs[:, i - 1], 1) == F_vel(pol_coeffs[:, i], 0))

        # add acceleration constraints
        opti.subject_to(F_acc(pol_coeffs[:, i - 1], 1) == F_acc(pol_coeffs[:, i], 0))

        # add jerk constraints
        opti.subject_to(F_jerk(pol_coeffs[:, i - 1], 1) == F_jerk(pol_coeffs[:, i], 0))

        # add snap constraints
        opti.subject_to(F_snap(pol_coeffs[:, i - 1], 1) == F_snap(pol_coeffs[:, i], 0))

        # add crackle constraints
        opti.subject_to(F_crackle(pol_coeffs[:, i - 1], 1) == F_crackle(pol_coeffs[:, i], 0))

    # define cost function
    obj = 0

    num_intervals = 100
    for i in range(number_of_points - 1):  # for each polinomial
        dt_interval = time_points[i + 1] - time_points[i]
        dt = 1 / num_intervals

        for j in range(num_intervals):
            # update the cost function
            obj += F_jerk(pol_coeffs[:, i], j * dt) ** 2

            # add velocity constraints
            opti.subject_to(F_vel(pol_coeffs[:, i], j * dt) / dt_interval <= max_v)
            opti.subject_to(F_vel(pol_coeffs[:, i], j * dt) / dt_interval >= min_v)

            # add acceleration constraints
            opti.subject_to(F_acc(pol_coeffs[:, i], j * dt) / dt_interval**2 <= max_a)
            opti.subject_to(F_acc(pol_coeffs[:, i], j * dt) / dt_interval**2 >= min_a)

    # add final cost
    obj += F_jerk(pol_coeffs[:, i], 1) ** 2

    # define the optimization problem
    opti.minimize(obj)

    # initialize the polinomial coefficients
    initial_guess, _ = unconstrained_pol_interpolation(states)
    opti.set_initial(pol_coeffs, initial_guess)

    # configure the solver
    ipopt_options = {
        "verbose": False,
        "ipopt.tol": 1e-8,
        "ipopt.acceptable_tol": 1e-8,
        "ipopt.max_iter": 500,
        "ipopt.warm_start_init_point": "yes",
        "ipopt.print_level": 5,
        "print_time": False,
    }

    # select the desired solver
    opti.solver("ipopt", ipopt_options)

    print("Interpolating trajectory...")
    sol = opti.solve()
    print("Interpolation done!")
    print("Solution: \n", sol.value(pol_coeffs))

    return sol.value(pol_coeffs), time_points
