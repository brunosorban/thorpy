import casadi as ca
import sys
import os
from copy import deepcopy

# Get the parent folder path
parent_folder = os.path.dirname(os.path.abspath(__file__))

# Add the parent folder to sys.path
sys.path.append(os.path.join(parent_folder, ".."))

from Traj_planning.traj_3ST.auxiliar_codes.coeffs2derivatives import *
from Traj_planning.traj_3ST.auxiliar_codes.pol_processor import *
from Traj_planning.traj_3ST.unc_pol_interpolation import *
from Traj_planning.traj_3ST.auxiliar_codes.new_get_f1f2 import *


def coupled_pol_interpolation(states, contraints, rocket_params, controller_params, env_params):
    """
    This function interpolates the polynomials between the points
    """

    #######################################
    ############ retrieve data ############
    #######################################
    time_points = states["t"]
    x_points = states["x"]
    y_points = states["y"]
    z_points = states["z"]

    max_vx = contraints["max_vx"]
    max_vy = contraints["max_vy"]
    max_vz = contraints["max_vz"]

    min_vx = contraints["min_vx"]
    min_vy = contraints["min_vy"]
    min_vz = contraints["min_vz"]

    max_ax = contraints["max_ax"]
    max_ay = contraints["max_ay"]
    max_az = contraints["max_az"]

    min_ax = contraints["min_ax"]
    min_ay = contraints["min_ay"]
    min_az = contraints["min_az"]
    
    params = deepcopy(rocket_params)
    params["g"] = env_params["g"]

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

    # vel = ca.jacobian(pos, t)
    # acc = ca.jacobian(vel, t)
    # jerk = ca.jacobian(acc, t)
    # snap = ca.jacobian(jerk, t)
    # crackle = ca.jacobian(snap, t)

    F_pos = ca.Function("F_pos", [coefs, t], [pos], ["[p0:p3]", "[t]"], ["position"])
    F_vel = ca.Function("F_vel", [coefs, t], [vel], ["[p0:p3]", "[t]"], ["velocity"])
    F_acc = ca.Function(
        "F_acc", [coefs, t], [acc], ["[p0:p3]", "[t]"], ["acceleration"]
    )
    F_jerk = ca.Function("F_jerk", [coefs, t], [jerk], ["[p0:p3]", "[t]"], ["jerk"])
    F_snap = ca.Function("F_snap", [coefs, t], [snap], ["[p0:p3]", "[t]"], ["snap"])
    F_crackle = ca.Function(
        "F_crackle", [coefs, t], [crackle], ["[p0:p3]", "[t]"], ["crackle"]
    )

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

    num_intervals = 100
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
                F_vel(pol_coeffs_z[:, i], j * dt) / dt_interval
            )
            
            acc = ca.vertcat(
                F_acc(pol_coeffs_x[:, i], j * dt) / dt_interval**2,
                F_acc(pol_coeffs_y[:, i], j * dt) / dt_interval**2,
                F_acc(pol_coeffs_z[:, i], j * dt) / dt_interval**2
            )
            
            jerk = ca.vertcat(
                F_jerk(pol_coeffs_x[:, i], j * dt) / dt_interval**3,
                F_jerk(pol_coeffs_y[:, i], j * dt) / dt_interval**3,
                F_jerk(pol_coeffs_z[:, i], j * dt) / dt_interval**3
            )
            
            snap = ca.vertcat(
                F_snap(pol_coeffs_x[:, i], j * dt) / dt_interval**4,
                F_snap(pol_coeffs_y[:, i], j * dt) / dt_interval**4,
                F_snap(pol_coeffs_z[:, i], j * dt) / dt_interval**4
            )
            
            crackle = ca.vertcat(
                F_crackle(pol_coeffs_x[:, i], j * dt) / dt_interval**4,
                F_crackle(pol_coeffs_y[:, i], j * dt) / dt_interval**4,
                F_crackle(pol_coeffs_z[:, i], j * dt) / dt_interval**4
            )
            
            f1, f2 = get_f1f2(acc[0], acc[1], jerk[0], jerk[1], snap[0], snap[1], params)
            f = ca.sqrt(f1**2 + f2**2)
            
            f1_dot, f2_dot = get_f1f2_dot(acc[0], acc[1], jerk[0], jerk[1], snap[0], snap[1], crackle[0], crackle[1], params)
            f_dot = f_dot = (f1 * f1_dot + f2 * f2_dot) / f
            
            opti.subject_to(f1 >= controller_params["thrust_bounds"][0])
            opti.subject_to(f1 <= controller_params["thrust_bounds"][1])
            
            # it is assumed that f1 > 0
            opti.subject_to(f2 >= -ca.sin(controller_params["delta_tvc_bounds"][1]) * f1)
            opti.subject_to(f2 <= ca.sin(controller_params["delta_tvc_bounds"][1]) * f1)
            
            # opti.subject_to(f2 >= -ca.sin(controller_params["delta_tvc_bounds"][1]) * controller_params["thrust_bounds"][1])
            # opti.subject_to(f2 <=  ca.sin(controller_params["delta_tvc_bounds"][1]) * controller_params["thrust_bounds"][1])
            
            opti.subject_to(f >= controller_params["thrust_bounds"][0])
            opti.subject_to(f <= controller_params["thrust_bounds"][1])
            
            opti.subject_to(f1_dot >= controller_params["u_bounds"][0][0])
            opti.subject_to(f1_dot <= controller_params["u_bounds"][0][1])
            
            # # add absolute velocity constraints
            # opti.subject_to(F_vel(pol_coeffs_x[:, i], j * dt) / dt_interval <= max_vx)
            # opti.subject_to(F_vel(pol_coeffs_x[:, i], j * dt) / dt_interval >= min_vx)
            
            # opti.subject_to(F_vel(pol_coeffs_y[:, i], j * dt) / dt_interval <= max_vy)
            # opti.subject_to(F_vel(pol_coeffs_y[:, i], j * dt) / dt_interval >= min_vy)
            
            # opti.subject_to(F_vel(pol_coeffs_z[:, i], j * dt) / dt_interval <= max_vz)
            # opti.subject_to(F_vel(pol_coeffs_z[:, i], j * dt) / dt_interval >= min_vz)

            # # add absolute acceleration constraints
            # opti.subject_to(F_acc(pol_coeffs_x[:, i], j * dt) / dt_interval**2 <= max_ax)
            # opti.subject_to(F_acc(pol_coeffs_x[:, i], j * dt) / dt_interval**2 >= min_ax)

            # opti.subject_to(F_acc(pol_coeffs_y[:, i], j * dt) / dt_interval**2 <= max_ay)
            # opti.subject_to(F_acc(pol_coeffs_y[:, i], j * dt) / dt_interval**2 >= min_ay)
            
            # opti.subject_to(F_acc(pol_coeffs_z[:, i], j * dt) / dt_interval**2 <= max_az)
            # opti.subject_to(F_acc(pol_coeffs_z[:, i], j * dt) / dt_interval**2 >= min_az)

        # obj += pol_coeffs[pol_order, i]**2

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
    print("Solution: \n")
    print("\nx: \n", sol.value(pol_coeffs_x))
    print("\ny: \n", sol.value(pol_coeffs_y))
    print("\nz: \n", sol.value(pol_coeffs_z))

    return sol.value(pol_coeffs_x), sol.value(pol_coeffs_y), sol.value(pol_coeffs_z), time_points
    # return initial_guess, time_points
