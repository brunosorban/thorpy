import copy
import numpy as np
import casadi as ca
from Traj_planning.traj_3ST.auxiliar_codes.estimate_coeffs import estimate_coeffs
from Traj_planning.traj_3ST.auxiliar_codes.get_f1f2 import get_f1f2, get_f1f2_dot
from Traj_planning.traj_3ST.auxiliar_codes.coeffs2derivatives import *


def min_snap_traj(states, constraints, rocket_params, control_params):
    """This function calculates the trajectory via polynomial interpolation. It
    uses the minimum snap as cost function to be minimized, together with time.

    Args:
        states (dict): dictionary containing the states of the system x, y, z,
        vx, vy, vz, ax, ay, az and time. The states vi and ai shall be given as
        None if they are not to be specified.

        constraints (dict): dictionary containing the constraints of the system
        min_vx, max_vx, min_vy, max_vy, min_vz, max_vz, min_ax, max_ax, min_ay,

    Returns:
        Px_coeffs, Py_coeffs and Pz_coeffs: the coefficients of the polynomials for x, y and z
    """
    time_points = states["t"]

    if len(time_points) != len(states["x"]):
        raise ValueError(
            "The number of time points must be equal to the number of states"
        )

    number_of_points = len(time_points)

    if number_of_points < 2:
        raise ValueError("The number of points must be at least 2")

    pol_order = 13  # order of the polynom +1
    g = constraints["g"]

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
    # pop = get_pop(coefs, t)

    F_pos = ca.Function(
        "F_pos",
        [coefs, t],
        [pos],
        [
            "[p0, p1, p2, p3, p4, p5, p6, p7, p8]",
            "[t]",
        ],
        ["position"],
    )

    F_vel = ca.Function(
        "F_vel",
        [coefs, t],
        [vel],
        [
            "[p0, p1, p2, p3, p4, p5, p6, p7, p8]",
            "[t]",
        ],
        ["velocity"],
    )

    F_acc = ca.Function(
        "F_acc",
        [coefs, t],
        [acc],
        [
            "[p0, p1, p2, p3, p4, p5, p6, p7, p8]",
            "[t]",
        ],
        ["acceleration"],
    )

    F_jerk = ca.Function(
        "F_jerk",
        [coefs, t],
        [jerk],
        [
            "[p0, p1, p2, p3, p4, p5, p6, p7, p8]",
            "[t]",
        ],
        ["jerk"],
    )

    F_snap = ca.Function(
        "F_snap",
        [coefs, t],
        [snap],
        [
            "[p0, p1, p2, p3, p4, p5, p6, p7, p8]",
            "[t]",
        ],
        ["snap"],
    )

    F_crackle = ca.Function(
        "F_crackle",
        [coefs, t],
        [crackle],
        [
            "[p0, p1, p2, p3, p4, p5, p6, p7, p8]",
            "[t]",
        ],
        ["crackle"],
    )

    # F_pop = ca.Function(
    #     "F_pop",
    #     [coefs, t],
    #     [pop],
    #     [
    #         "[p0, p1, p2, p3, p4, p5, p6, p7, p8]",
    #         "[t]",
    #     ],
    #     ["pop"],
    # )

    params = copy.deepcopy(rocket_params)
    params["g"] = g

    # building the optimal control problem
    opti = ca.Opti()

    # create the variables with the polynom coefficients
    Px_coefs = opti.variable(pol_order, number_of_points - 1)
    Py_coefs = opti.variable(pol_order, number_of_points - 1)
    Pz_coefs = opti.variable(pol_order, number_of_points - 1)
    time = opti.variable(number_of_points)
    
    # define cost function
    obj = 0 

    # set the contraints
    print("number of points (per axis): ", number_of_points)
    print("number of polynoms (per axis): ", number_of_points - 1)
    print("Adding constraints...")
    
    # treat t=0 as a special case - only position and velocity can be set
    opti.subject_to(F_pos(Px_coefs[:, 0], time[0]) == states["x"][0])
    opti.subject_to(F_pos(Py_coefs[:, 0], time[0]) == states["y"][0])
    opti.subject_to(F_pos(Pz_coefs[:, 0], time[0]) == states["z"][0])
    
    opti.subject_to(F_vel(Px_coefs[:, 0], time[0]) == states["vx"][0])
    opti.subject_to(F_vel(Py_coefs[:, 0], time[0]) == states["vy"][0])
    opti.subject_to(F_vel(Pz_coefs[:, 0], time[0]) == states["vz"][0])
    
    opti.subject_to(F_acc(Px_coefs[:, 0], time[0]) == 0)
    opti.subject_to(F_acc(Py_coefs[:, 0], time[0]) == 0)
    opti.subject_to(F_acc(Pz_coefs[:, 0], time[0]) == 0)
    
    opti.subject_to(F_jerk(Px_coefs[:, 0], time[0]) == 0)
    opti.subject_to(F_jerk(Py_coefs[:, 0], time[0]) == 0)
    opti.subject_to(F_jerk(Pz_coefs[:, 0], time[0]) == 0)
    
    opti.subject_to(F_snap(Px_coefs[:, 0], time[0]) == 0)
    opti.subject_to(F_snap(Py_coefs[:, 0], time[0]) == 0)
    opti.subject_to(F_snap(Pz_coefs[:, 0], time[0]) == 0)
    
    opti.subject_to(F_crackle(Px_coefs[:, 0], time[0]) == 0)
    opti.subject_to(F_crackle(Py_coefs[:, 0], time[0]) == 0)
    opti.subject_to(F_crackle(Pz_coefs[:, 0], time[0]) == 0)
       
    # treat middle points
    for i in range(1, number_of_points - 1):
        # add the position constraints
        opti.subject_to(F_pos(Px_coefs[:, i-1], time[i]) == states["x"][i])
        opti.subject_to(F_pos(Px_coefs[:, i], time[i]) == states["x"][i])
        
        opti.subject_to(F_pos(Py_coefs[:, i-1], time[i]) == states["y"][i])
        opti.subject_to(F_pos(Py_coefs[:, i], time[i]) == states["y"][i])
        
        opti.subject_to(F_pos(Pz_coefs[:, -1], time[i]) == states["z"][i])
        opti.subject_to(F_pos(Pz_coefs[:, i], time[i]) == states["z"][i])
        
        # add the velocity continuity constraints
        opti.subject_to(F_vel(Px_coefs[:, i-1], time[i]) == F_vel(Px_coefs[:, i], time[i]))
        opti.subject_to(F_vel(Py_coefs[:, i-1], time[i]) == F_vel(Py_coefs[:, i], time[i]))
        opti.subject_to(F_vel(Pz_coefs[:, i-1], time[i]) == F_vel(Pz_coefs[:, i], time[i]))
        
        # add the acceleration continuity constraints
        opti.subject_to(F_acc(Px_coefs[:, i-1], time[i]) == F_acc(Px_coefs[:, i], time[i]))
        opti.subject_to(F_acc(Py_coefs[:, i-1], time[i]) == F_acc(Py_coefs[:, i], time[i]))
        opti.subject_to(F_acc(Pz_coefs[:, i-1], time[i]) == F_acc(Pz_coefs[:, i], time[i]))
        
        # add the jerk continuity constraints
        opti.subject_to(F_jerk(Px_coefs[:, i-1], time[i]) == F_jerk(Px_coefs[:, i], time[i]))
        opti.subject_to(F_jerk(Py_coefs[:, i-1], time[i]) == F_jerk(Py_coefs[:, i], time[i]))
        opti.subject_to(F_jerk(Pz_coefs[:, i-1], time[i]) == F_jerk(Pz_coefs[:, i], time[i]))
        
        # add the snap continuity constraints
        opti.subject_to(F_snap(Px_coefs[:, i-1], time[i]) == F_snap(Px_coefs[:, i], time[i]))
        opti.subject_to(F_snap(Py_coefs[:, i-1], time[i]) == F_snap(Py_coefs[:, i], time[i]))
        opti.subject_to(F_snap(Pz_coefs[:, i-1], time[i]) == F_snap(Pz_coefs[:, i], time[i]))
        
        # add the crackle continuity constraints
        opti.subject_to(F_crackle(Px_coefs[:, i-1], time[i]) == F_crackle(Px_coefs[:, i], time[i]))
        opti.subject_to(F_crackle(Py_coefs[:, i-1], time[i]) == F_crackle(Py_coefs[:, i], time[i]))
        opti.subject_to(F_crackle(Pz_coefs[:, i-1], time[i]) == F_crackle(Pz_coefs[:, i], time[i]))
        
    # treat t=tf as a special case - only position and velocity can be set
    opti.subject_to(F_pos(Px_coefs[:, -1], time[-1]) == states["x"][-1])
    opti.subject_to(F_pos(Py_coefs[:, -1], time[-1]) == states["y"][-1])
    opti.subject_to(F_pos(Pz_coefs[:, -1], time[-1]) == states["z"][-1])   
    
    opti.subject_to(F_vel(Px_coefs[:, -1], time[-1]) == states["vx"][-1])
    opti.subject_to(F_vel(Py_coefs[:, -1], time[-1]) == states["vy"][-1])
    opti.subject_to(F_vel(Pz_coefs[:, -1], time[-1]) == states["vz"][-1])
    
    opti.subject_to(F_acc(Px_coefs[:, -1], time[-1]) == 0)
    opti.subject_to(F_acc(Py_coefs[:, -1], time[-1]) == 0)
    opti.subject_to(F_acc(Pz_coefs[:, -1], time[-1]) == 0)
    
    opti.subject_to(F_jerk(Px_coefs[:, -1], time[-1]) == 0)
    opti.subject_to(F_jerk(Py_coefs[:, -1], time[-1]) == 0)
    opti.subject_to(F_jerk(Pz_coefs[:, -1], time[-1]) == 0)
    
    opti.subject_to(F_snap(Px_coefs[:, -1], time[-1]) == 0)
    opti.subject_to(F_snap(Py_coefs[:, -1], time[-1]) == 0)
    opti.subject_to(F_snap(Pz_coefs[:, -1], time[-1]) == 0)
    
    opti.subject_to(F_crackle(Px_coefs[:, -1], time[-1]) == 0)
    opti.subject_to(F_crackle(Py_coefs[:, -1], time[-1]) == 0)
    opti.subject_to(F_crackle(Pz_coefs[:, -1], time[-1]) == 0)
    
    # add force constraints
    print("Adding bounds...")

    eval_number = 20  # number of points to evaluate the cost and constrints per segment

    # define the cost function and check if the hopper is inside bounds
    for i in range(number_of_points - 1):
        dt = (time[i + 1] - time[i]) / eval_number
        for j in range(eval_number):
            cur_time = time[i] + j * dt  # current time

            # v_cur = ca.vertcat(
            #     F_vel(Px_coefs[:, i], cur_time),
            #     F_vel(Py_coefs[:, i], cur_time),
            #     F_vel(Py_coefs[:, i], cur_time),
            # )
            # opti.subject_to(v_cur[0] > constraints["min_vx"])
            # opti.subject_to(v_cur[0] < constraints["max_vx"])
            # opti.subject_to(v_cur[1] > constraints["min_vy"])
            # opti.subject_to(v_cur[1] < constraints["max_vy"])
            # opti.subject_to(v_cur[2] > constraints["min_vz"])
            # opti.subject_to(v_cur[2] < constraints["max_vz"])

            # a_cur = ca.vertcat(
            #     F_acc(Px_coefs[:, i], cur_time),
            #     F_acc(Py_coefs[:, i], cur_time),
            #     F_acc(Pz_coefs[:, i], cur_time),
            # )
            # opti.subject_to(a_cur[0] > constraints["min_ax"])
            # opti.subject_to(a_cur[0] < constraints["max_ax"])
            # opti.subject_to(a_cur[1] > constraints["min_ay"])
            # opti.subject_to(a_cur[1] < constraints["max_ay"])
            # opti.subject_to(a_cur[2] > constraints["min_az"])
            # opti.subject_to(a_cur[2] < constraints["max_az"])

            # # set maximum and minimum values for control inputs for the force
            # f1_cur, f2_cur = get_f1f2(cur_time, Px_coefs, Py_coefs, params)
            # f1_dot_cur, f2_dot_cur = get_f1f2_dot(cur_time, Px_coefs, Py_coefs, params)

            # f_cur = ca.sqrt(f1_cur**2 + f2_cur**2)
            # f_cur = ca.sqrt(f2_cur**2)
            # f_dot_cur = (f1_cur * f1_dot_cur + f2_cur * f2_dot_cur) / f_cur

            # opti.subject_to(f_cur < control_params["thrust_bounds"][1])
            # opti.subject_to(f_cur > control_params["thrust_bounds"][0])
            # opti.subject_to(f_dot_cur < control_params["u_bounds"][0][1])
            # opti.subject_to(f_dot_cur > control_params["u_bounds"][0][0])

            # # for the delta_tvc
            # delta_tvc_cur = ca.arctan2(f2_cur, f1_cur)
            # delta_tvc_dot_cur = (f1_cur * f2_dot_cur - f2_cur * f1_dot_cur) / (
            #     f1_cur**2 + f2_cur**2
            # )

            # opti.subject_to(delta_tvc_cur < control_params["delta_tvc_bounds"][1])
            # opti.subject_to(delta_tvc_cur > control_params["delta_tvc_bounds"][0])
            # opti.subject_to(delta_tvc_dot_cur < control_params["u_bounds"][1][1])
            # opti.subject_to(delta_tvc_dot_cur > control_params["u_bounds"][1][0])
    
    # add time constraints
    opti.subject_to(time[0] == time_points[0])
    
    # add time continuity constraints
    for i in range(1, number_of_points):
        # opti.subject_to(time[i] >= time[i-1])
        opti.set_initial(time[i], time_points[i])
        opti.subject_to(time[i] == time_points[i])
        opti.subject_to(time[i] > 0)
            
    obj += time[-1]
        
    
    # define the optimization problem
    opti.minimize(obj)
    
    # select the desired solver
    opti.solver("ipopt")

    for i in range(len(time_points) - 1):
        opti.set_initial(
            Px_coefs[:, i],
            estimate_coeffs(
                time_points[i : i + 2], [states["x"][i], states["x"][i + 1]]
            ),
        )
        opti.set_initial(
            Py_coefs[:, i],
            estimate_coeffs(
                time_points[i : i + 2], [states["y"][i], states["y"][i + 1]]
            ),
        )
        opti.set_initial(
            Pz_coefs[:, i],
            estimate_coeffs(
                time_points[i : i + 2], [states["z"][i], states["z"][i + 1]]
            ),
        )

    print("Interpolating trajectory...")
    sol = opti.solve()
    print("Interpolation done!")

    if number_of_points == 2:
        return (
            np.array([sol.value(Px_coefs)]).T,
            np.array([sol.value(Py_coefs)]).T,
            np.array([sol.value(Pz_coefs)]).T,
            time_points,
        )
    else:
        return (
            sol.value(Px_coefs),
            sol.value(Py_coefs),
            sol.value(Pz_coefs),
            time_points,
        )
