import casadi as ca
import numpy as np
import copy
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

    pol_order = 9  # order of the polynom +1
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
    # p9 = ca.SX.sym("p9")
    t = ca.SX.sym("t")

    coefs = ca.vertcat(p0, p1, p2, p3, p4, p5, p6, p7, p8)
    pos = get_pos(coefs, t)
    vel = get_vel(coefs, t)
    acc = get_acc(coefs, t)
    jerk = get_jerk(coefs, t)
    snap = get_snap(coefs, t)
    crackle = get_crackle(coefs, t)
    pop = get_pop(coefs, t)

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

    F_pop = ca.Function(
        "F_pop",
        [coefs, t],
        [pop],
        [
            "[p0, p1, p2, p3, p4, p5, p6, p7, p8]",
            "[t]",
        ],
        ["pop"],
    )

    params = copy.deepcopy(rocket_params)
    params["g"] = g

    # building the optimal control problem
    opti = ca.Opti()

    # create the variables with the polynom coefficients
    Px_coefs = opti.variable(pol_order, number_of_points - 1)
    Py_coefs = opti.variable(pol_order, number_of_points - 1)
    Pz_coefs = opti.variable(pol_order, number_of_points - 1)
    
    # define the cost function
    # it will be the sum of the crackle and position error as soft constraints
    obj = 0 

    # set the contraints
    print("number of points (per axis): ", number_of_points)
    print("number of polynoms (per axis): ", number_of_points - 1)
    print("Adding constraints...")
    for k in range(number_of_points - 1):
        # print()
        # print("k = ", k, " of ", number_of_points - 2, " time = ", time_points[k])
        # initial postion constraints
        obj += ca.power(F_pos(Px_coefs[:, k], time_points[k]) - states["x"][k], 2)
        obj += ca.power(F_pos(Py_coefs[:, k], time_points[k]) - states["y"][k], 2)
        obj += ca.power(F_pos(Pz_coefs[:, k], time_points[k]) - states["z"][k], 2)

        # final postion constraints
        obj += ca.power(F_pos(Px_coefs[:, k], time_points[k + 1]) - states["x"][k + 1], 2)
        obj += ca.power(F_pos(Py_coefs[:, k], time_points[k + 1]) - states["y"][k + 1], 2)
        obj += ca.power(F_pos(Pz_coefs[:, k], time_points[k + 1]) - states["z"][k + 1], 2)

        if k < number_of_points - 2:
            # position constraints (continuity)
            opti.subject_to(
                F_pos(Px_coefs[:, k], time_points[k + 1])
                == F_pos(Px_coefs[:, k + 1], time_points[k + 1])
            )
            opti.subject_to(
                F_pos(Py_coefs[:, k], time_points[k + 1])
                == F_pos(Py_coefs[:, k + 1], time_points[k + 1])
            )
            opti.subject_to(
                F_pos(Pz_coefs[:, k], time_points[k + 1])
                == F_pos(Pz_coefs[:, k + 1], time_points[k + 1])
            )
            
            # velocity constraints (continuity)
            opti.subject_to(
                F_vel(Px_coefs[:, k], time_points[k + 1])
                == F_vel(Px_coefs[:, k + 1], time_points[k + 1])
            )
            opti.subject_to(
                F_vel(Py_coefs[:, k], time_points[k + 1])
                == F_vel(Py_coefs[:, k + 1], time_points[k + 1])
            )
            opti.subject_to(
                F_vel(Pz_coefs[:, k], time_points[k + 1])
                == F_vel(Pz_coefs[:, k + 1], time_points[k + 1])
            )

            # acceleration constraints (continuity)
            opti.subject_to(
                F_acc(Px_coefs[:, k], time_points[k + 1])
                == F_acc(Px_coefs[:, k + 1], time_points[k + 1])
            )
            opti.subject_to(
                F_acc(Py_coefs[:, k], time_points[k + 1])
                == F_acc(Py_coefs[:, k + 1], time_points[k + 1])
            )
            opti.subject_to(
                F_acc(Pz_coefs[:, k], time_points[k + 1])
                == F_acc(Pz_coefs[:, k + 1], time_points[k + 1])
            )

            # jerk constraints (continuity)
            opti.subject_to(
                F_jerk(Px_coefs[:, k], time_points[k + 1])
                == F_jerk(Px_coefs[:, k + 1], time_points[k + 1])
            )
            opti.subject_to(
                F_jerk(Py_coefs[:, k], time_points[k + 1])
                == F_jerk(Py_coefs[:, k + 1], time_points[k + 1])
            )
            opti.subject_to(
                F_jerk(Pz_coefs[:, k], time_points[k + 1])
                == F_jerk(Pz_coefs[:, k + 1], time_points[k + 1])
            )

            # snap constraints (continuity)
            opti.subject_to(
                F_snap(Px_coefs[:, k], time_points[k + 1])
                == F_snap(Px_coefs[:, k + 1], time_points[k + 1])
            )
            opti.subject_to(
                F_snap(Py_coefs[:, k], time_points[k + 1])
                == F_snap(Py_coefs[:, k + 1], time_points[k + 1])
            )
            opti.subject_to(
                F_snap(Pz_coefs[:, k], time_points[k + 1])
                == F_snap(Pz_coefs[:, k + 1], time_points[k + 1])
            )

            # crackle constraints (continuity)
            opti.subject_to(
                F_crackle(Px_coefs[:, k], time_points[k + 1])
                == F_crackle(Px_coefs[:, k + 1], time_points[k + 1])
            )
            opti.subject_to(
                F_crackle(Py_coefs[:, k], time_points[k + 1])
                == F_crackle(Py_coefs[:, k + 1], time_points[k + 1])
            )
            opti.subject_to(
                F_crackle(Pz_coefs[:, k], time_points[k + 1])
                == F_crackle(Pz_coefs[:, k + 1], time_points[k + 1])
            )

            # # pop constraints (continuity)
            # opti.subject_to(
            #     F_pop(Px_coefs[:, k], time_points[k + 1])
            #     == F_pop(Px_coefs[:, k + 1], time_points[k + 1])
            # )
            # opti.subject_to(
            #     F_pop(Py_coefs[:, k], time_points[k + 1])
            #     == F_pop(Py_coefs[:, k + 1], time_points[k + 1])
            # )
            # opti.subject_to(
            #     F_pop(Pz_coefs[:, k], time_points[k + 1])
            #     == F_pop(Pz_coefs[:, k + 1], time_points[k + 1])
            # )

        # check if velocity and acceleration constraints were given
        if states["vx"][k] != None:
            opti.subject_to(F_vel(Px_coefs[:, k], time_points[k]) == states["vx"][k])
        if states["vy"][k] != None:
            opti.subject_to(F_vel(Py_coefs[:, k], time_points[k]) == states["vy"][k])
        if states["vz"][k] != None:
            opti.subject_to(F_vel(Px_coefs[:, k], time_points[k]) == states["vz"][k])
        if states["ax"][k] != None:
            opti.subject_to(F_acc(Px_coefs[:, k], time_points[k]) == states["ax"][k])
        if states["ay"][k] != None:
            opti.subject_to(F_acc(Py_coefs[:, k], time_points[k]) == states["ay"][k])
        if states["az"][k] != None:
            opti.subject_to(F_acc(Pz_coefs[:, k], time_points[k]) == states["az"][k])
        if states["gamma_dot"][k] != None:
            gamma_dot = (
                F_jerk(Py_coefs, time_points[k]) * F_acc(Px_coefs[:, k], time_points[k])
                - (F_acc(Py_coefs[:, k], time_points[k]) + g)
                * F_jerk(Px_coefs[:, k], time_points[k])
            ) / (
                F_acc(Px_coefs[:, k], time_points[k]) ** 2
                + (F_acc(Py_coefs[:, k], time_points[k]) + g) ** 2
            )
            opti.subject_to(gamma_dot == states["gamma_dot"][k])

    # velocity and acceleration contraints at final point
    # check if velocity and acceleration constraints were given
    if states["vx"][k + 1] != None:
        opti.subject_to(
            F_vel(Px_coefs[:, k], time_points[k + 1]) == states["vx"][k + 1]
        )
    if states["vy"][k + 1] != None:
        opti.subject_to(
            F_vel(Py_coefs[:, k], time_points[k + 1]) == states["vy"][k + 1]
        )
    if states["vz"][k + 1] != None:
        opti.subject_to(
            F_vel(Pz_coefs[:, k], time_points[k + 1]) == states["vz"][k + 1]
        )
    if states["ax"][k + 1] != None:
        opti.subject_to(
            F_acc(Px_coefs[:, k], time_points[k + 1]) == states["ax"][k + 1]
        )
    if states["ay"][k + 1] != None:
        opti.subject_to(
            F_acc(Py_coefs[:, k], time_points[k + 1]) == states["ay"][k + 1]
        )
    if states["az"][k + 1] != None:
        opti.subject_to(
            F_acc(Pz_coefs[:, k], time_points[k + 1]) == states["az"][k + 1]
        )
    if states["gamma_dot"][k + 1] != None:
        gamma_dot = (
            F_jerk(Py_coefs, time_points[k + 1])
            * F_acc(Px_coefs[:, k], time_points[k + 1])
            - (F_acc(Py_coefs[:, k], time_points[k + 1]) + g)
            * F_jerk(Px_coefs[:, k], time_points[k + 1])
        ) / (
            F_acc(Px_coefs[:, k], time_points[k + 1]) ** 2
            + (F_acc(Py_coefs[:, k], time_points[k + 1]) + g) ** 2
        )
        opti.subject_to(gamma_dot == states["gamma_dot"][k + 1])

    print("Adding bounds...")

    eval_number = 20  # number of points to evaluate the cost and constrints per segment

    # define the cost function and check if the hopper is inside bounds
    for i in range(number_of_points - 1):
        dt = (time_points[i + 1] - time_points[i]) / eval_number
        for j in range(eval_number):
            cur_time = time_points[i] + j * dt  # current time

            # minimize the crackle
            obj += (
                ca.power(F_crackle(Px_coefs[0, i], cur_time), 2)
                + ca.power(F_crackle(Py_coefs[0, i], cur_time), 2)
                + ca.power(F_crackle(Pz_coefs[0, i], cur_time), 2)
            )
            
            # opti.subject_to(F_pos(Py_coefs[:, i], cur_time) >= 0)

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

            # set maximum and minimum values for control inputs
            # for the force
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

    opti.minimize(obj)
    # select the desired solver
    # hide solution output
    opts = {"ipopt.print_level": 0, "print_time": 0}
    opti.solver("ipopt")  # , opts)

    # opti.set_initial(Px_coefs, np.ones((pol_order, number_of_points - 1)))
    # opti.set_initial(Py_coefs, np.ones((pol_order, number_of_points - 1)))
    # opti.set_initial(Pz_coefs, np.ones((pol_order, number_of_points - 1)))

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
