import casadi as ca
import numpy as np

def get_pos(coefs, t):
    return (
        coefs[0] * t ** 7
        + coefs[1] * t ** 6
        + coefs[2] * t ** 5
        + coefs[3] * t ** 4
        + coefs[4] * t ** 3
        + coefs[5] * t ** 2
        + coefs[6] * t
        + coefs[7]
    )
    
def get_vel(coefs, t):
    return (
        7 * coefs[0] * t ** 6
        + 6 * coefs[1] * t ** 5
        + 5 * coefs[2] * t ** 4
        + 4 * coefs[3] * t ** 3
        + 3 * coefs[4] * t ** 2
        + 2 * coefs[5] * t
        + coefs[6]
    )
    
def get_acc(coefs, t):
    return (
        42 * coefs[0] * t ** 5
        + 30 * coefs[1] * t ** 4
        + 20 * coefs[2] * t ** 3
        + 12 * coefs[3] * t ** 2
        + 6 * coefs[4] * t
        + 2 * coefs[5]
    )
    
def get_jerk(coefs, t):
    return (
        210 * coefs[0] * t ** 4
        + 120 * coefs[1] * t ** 3
        + 60 * coefs[2] * t ** 2
        + 24 * coefs[3] * t
        + 6 * coefs[4]
    )
    
def get_snap(coefs, t):
    return (
        840 * coefs[0] * t ** 3
        + 360 * coefs[1] * t ** 2
        + 120 * coefs[2] * t
        + 24 * coefs[3]
    )
    
def get_crackle(coefs, t):
    return (
        2520 * coefs[0] * t ** 2
        + 720 * coefs[1] * t
        + 120 * coefs[2]
    )
    
def get_pop(coefs, t):
    return (
        5040 * coefs[0] * t
        + 720 * coefs[1]
    )
    
# def get_pos(coefs, t):
#     return (
#         coefs[0] * t ** 8
#         + coefs[1] * t ** 7
#         + coefs[2] * t ** 6
#         + coefs[3] * t ** 5
#         + coefs[4] * t ** 4
#         + coefs[5] * t ** 3
#         + coefs[6] * t ** 2
#         + coefs[7] * t
#         + coefs[8]
#     )
    
# def get_vel(coefs, t):
#     return (
#         56 * coefs[0] * t ** 7
#         + 42 * coefs[1] * t ** 6
#         + 30 * coefs[2] * t ** 5
#         + 20 * coefs[3] * t ** 4
#         + 12 * coefs[4] * t ** 3
#         + 6 * coefs[5] * t ** 2
#         + 2 * coefs[6] * t
#         + coefs[7]
#     )
    
# def get_acc(coefs, t):
#     return (
#         336 * coefs[0] * t ** 6
#         + 252 * coefs[1] * t ** 5
#         + 150 * coefs[2] * t ** 4
#         + 80 * coefs[3] * t ** 3
#         + 36 * coefs[4] * t ** 2
#         + 12 * coefs[5] * t
#         + 2 * coefs[6]
#     )
    
# def get_jerk(coefs, t):
#     return (
#         1680 * coefs[0] * t ** 5
#         + 1260 * coefs[1] * t ** 4
#         + 600 * coefs[2] * t ** 3
#         + 240 * coefs[3] * t ** 2
#         + 72 * coefs[4] * t
#         + 12 * coefs[5]
#     )
    
# def get_snap(coefs, t):
#     return (
#         6720 * coefs[0] * t ** 4
#         + 5040 * coefs[1] * t ** 3
#         + 1800 * coefs[2] * t ** 2
#         + 480 * coefs[3] * t
#         + 72 * coefs[4]
#     )
    
# def get_crackle(coefs, t):
#     return (
#         26880 * coefs[0] * t ** 3
#         + 15120 * coefs[1] * t ** 2
#         + 3600 * coefs[2] * t
#         + 480 * coefs[3]
#     )
    
# def get_pop(coefs, t):
#     return (
#         80640 * coefs[0] * t ** 2
#         + 30240 * coefs[1] * t
#         + 3600 * coefs[2]
#     )


def min_snap_traj(states, constraints):
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

    pol_order = 8  # order of the polynom +1
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
    # p8 = ca.SX.sym("p8")
    t = ca.SX.sym("t")

    coefs = ca.vertcat(p0, p1, p2, p3, p4, p5, p6, p7)
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

    # building the optimal control problem
    opti = ca.Opti()

    # create the variables with the polynom coefficients
    Px_coefs = opti.variable(pol_order, number_of_points - 1)
    Py_coefs = opti.variable(pol_order, number_of_points - 1)
    Pz_coefs = opti.variable(pol_order, number_of_points - 1)

    # define the cost function
    obj = 0
    for k in range(0, number_of_points - 1):
        # minimize the crackle
        obj += (
            ca.power(F_crackle(Px_coefs[0, k], time_points[k]), 2)
            + ca.power(F_crackle(Py_coefs[0, k], time_points[k]), 2)
            + ca.power(F_crackle(Pz_coefs[0, k], time_points[k]), 2)
        )

    opti.minimize(obj)

    # set the contraints
    print("number of points (per axis): ", number_of_points)
    print("number of polynoms (per axis): ", number_of_points - 1)
    print("Adding constraints...")
    for k in range(number_of_points - 1):
        # print()
        # print("k = ", k, " of ", number_of_points - 2, " time = ", time_points[k])
        # initial postion constraints
        opti.subject_to(F_pos(Px_coefs[:, k], time_points[k]) == states["x"][k])
        opti.subject_to(F_pos(Py_coefs[:, k], time_points[k]) == states["y"][k])
        opti.subject_to(F_pos(Pz_coefs[:, k], time_points[k]) == states["z"][k])

        # final postion constraints
        opti.subject_to(F_pos(Px_coefs[:, k], time_points[k + 1]) == states["x"][k + 1])
        opti.subject_to(F_pos(Py_coefs[:, k], time_points[k + 1]) == states["y"][k + 1])
        opti.subject_to(F_pos(Pz_coefs[:, k], time_points[k + 1]) == states["z"][k + 1])

        if k < number_of_points - 2:
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
            
            # pop constraints (continuity)
            opti.subject_to(
                F_pop(Px_coefs[:, k], time_points[k + 1])
                == F_pop(Px_coefs[:, k + 1], time_points[k + 1])
            )
            opti.subject_to(
                F_pop(Py_coefs[:, k], time_points[k + 1])
                == F_pop(Py_coefs[:, k + 1], time_points[k + 1])
            )
            opti.subject_to(
                F_pop(Pz_coefs[:, k], time_points[k + 1])
                == F_pop(Pz_coefs[:, k + 1], time_points[k + 1])
            )

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

    # velocity and acceleration contraints at final point
    # check if velocity and acceleration constraints were given
    k += 1
    if states["vx"][k] != None:
        opti.subject_to(F_vel(Px_coefs[:, k - 1], time_points[k]) == states["vx"][k])
    if states["vy"][k] != None:
        opti.subject_to(F_vel(Py_coefs[:, k - 1], time_points[k]) == states["vy"][k])
    if states["vz"][k] != None:
        opti.subject_to(F_vel(Pz_coefs[:, k - 1], time_points[k]) == states["vz"][k])
    if states["ax"][k] != None:
        opti.subject_to(F_acc(Px_coefs[:, k - 1], time_points[k]) == states["ax"][k])
    if states["ay"][k] != None:
        opti.subject_to(F_acc(Py_coefs[:, k - 1], time_points[k]) == states["ay"][k])
    if states["az"][k] != None:
        opti.subject_to(F_acc(Pz_coefs[:, k - 1], time_points[k]) == states["az"][k])

    print("Adding bounds...")

    # check if the hopper is inside bounds
    for i in range(number_of_points - 1):
        dt = (time_points[i + 1] - time_points[i]) / 100
        for j in range(100):
            cur_time = time_points[i] + j * dt

            v = ca.vertcat(
                F_vel(Px_coefs[:, i], cur_time),
                F_vel(Py_coefs[:, i], cur_time),
                F_vel(Py_coefs[:, i], cur_time),
            )
            opti.subject_to(v[0] > constraints["min_vx"])
            opti.subject_to(v[0] < constraints["max_vx"])
            opti.subject_to(v[1] > constraints["min_vy"])
            opti.subject_to(v[1] < constraints["max_vy"])
            opti.subject_to(v[2] > constraints["min_vz"])
            opti.subject_to(v[2] < constraints["max_vz"])

            a = ca.vertcat(
                F_acc(Px_coefs[:, i], cur_time),
                F_acc(Py_coefs[:, i], cur_time),
                F_acc(Pz_coefs[:, i], cur_time),
            )
            opti.subject_to(a[0] > constraints["min_ax"])
            opti.subject_to(a[0] < constraints["max_ax"])
            opti.subject_to(a[1] > constraints["min_ay"])
            opti.subject_to(a[1] < constraints["max_ay"])
            opti.subject_to(a[2] > constraints["min_az"])
            opti.subject_to(a[2] < constraints["max_az"])

    # select the desired solver
    # hide solution output
    opts = {"ipopt.print_level": 0, "print_time": 0}
    opti.solver("ipopt")  # , opts)

    opti.set_initial(Px_coefs, np.ones((pol_order, number_of_points - 1)))
    opti.set_initial(Py_coefs, np.ones((pol_order, number_of_points - 1)))
    opti.set_initial(Pz_coefs, np.ones((pol_order, number_of_points - 1)))

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
