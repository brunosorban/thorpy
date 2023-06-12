import casadi as ca
import numpy as np


def get_pos(coefs, t):
    return (
        coefs[0] * t**6
        + coefs[1] * t**5
        + coefs[2] * t**4
        + coefs[3] * t**3
        + coefs[4] * t**2
        + coefs[5] * t
        + coefs[6]
    )


def get_vel(coefs, t):
    return (
        6 * coefs[0] * t**5
        + 5 * coefs[1] * t**4
        + 4 * coefs[2] * t**3
        + 3 * coefs[3] * t**2
        + 2 * coefs[4] * t
        + coefs[5]
    )


def get_acc(coefs, t):
    return (
        30 * coefs[0] * t**4
        + 20 * coefs[1] * t**3
        + 12 * coefs[2] * t**2
        + 6 * coefs[3] * t
        + 2 * coefs[4]
    )


def get_jerk(coefs, t):
    return (
        120 * coefs[0] * t**3
        + 60 * coefs[1] * t**2
        + 24 * coefs[2] * t
        + 6 * coefs[3]
    )

def get_snap(coefs, t):
    return 360 * coefs[0] * t**2 + 120 * coefs[1] * t + 24 * coefs[2]

def get_crackle(coefs, t):
    return 720 * coefs[0] * t + 120 * coefs[1]


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

    pol_order = 7  # order of the polynom +1
    g = constraints["g"]

    # calculate the polynom
    p0 = ca.SX.sym("p0")
    p1 = ca.SX.sym("p1")
    p2 = ca.SX.sym("p2")
    p3 = ca.SX.sym("p3")
    p4 = ca.SX.sym("p4")
    p5 = ca.SX.sym("p5")
    p6 = ca.SX.sym("p6")
    t = ca.SX.sym("t")

    coefs = ca.vertcat(p0, p1, p2, p3, p4, p5, p6)
    pos = get_pos(coefs, t)
    vel = get_vel(coefs, t)
    acc = get_acc(coefs, t)
    jerk = get_jerk(coefs, t)
    snap = get_snap(coefs, t)
    crackle = get_crackle(coefs, t)

    F_pos = ca.Function(
        "F_pos",
        [coefs, t],
        [pos],
        [
            "[p0, p1, p2, p3, p4, p5, p6]",
            "[t]",
        ],
        ["position"],
    )

    F_vel = ca.Function(
        "F_vel",
        [coefs, t],
        [vel],
        [
            "[p0, p1, p2, p3, p4, p5, p6]",
            "[t]",
        ],
        ["velocity"],
    )

    F_acc = ca.Function(
        "F_acc",
        [coefs, t],
        [acc],
        [
            "[p0, p1, p2, p3, p4, p5, p6]",
            "[t]",
        ],
        ["acceleration"],
    )

    F_jerk = ca.Function(
        "F_jerk",
        [coefs, t],
        [jerk],
        [
            "[p0, p1, p2, p3, p4, p5, p6]",
            "[t]",
        ],
        ["jerk"],
    )

    F_snap = ca.Function(
        "F_snap",
        [coefs, t],
        [snap],
        [
            "[p0, p1, p2, p3, p4, p5, p6]",
            "[t]",
        ],
        ["snap"],
    )

    F_crackle = ca.Function(
        "F_crackle",
        [coefs, t],
        [crackle],
        [
            "[p0, p1, p2, p3, p4, p5, p6]",
            "[t]",
        ],
        ["crackle"],
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
            
            # # crackle constraints (continuity) -> this is too much
            # opti.subject_to(
            #     F_crackle(Px_coefs[:, k], time_points[k + 1])
            #     == F_crackle(Px_coefs[:, k + 1], time_points[k + 1])
            # )
            # opti.subject_to(
            #     F_crackle(Py_coefs[:, k], time_points[k + 1])
            #     == F_crackle(Py_coefs[:, k + 1], time_points[k + 1])
            # )
            # opti.subject_to(
            #     F_crackle(Pz_coefs[:, k], time_points[k + 1])
            #     == F_crackle(Pz_coefs[:, k + 1], time_points[k + 1])
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
