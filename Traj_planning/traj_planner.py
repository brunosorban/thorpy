import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


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


def get_pos_gamma(coefs, t):
    return coefs[0] * t * 2 + coefs[1] * t + coefs[2]


def get_vel_gamma(coefs, t):
    return 2 * coefs[0] * t + coefs[1]


def get_acc_gamma(coefs, t):
    return 2 * coefs[0]


def calculate_trajectory(time_points, states, constraints):
    if len(time_points) != len(states["x"]):
        raise ValueError(
            "The number of time points must be equal to the number of states"
        )

    number_of_points = len(time_points)

    pol_order = 7  # order of the polynom +1
    pol_order_gamma = 3

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

    coefs_gamma = ca.vertcat(p0, p1, p2)
    pos_gamma = get_pos_gamma(coefs_gamma, t)
    vel_gamma = get_vel_gamma(coefs_gamma, t)
    acc_gamma = get_acc_gamma(coefs_gamma, t)

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

    F_pos_gamma = ca.Function(
        "F_pos_gamma",
        [coefs_gamma, t],
        [pos_gamma],
        [
            "[p0, p1, p2]",
            "[t]",
        ],
        ["angular position"],
    )

    F_vel_gamma = ca.Function(
        "F_vel_gamma",
        [coefs_gamma, t],
        [vel_gamma],
        [
            "[p0, p1, p2]",
            "[t]",
        ],
        ["angular velocity"],
    )

    F_acc_gamma = ca.Function(
        "F_acc_gamma",
        [coefs_gamma, t],
        [acc_gamma],
        [
            "[p0, p1, p2]",
            "[t]",
        ],
        ["angular acceleration"],
    )
    # building the optimal control problem
    opti = ca.Opti()

    # create the variables with the polynom coefficients
    Px_coefs = opti.variable(pol_order, number_of_points - 1)
    Py_coefs = opti.variable(pol_order, number_of_points - 1)
    Pgamma_coefs = opti.variable(pol_order_gamma, number_of_points - 1)

    # define the cost function
    obj = 0
    for k in range(0, number_of_points - 1):
        # minimize the jerk
        obj += (
            ca.power(Px_coefs[0, k], 2)
            + ca.power(Py_coefs[0, k], 2)
            + ca.power(Pgamma_coefs[0, k], 2)
        )

    opti.minimize(obj)

    # set the contraints
    print("number of points: ", number_of_points)
    print("number of polynoms: ", number_of_points - 1)
    for k in range(0, number_of_points - 1):
        print()
        print("k = ", k, " of ", number_of_points - 2, " time = ", time_points[k])
        # initial postion constraints
        opti.subject_to(F_pos(Px_coefs[:, k], time_points[k]) == states["x"][k])
        opti.subject_to(F_pos(Py_coefs[:, k], time_points[k]) == states["y"][k])
        opti.subject_to(
            F_pos_gamma(Pgamma_coefs[:, k], time_points[k]) == states["gamma"][k]
        )

        # final postion constraints
        opti.subject_to(F_pos(Px_coefs[:, k], time_points[k + 1]) == states["x"][k + 1])
        opti.subject_to(F_pos(Py_coefs[:, k], time_points[k + 1]) == states["y"][k + 1])
        opti.subject_to(
            F_pos_gamma(Pgamma_coefs[:, k], time_points[k + 1])
            == states["gamma"][k + 1]
        )

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
                F_vel_gamma(Pgamma_coefs[:, k], time_points[k + 1])
                == F_vel_gamma(Pgamma_coefs[:, k + 1], time_points[k + 1])
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
            # opti.subject_to(F_acc_gamma(Pgamma_coefs[:, k], time_points[k+1]) == F_acc_gamma(Pgamma_coefs[:, k+1], time_points[k+1]))

        # check if velocity and acceleration constraints were given
        if states["vx"][k] != None:
            opti.subject_to(F_vel(Px_coefs[:, k], time_points[k]) == states["vx"][k])
        if states["vy"][k] != None:
            opti.subject_to(F_vel(Py_coefs[:, k], time_points[k]) == states["vy"][k])
        if states["ax"][k] != None:
            opti.subject_to(F_acc(Px_coefs[:, k], time_points[k]) == states["ax"][k])
            print(
                "Acceleration constraint added at t={:.1f} of ax={:.1f}".format(
                    time_points[k], states["ax"][k]
                )
            )
        if states["ay"][k] != None:
            opti.subject_to(F_acc(Py_coefs[:, k], time_points[k]) == states["ay"][k])
            print(
                "Acceleration constraint added at t={:.1f} of ay={:.1f}".format(
                    time_points[k], states["ay"][k]
                )
            )

    # velocity constraints (initial conditions)
    opti.subject_to(F_vel(Px_coefs[:, 0], time_points[0]) == constraints["vx0"])
    opti.subject_to(F_vel(Py_coefs[:, 0], time_points[0]) == constraints["vy0"])
    opti.subject_to(
        F_vel_gamma(Pgamma_coefs[:, 0], time_points[0]) == constraints["gamma_dot0"]
    )

    # velocity and acceleration contraints at final point
    # check if velocity and acceleration constraints were given
    k += 1
    if states["vx"][k] != None:
        opti.subject_to(F_vel(Px_coefs[:, k - 1], time_points[k]) == states["vx"][k])
    if states["vy"][k] != None:
        opti.subject_to(F_vel(Py_coefs[:, k - 1], time_points[k]) == states["vy"][k])
    if states["ax"][k] != None:
        opti.subject_to(F_acc(Px_coefs[:, k - 1], time_points[k]) == states["ax"][k])
        print(
            "Acceleration constraint added at t={:.1f} of ax={:.1f}".format(
                time_points[k], states["ax"][k]
            )
        )
    if states["ay"][k] != None:
        opti.subject_to(F_acc(Py_coefs[:, k - 1], time_points[k]) == states["ay"][k])
        print(
            "Acceleration constraint added at t={:.1f} of ay={:.1f}".format(
                time_points[k], states["ay"][k]
            )
        )

    # check if the hopper is inside bounds
    for i in range(number_of_points - 1):
        dt = (time_points[i + 1] - time_points[i]) / 100
        for j in range(100):
            cur_time = time_points[i] + j * dt
            t_interval = time_points[i + 1] - time_points[i]
            dx = states["x"][i + 1] - states["x"][i]
            dy = states["y"][i + 1] - states["y"][i]
            dgamma = states["gamma"][i + 1] - states["gamma"][i]

            cur_interpolated_pos_x = states["x"][i] + (dx / t_interval) * (
                cur_time - time_points[i]
            )
            cur_interpolated_pos_y = states["y"][i] + (dy / t_interval) * (
                cur_time - time_points[i]
            )
            cur_interpolated_pos_gamma = states["gamma"][i] + (dgamma / t_interval) * (
                cur_time - time_points[i]
            )

            # opti.subject_to(F_pos(Px_coefs[:, i], cur_time) > cur_interpolated_pos_x - constraints["offset"])
            # opti.subject_to(F_pos(Py_coefs[:, i], cur_time) > cur_interpolated_pos_y - constraints["offset"])
            # opti.subject_to(F_pos(Px_coefs[:, i], cur_time) < cur_interpolated_pos_x + constraints["offset"])
            # opti.subject_to(F_pos(Py_coefs[:, i], cur_time) < cur_interpolated_pos_y + constraints["offset"])
            opti.subject_to(
                F_pos_gamma(Pgamma_coefs[:, i], cur_time)
                > cur_interpolated_pos_gamma - constraints["angle_offset"]
            )
            opti.subject_to(
                F_pos_gamma(Pgamma_coefs[:, i], cur_time)
                < cur_interpolated_pos_gamma + constraints["angle_offset"]
            )

            # acceleration constraints
            opti.subject_to(F_acc(Px_coefs[:, i], cur_time) > constraints["min_acc_x"])
            opti.subject_to(F_acc(Py_coefs[:, i], cur_time) > constraints["min_acc_y"])
            opti.subject_to(F_acc(Py_coefs[:, i], cur_time) < constraints["max_acc_y"])
            opti.subject_to(F_acc(Px_coefs[:, i], cur_time) < constraints["max_acc_x"])

    # select the desired solver
    # hide solution output
    opts = {"ipopt.print_level": 0, "print_time": 0}
    opti.solver("ipopt")  # , opts)

    sol = opti.solve()
    return sol.value(Px_coefs), sol.value(Py_coefs), sol.value(Pgamma_coefs)


def get_traj_params(time_points, states, constraints, plot=False):
    # calculate trajectory
    Px_coefs, Py_coefs, Pgamma_coefs = calculate_trajectory(
        time_points, states, constraints
    )

    # initialize variables
    n = 5000
    t = np.linspace(time_points[0], time_points[-1], n, endpoint=True)
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)
    vx = np.zeros(n)
    vy = np.zeros(n)
    vz = np.zeros(n)
    ax = np.zeros(n)
    ay = np.zeros(n)
    az = np.zeros(n)
    gamma = np.zeros(n)
    gamma_dot = np.zeros(n)
    gamma_ddot = np.zeros(n)

    for i in range(n):
        ind = np.searchsorted(time_points, t[i])
        if ind > 0:
            ind -= 1
        x[i] = get_pos(Px_coefs[:, ind], t[i])
        y[i] = get_pos(Py_coefs[:, ind], t[i])
        vx[i] = get_vel(Px_coefs[:, ind], t[i])
        vy[i] = get_vel(Py_coefs[:, ind], t[i])
        ax[i] = get_acc(Px_coefs[:, ind], t[i])
        ay[i] = get_acc(Py_coefs[:, ind], t[i])
        gamma[i] = get_pos_gamma(Pgamma_coefs[:, ind], t[i])
        gamma_dot[i] = get_vel_gamma(Pgamma_coefs[:, ind], t[i])
        gamma_ddot[i] = get_acc_gamma(Pgamma_coefs[:, ind], t[i])

    trajectory = {
        "t": t,
        "x": x,
        "y": y,
        "z": z,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "ax": ax,
        "ay": ay,
        "az": az,
        "gamma": gamma,  # + np.deg2rad(90),
        "gamma_dot": gamma_dot,
        "gamma_ddot": gamma_ddot,
    }

    if plot:
        plot_trajectory(time_points, states, Px_coefs, Py_coefs, Pgamma_coefs)

    return trajectory


def plot_trajectory(time_points, states, Px_coefs, Py_coefs, Pgamma_coefs):
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))

    t = np.linspace(time_points[0], time_points[-1], 100)
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    gamma = np.zeros(len(t))
    vx = np.zeros(len(t))
    vy = np.zeros(len(t))
    gamma_dot = np.zeros(len(t))
    ax = np.zeros(len(t))
    ay = np.zeros(len(t))

    for k in range(0, len(time_points) - 1):
        x[t >= time_points[k]] = get_pos(Px_coefs[:, k], t[t >= time_points[k]])
        y[t >= time_points[k]] = get_pos(Py_coefs[:, k], t[t >= time_points[k]])
        gamma[t >= time_points[k]] = get_pos_gamma(
            Pgamma_coefs[:, k], t[t >= time_points[k]]
        )
        # gamma[t >= time_points[k]] = np.arctan2(get_vel(Py_coefs[:, k], t[t >= time_points[k]]), get_vel(Px_coefs[:, k], t[t >= time_points[k]]))
        vx[t >= time_points[k]] = get_vel(Px_coefs[:, k], t[t >= time_points[k]])
        vy[t >= time_points[k]] = get_vel(Py_coefs[:, k], t[t >= time_points[k]])
        gamma_dot[t >= time_points[k]] = get_vel_gamma(
            Pgamma_coefs[:, k], t[t >= time_points[k]]
        )
        ax[t >= time_points[k]] = get_acc(Px_coefs[:, k], t[t >= time_points[k]])
        ay[t >= time_points[k]] = get_acc(Py_coefs[:, k], t[t >= time_points[k]])

    axs[0, 0].plot(x, y, label="trajectory")
    axs[0, 0].plot(states["x"], states["y"], "o", label="target points")
    axs[0, 0].set_xlabel("Horizontal Position (m)")
    axs[0, 0].set_ylabel("Vertical Position (m)")
    axs[0, 0].set_title("Trajectory")
    axs[0, 0].set_aspect("equal")
    axs[0, 0].legend()
    axs[0, 0].grid()

    axs[1, 0].plot(t, x, label="x")
    axs[1, 0].plot(t, y, label="y")
    axs[1, 0].plot(time_points, states["x"], "o", label="$x_{ref}$")
    axs[1, 0].plot(time_points, states["y"], "o", label="$y_{ref}$")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Position (m)")
    axs[1, 0].set_title("Position vs Time")
    axs[1, 0].legend()
    axs[1, 0].grid()

    axs[2, 0].plot(t, np.rad2deg(gamma), label="$\\gamma$")
    axs[2, 0].plot(
        time_points, np.rad2deg(states["gamma"]), "o", label="$\\gamma_{ref}$"
    )
    axs[2, 0].set_xlabel("Time (s)")
    axs[2, 0].set_ylabel("Angle (deg)")
    axs[2, 0].set_title("Angle vs Time")
    axs[2, 0].legend()
    axs[2, 0].grid()

    axs[0, 1].plot(t, vx, label="vx")
    axs[0, 1].plot(t, vy, label="vy")
    axs[0, 1].plot(time_points, states["vx"], "o", label="$v_{x_{ref}}$")
    axs[0, 1].plot(time_points, states["vy"], "o", label="$v_{y_{ref}}$")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Velocity (m/s)")
    axs[0, 1].set_title("Velocity vs Time")
    axs[0, 1].legend()
    axs[0, 1].grid()

    axs[1, 1].plot(t, ax, label="ax")
    axs[1, 1].plot(t, ay, label="ay")
    axs[1, 1].plot(time_points, states["ax"], "o", label="$a_{x_{ref}}$")
    axs[1, 1].plot(time_points, states["ay"], "o", label="$a_{y_{ref}}$")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Acceleration (m/s^2)")
    axs[1, 1].set_title("Acceleration vs Time")
    axs[1, 1].legend()
    axs[1, 1].grid()

    axs[2, 1].plot(t, np.rad2deg(gamma_dot), label="$\\dot{\\gamma}$")
    # axs[2,1].plot(time_points, np.rad2deg(states["gamma_dot"]), "o", label="$\\gamma_dot_{ref}$")
    axs[2, 1].set_xlabel("Time (s)")
    axs[2, 1].set_ylabel("Angular Velocity (deg/s)")
    axs[2, 1].set_title("Angular Velocity vs Time")
    axs[2, 1].legend()
    axs[2, 1].grid()

    plt.show()


def calculate_gamma(time_points, Px_coefs, Py_coefs):  # , constraints):
    number_of_points = len(time_points)
    discretization = 1000

    # calculate gamma points from x and y points
    vx_points = np.zeros(discretization * (number_of_points - 1))
    vy_points = np.zeros(discretization * (number_of_points - 1))
    gamma_points = np.zeros(discretization * (number_of_points - 1))

    for i in range(0, number_of_points - 1):
        for j in range(discretization):
            k = i * discretization + j
            time = (
                time_points[i]
                + j * (time_points[i + 1] - time_points[i]) / discretization
            )
            vx_points[k] = get_vel(Px_coefs[:, i], time)
            vy_points[k] = get_vel(Py_coefs[:, i], time)
            gamma_points[k] = np.arctan2(vy_points[k], vx_points[k])
    return gamma_points
