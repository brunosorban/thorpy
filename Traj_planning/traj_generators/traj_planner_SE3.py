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


def calculate_trajectory(time_points, states, constraints):
    if len(time_points) != len(states["x"]):
        raise ValueError(
            "The number of time points must be equal to the number of states"
        )

    number_of_points = len(time_points)
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

    # building the optimal control problem
    opti = ca.Opti()

    # create the variables with the polynom coefficients
    Px_coefs = opti.variable(pol_order, number_of_points - 1)
    Py_coefs = opti.variable(pol_order, number_of_points - 1)

    # define the cost function
    obj = 0
    for k in range(0, number_of_points - 1):
        # minimize the jerk
        obj += ca.power(Px_coefs[0, k], 2) + ca.power(Py_coefs[0, k], 2)

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

        # final postion constraints
        opti.subject_to(F_pos(Px_coefs[:, k], time_points[k + 1]) == states["x"][k + 1])
        opti.subject_to(F_pos(Py_coefs[:, k], time_points[k + 1]) == states["y"][k + 1])

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

            # acceleration constraints (continuity)
            opti.subject_to(
                F_acc(Px_coefs[:, k], time_points[k + 1])
                == F_acc(Px_coefs[:, k + 1], time_points[k + 1])
            )
            opti.subject_to(
                F_acc(Py_coefs[:, k], time_points[k + 1])
                == F_acc(Py_coefs[:, k + 1], time_points[k + 1])
            )

        # check if velocity and acceleration constraints were given
        if states["vx"][k] != None:
            opti.subject_to(F_vel(Px_coefs[:, k], time_points[k]) == states["vx"][k])
        if states["vy"][k] != None:
            opti.subject_to(F_vel(Py_coefs[:, k], time_points[k]) == states["vy"][k])
        if states["ax"][k] != None:
            opti.subject_to(F_acc(Px_coefs[:, k], time_points[k]) == states["ax"][k])
        if states["ay"][k] != None:
            opti.subject_to(F_acc(Py_coefs[:, k], time_points[k]) == states["ay"][k])

    # velocity and acceleration contraints at final point
    # check if velocity and acceleration constraints were given
    k += 1
    if states["vx"][k] != None:
        opti.subject_to(F_vel(Px_coefs[:, k - 1], time_points[k]) == states["vx"][k])
    if states["vy"][k] != None:
        opti.subject_to(F_vel(Py_coefs[:, k - 1], time_points[k]) == states["vy"][k])
    if states["ax"][k] != None:
        opti.subject_to(F_acc(Px_coefs[:, k - 1], time_points[k]) == states["ax"][k])
    if states["ay"][k] != None:
        opti.subject_to(F_acc(Py_coefs[:, k - 1], time_points[k]) == states["ay"][k])

    # check if the hopper is inside bounds
    for i in range(number_of_points - 1):
        dt = (time_points[i + 1] - time_points[i]) / 100
        for j in range(100):
            cur_time = time_points[i] + j * dt
            # if i <=1:
            #     # this is a specific constraint for the circle
            #     opti.subject_to(F_acc(Py_coefs[:, 0], cur_time) == 0)

            # v_norm = ca.sqrt(
            #     ca.power(F_vel(Px_coefs[:, i], cur_time), 2)
            #     + ca.power(F_vel(Py_coefs[:, i], cur_time), 2)
            # )

            # v = ca.vertcat(F_vel(Px_coefs[:, i], cur_time), F_vel(Py_coefs[:, i], cur_time))
            # v_norm = ca.norm_2(v)
            # e1bx = F_vel(Px_coefs[:, i], cur_time) / v_norm
            # e1by = F_vel(Py_coefs[:, i], cur_time) / v_norm

            # e1 = ca.vertcat(e1bx, e1by)

            # e1 = ca.vertcat(0, 1) # just using static constraints for now

            # a = ca.vertcat(F_acc(Px_coefs[:, i], cur_time), F_acc(Py_coefs[:, i], cur_time))

            # opti.subject_to(a > e1 * constraints["min_acc"] - ca.vertcat(0.5 * g, g))
            # opti.subject_to(a < e1 * constraints["max_acc"] - ca.vertcat(-0.5 * g, g))

            # opti.subject_to(a[0] > constraints["min_acc_x"])
            # opti.subject_to(a[0] < constraints["max_acc_x"])
            # opti.subject_to(a[1] > constraints["min_acc_y"])
            # opti.subject_to(a[1] < constraints["max_acc_y"])

    # select the desired solver
    # hide solution output
    opts = {"ipopt.print_level": 0, "print_time": 0}
    opti.solver("ipopt")  # , opts)

    opti.set_initial(Px_coefs, np.ones((pol_order, number_of_points - 1)))
    opti.set_initial(Py_coefs, np.ones((pol_order, number_of_points - 1)))

    sol = opti.solve()
    return (
        sol.value(Px_coefs),
        sol.value(Py_coefs),
    )


def get_traj_params(time_points, states, constraints, plot=False):
    # calculate trajectory
    (
        Px_coefs,
        Py_coefs,
    ) = calculate_trajectory(time_points, states, constraints)

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
    e1bx = np.zeros(n)
    e1by = np.zeros(n)
    e2bx = np.zeros(n)
    e2by = np.zeros(n)
    gamma = np.zeros(n)
    gamma_dot = np.zeros(n)

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

        v_norm = np.sqrt(vx[i] ** 2 + vy[i] ** 2)
        e1bx[i] = vx[i] / v_norm
        e1by[i] = vy[i] / v_norm
        [e2bx[i], e2by[i], _] = np.cross(
            np.array([0, 0, 1.0]), np.array([e1bx[i], e1by[i], 0])
        )

        gamma[i] = np.arctan2(e1by[i], e1bx[i])

        if i == 0:
            gamma_dot[i] = (gamma[i + 1] - gamma[i]) / (t[i + 1] - t[i])
        else:
            gamma_dot[i] = (gamma[i] - gamma[i - 1]) / (t[i] - t[i - 1])

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
        "e1bx": e1bx,
        "e1by": e1by,
        "e2bx": e2bx,
        "e2by": e2by,
        "gamma": gamma,
        "gamma_dot": gamma_dot,
    }

    if plot:
        plot_trajectory(
            time_points,
            states,
            Px_coefs,
            Py_coefs,
        )

    return trajectory


def plot_trajectory(
    time_points,
    states,
    Px_coefs,
    Py_coefs,
):
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))

    t = np.linspace(time_points[0], time_points[-1], 1001)
    x = np.zeros(len(t))
    y = np.zeros(len(t))
    vx = np.zeros(len(t))
    vy = np.zeros(len(t))
    ax = np.zeros(len(t))
    ay = np.zeros(len(t))
    e1bx = np.zeros(len(t))
    e1by = np.zeros(len(t))
    e2bx = np.zeros(len(t))
    e2by = np.zeros(len(t))
    last_t = t[0]

    for k in range(0, len(time_points) - 1):
        x[t >= time_points[k]] = get_pos(Px_coefs[:, k], t[t >= time_points[k]])
        y[t >= time_points[k]] = get_pos(Py_coefs[:, k], t[t >= time_points[k]])
        vx[t >= time_points[k]] = get_vel(Px_coefs[:, k], t[t >= time_points[k]])
        vy[t >= time_points[k]] = get_vel(Py_coefs[:, k], t[t >= time_points[k]])
        ax[t >= time_points[k]] = get_acc(Px_coefs[:, k], t[t >= time_points[k]])
        ay[t >= time_points[k]] = get_acc(Py_coefs[:, k], t[t >= time_points[k]])

        v_norm = np.sqrt(vx[t >= time_points[k]] ** 2 + vy[t >= time_points[k]] ** 2)
        e1bx[t >= time_points[k]] = vx[t >= time_points[k]] / v_norm
        e1by[t >= time_points[k]] = vy[t >= time_points[k]] / v_norm
        e2bx[t >= time_points[k]] = -e1by[t >= time_points[k]]
        e2by[t >= time_points[k]] = e1bx[t >= time_points[k]]

    gamma = np.arctan2(e1by, e1bx)
    gamma_dot = np.zeros(len(t))

    # calculate gamma_dot
    for i in range(0, len(t)):
        if i == 0:
            gamma_dot[i] = (gamma[i + 1] - gamma[i]) / (t[i + 1] - t[i])
        else:
            gamma_dot[i] = (gamma[i] - gamma[i - 1]) / (t[i] - t[i - 1])

    axs[0, 0].plot(x, y, label="trajectory")
    axs[0, 0].scatter([-100, 100], [0, 0], s=1e-3)
    axs[0, 0].plot(states["x"], states["y"], "o", label="target points")
    axs[0, 0].set_xlabel("Horizontal Position (m)")
    axs[0, 0].set_ylabel("Vertical Position (m)")
    axs[0, 0].set_title("Trajectory")
    axs[0, 0].set_aspect("equal")
    axs[0, 0].legend()
    axs[0, 0].grid()

    # make the arrows
    for k in range(0, len(time_points) - 1):
        temp = t[t >= time_points[k]]

        for i in range(len(temp)):
            if (
                temp[i] - last_t >= 5 and temp[i] < time_points[k + 1]
            ):  # Plot frame every 10 seconds
                ind = np.searchsorted(t, temp[i])
                origin_x = x[ind]
                origin_y = y[ind]
                e1_x = e1bx[ind]
                e1_y = e1by[ind]
                e2_x = e2bx[ind]
                e2_y = e2by[ind]
                axs[0, 0].quiver(origin_x, origin_y, e1_x, e1_y, scale=10, color="red")
                axs[0, 0].quiver(origin_x, origin_y, e2_x, e2_y, scale=10, color="blue")
                print(
                    "At time: {:.1f}, x: {:.1f}, y: {:.1f} e1x: {:.1f}, e1y: {:.1f}, e2x: {:.1f}, e2y: {:.1f}".format(
                        temp[i], origin_x, origin_y, e1_x, e1_y, e2_x, e2_y
                    )
                )
                last_t = temp[i]

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

    # axs[2, 1].plot(t, np.rad2deg(gamma_dot), label="$\\dot{\\gamma}$")
    axs[2, 1].plot(t, e1bx, label="$e_{1bx}$")
    axs[2, 1].plot(t, e1by, label="$e_{1by}$")
    axs[2, 1].plot(t, e2bx, label="$e_{2bx}$")
    axs[2, 1].plot(t, e2by, label="$e_{2by}$")
    axs[2, 1].set_xlabel("Time (s)")
    axs[2, 1].set_ylabel("Frame parameters (-)")
    axs[2, 1].set_title("Frame parameters vs Time")
    axs[2, 1].legend()
    axs[2, 1].grid()

    plt.show()
