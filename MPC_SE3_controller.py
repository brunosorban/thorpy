import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from RK4 import RK4


class MPC_controller:
    """
    Class to implement the MPC controller. The controller is implemented as a
    class so that the controller can be updated with a new target position.
    """

    def __init__(
        self,
        env_params,
        rocket_params,
        controller_params,
        trajectory_params=None,
    ):
        # store the parameters
        self.env_params = env_params
        self.rocket_params = rocket_params
        self.controller_params = controller_params
        self.trajectory_params = trajectory_params

        self.epos_list = [np.array([0, 0])]
        self.er_list = [0]

        if trajectory_params is not None:
            self.follow_trajectory = True

        # rocket and environment parameters
        g = env_params["g"]
        m = rocket_params["m"]
        J_z = rocket_params["J_z"]
        l_tvc = rocket_params["l_tvc"]

        # controller parameters
        dt = controller_params["dt"]
        T = controller_params["T"]  # time hotizon
        N = controller_params["N"]  # Number of control intervals

        thrust_bounds = controller_params["thrust_bounds"]
        delta_tvc_bounds = controller_params["delta_tvc_bounds"]
        thrust_dot_bounds = controller_params["thrust_dot_bounds"]
        delta_tvc_dot_bounds = controller_params["delta_tvc_dot_bounds"]

        # state parameters
        t0_val = controller_params["t0"]
        x0_val = controller_params["x0"]
        x_target = controller_params["x_target"]

        Q = controller_params["Q"]
        self.Q = Q
        Qf = controller_params["Qf"]
        R = controller_params["R"]

        # create variables used during operation
        self.dt = T / N
        self.x0_val = x0_val
        self.out = [0, 0]
        self.J_total = 0
        self.J_total_list = []
        self.horizon_list = []
        self.last_time_control = -self.dt
        self.thrust = []  # each line contains a control input
        self.torque = []  # each line contains a control input
        self.state_horizon_list = []
        self.control_horizon_list = []
        self.t = []

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
        omega_z = ca.SX.sym("omega")  # states
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
            omega_z,
            thrust,
        )

        # initializing the control varibles
        thrust_dot = ca.SX.sym("thrust_dot")  # controls
        delta_tvc_dot = ca.SX.sym("delta_tvc_dot")  # controls

        # create a vector with variable names
        u = ca.vertcat(thrust_dot, delta_tvc_dot)

        # define the nonlinear ode
        ode = ca.vertcat(
            x_dot,  # x
            thrust / m * e1tx,  # v_x
            y_dot,  # y
            thrust / m * e1ty - g,  # v_y
            omega_z * e2bx,  # e1bx
            omega_z * e2by,  # e1by
            -omega_z * e1bx,  # e2bx
            -omega_z * e1by,  # e2by
            (delta_tvc_dot + omega_z) * e2tx,  # e1tx
            (delta_tvc_dot + omega_z) * e2ty,  # e1ty
            -(delta_tvc_dot + omega_z) * e1tx,  # e2tx
            -(delta_tvc_dot + omega_z) * e1ty,  # e2ty
            -thrust * l_tvc * (e1tx * e2bx + e1ty * e2by) / J_z,  # omega
            thrust_dot,  # thrust
        )

        # creating the ploblem statement function
        f = ca.Function(
            "f",
            [variables, u],
            [ode],
            [
                "[x, x_dot, y, y_dot, e1bx, e1by, e2bx, e2by, e1tx, e1ty, e2tx, e2ty, omega, thrust]",
                "[thrust, delta_tvc_dot]",
            ],
            ["ode"],
        )

        # integrator
        x_next = RK4(f, variables, u, self.dt)

        F = ca.Function(
            "F",
            [variables, u],
            [x_next],
            [
                "[x, x_dot, y, y_dot, e1bx, e1by, e2bx, e2by, e1tx, e1ty, e2tx, e2ty, omega, thrust]",
                "[thrust, delta_tvc_dot]",
            ],
            ["x_next"],
        )

        # building the optimal control problem
        self.opti = ca.Opti()

        # create the state, control and initial state varibles
        self.x = self.opti.variable(14, N + 1)
        self.u = self.opti.variable(2, N)
        self.x_initial = self.opti.parameter(14, 1)
        self.x_target = self.opti.parameter(14, N + 1)

        # define the cost function
        self.obj = 0
        for k in range(N):
            # position and velocity tracking error
            tracking_error = self.x[0:4, k] - self.x_target[0:4, k]
            self.obj += ca.mtimes([tracking_error.T, Q[0:4, 0:4], tracking_error])

            # # rotation error
            er = (
                0.5 * self.x[4, k] * self.x_target[6, k]
                + 0.5 * self.x[5, k] * self.x_target[7, k]
                - 0.5 * self.x_target[4, k] * self.x[6, k]
                - 0.5 * self.x_target[5, k] * self.x[7, k]
            )

            self.obj += ca.mtimes([er, Q[4:5, 4:5], er])

            # angular velocity error
            self.obj += ca.mtimes([self.x[12, k], Q[5, 5], self.x[12, k]])

            # thrust error (almost 0, accounted on the control effort)
            self.obj += ca.mtimes([self.x[13, k], Q[6, 6], self.x[13, k]])

            # control effort
            self.obj += ca.mtimes([self.u[:, k].T, R, self.u[:, k]])

        # position and velocity tracking error
        tracking_error = self.x[0:4, N] - self.x_target[0:4, N]
        self.obj += ca.mtimes([tracking_error.T, Q[0:4, 0:4], tracking_error])

        # rotation error
        er = (
            0.5 * self.x[4, k] * self.x_target[6, k]
            + 0.5 * self.x[5, k] * self.x_target[7, k]
            - 0.5 * self.x_target[4, k] * self.x[6, k]
            - 0.5 * self.x_target[5, k] * self.x[7, k]
        )
        self.obj += ca.mtimes([er, Q[4:5, 4:5], er])

        # angular velocity error
        self.obj += ca.mtimes([self.x[12, k], Q[5, 5], self.x[12, k]])

        # thrust error (almost 0, accounted on the control effort)
        self.obj += ca.mtimes([self.x[13, k], Q[6, 6], self.x[13, k]])

        self.opti.minimize(self.obj)

        # apply the constraint that the state is defined by my system
        # set the initial position
        self.opti.subject_to(self.x[:, 0] == self.x_initial)

        # set the dynamics
        self.F = F
        for k in range(0, N + 1):
            if k < N:
                # apply the dynamics as constraints
                self.opti.subject_to(self.x[:, k + 1] == F(self.x[:, k], self.u[:, k]))

                # apply bounds to the inputs
                self.opti.subject_to(self.u[0, k] >= thrust_dot_bounds[0])
                self.opti.subject_to(self.u[0, k] <= thrust_dot_bounds[1])
                self.opti.subject_to(self.u[1, k] >= delta_tvc_dot_bounds[0])
                self.opti.subject_to(self.u[1, k] <= delta_tvc_dot_bounds[1])

            thrust = self.x[6, k]
            self.opti.subject_to(self.x[13, k] >= thrust_bounds[0])
            self.opti.subject_to(self.x[13, k] <= thrust_bounds[1])

            delta_tvc = self.x[7, k]
            self.opti.subject_to(self.x[7, k] >= delta_tvc_bounds[0])
            self.opti.subject_to(self.x[7, k] <= delta_tvc_bounds[1])

        # set target
        if self.follow_trajectory:
            self.update_traj(t0_val)
        else:
            self.opti.set_value(self.x_target, x_target)

        # set initial state
        self.opti.set_value(self.x_initial, x0_val)

        # select the desired solver
        # hide solution output
        opts = {"ipopt.print_level": 0, "print_time": 0}
        self.opti.solver("ipopt", opts)

    def calculate_Rd(self, gamma):
        """
        Calculates the rotation matrix from the body frame to the trajectory frame.

        Parameters:
            gamma (float): rotation angle.

        Returns:
            Rd (array-like): 3x3 rotation matrix.
        """
        Rd = np.array(
            [
                [np.cos(gamma), -np.sin(gamma), 0],
                [np.sin(gamma), np.cos(gamma), 0],
                [0, 0, 1],
            ]
        )
        return Rd

    def update_target(self, new_target):
        for k in range(len(new_target[0])):
            self.opti.set_value(self.x_target[:, k], ca.vertcat(*new_target[:, k]))

    def linear_spline(self, x, x_source, y_source):
        """Linear interpolation of x in x_source and y_source"""

        if x_source[0] <= x <= x_source[-1]:
            position = np.searchsorted(x_source, x)

        elif x > x_source[-1]:
            position = len(x_source) - 1

        else:
            position = 1

        dx = float(x_source[position] - x_source[position - 1])
        dy = float(y_source[position] - y_source[position - 1])

        return y_source[position - 1] + (dy / dx) * (x - x_source[position - 1])

    def update_traj(self, time):
        # trajectory_params = [x, x_dot, y, y_dot, gamma, gamma_dot, thrust, delta_tvc]
        px = self.trajectory_params["x"]
        py = self.trajectory_params["y"]
        vx = self.trajectory_params["vx"]
        vy = self.trajectory_params["vy"]
        e1bx = self.trajectory_params["e1bx"]
        e1by = self.trajectory_params["e1by"]
        e2bx = self.trajectory_params["e2bx"]
        e2by = self.trajectory_params["e2by"]
        e1tx = e1bx
        e1ty = e1by
        e2tx = e2bx
        e2ty = e2by

        N = self.controller_params["N"]
        t_list = np.linspace(time, time + N * self.dt, N + 1, endpoint=True)

        px_list = np.zeros(len(t_list))
        py_list = np.zeros(len(t_list))
        vx_list = np.zeros(len(t_list))
        vy_list = np.zeros(len(t_list))
        e1bx_list = np.zeros(len(t_list))
        e1by_list = np.zeros(len(t_list))
        e2bx_list = np.zeros(len(t_list))
        e2by_list = np.zeros(len(t_list))
        e1tx_list = np.zeros(len(t_list))
        e1ty_list = np.zeros(len(t_list))
        e2tx_list = np.zeros(len(t_list))
        e2ty_list = np.zeros(len(t_list))
        thrust_list = np.zeros(len(t_list))  # keep as 0

        for i in range(len(t_list)):
            px_list[i] = self.linear_spline(t_list[i], self.trajectory_params["t"], px)
            py_list[i] = self.linear_spline(t_list[i], self.trajectory_params["t"], py)
            vx_list[i] = self.linear_spline(t_list[i], self.trajectory_params["t"], vx)
            vy_list[i] = self.linear_spline(t_list[i], self.trajectory_params["t"], vy)
            e1bx_list[i] = self.linear_spline(
                t_list[i], self.trajectory_params["t"], e1bx
            )
            e1by_list[i] = self.linear_spline(
                t_list[i], self.trajectory_params["t"], e1by
            )
            e2bx_list[i] = self.linear_spline(
                t_list[i], self.trajectory_params["t"], e2bx
            )
            e2by_list[i] = self.linear_spline(
                t_list[i], self.trajectory_params["t"], e2by
            )
            e1tx_list[i] = self.linear_spline(
                t_list[i], self.trajectory_params["t"], e1tx
            )
            e1ty_list[i] = self.linear_spline(
                t_list[i], self.trajectory_params["t"], e1ty
            )
            e2tx_list[i] = self.linear_spline(
                t_list[i], self.trajectory_params["t"], e2tx
            )
            e2ty_list[i] = self.linear_spline(
                t_list[i], self.trajectory_params["t"], e2ty
            )

        target = np.array(
            [
                px_list,
                vx_list,
                py_list,
                vy_list,
                e1bx_list,
                e1by_list,
                e2bx_list,
                e2by_list,
                e1tx_list,
                e1ty_list,
                e2tx_list,
                e2ty_list,
                np.zeros_like(thrust_list),  # omega = angular velocity
                thrust_list,
            ]
        )
        self.update_target(target)

    def update(self, t, x):
        if t - self.last_time_control >= self.dt:  # update if time has reached
            # if in trajectory mode, update the target
            if self.follow_trajectory:
                self.update_traj(t)

            # update and solve the optimal control problem
            self.opti.set_value(self.x_initial, x)

            # solve the optimal control problem
            if t == 0:
                sol = self.opti.solve()
            else:
                # retrieve the results from the previous iteration
                self.opti.set_initial(self.x, self.state_horizon_list[-1])
                self.opti.set_initial(self.u, self.control_horizon_list[-1])
                sol = self.opti.solve()

            # retrieve the results
            u = sol.value(self.u)
            horizon = sol.value(self.x)

            self.out = np.array(sol.value(self.u))[:, 0]
            self.last_time_control = t  # update last time it was controlled

            # normalization of the angular parameters
            for i in range(4, 12, 2):
                for j in range(len(horizon)):
                    horizon[i][j] = horizon[i][j] / np.linalg.norm(
                        [horizon[i][j], horizon[i + 1][j]]
                    )
                    horizon[i + 1][j] = horizon[i + 1][j] / np.linalg.norm(
                        [horizon[i][j], horizon[i + 1][j]]
                    )

            # save the resutls
            self.state_horizon_list.append(horizon)
            self.control_horizon_list.append(u)
            self.thrust.append(u[0, 0])
            self.torque.append(u[1, 0])
            self.t.append(t + self.dt)

            # compute actual error
            px_target = self.linear_spline(
                t, self.trajectory_params["t"], self.trajectory_params["x"]
            )
            py_target = self.linear_spline(
                t, self.trajectory_params["t"], self.trajectory_params["y"]
            )

            pos_error = np.array([x[0], x[2]]) - np.array([px_target, py_target])
            self.epos_list.append(pos_error)

            e1bx_target = self.linear_spline(
                t, self.trajectory_params["t"], self.trajectory_params["e1bx"]
            )
            e1by_target = self.linear_spline(
                t, self.trajectory_params["t"], self.trajectory_params["e1by"]
            )
            e2bx_target = self.linear_spline(
                t, self.trajectory_params["t"], self.trajectory_params["e2bx"]
            )
            e2by_target = self.linear_spline(
                t, self.trajectory_params["t"], self.trajectory_params["e2by"]
            )

            er = (
                0.5 * x[4] * e2bx_target
                + 0.5 * x[5] * e2by_target
                - 0.5 * e1bx_target * x[6]
                - 0.5 * e1by_target * x[7]
            )
            self.er_list.append(er)

            # # add the cost payed on this solution
            # self.J_total += sol.value(self.obj)
            # self.J_total_list.append((t, self.J_total))

            print("t = ", t, "s", "    out = ", self.out)
        return self.out

    def simulate_inside(self, sim_time, plot_online=False):
        print("Starting simulation")
        t = [0]
        x_list = [np.squeeze(self.x0_val)]  # each line contains a state
        thrust = []  # each line contains a control input
        torque = []  # each line contains a control input
        state_horizon_list = []
        control_horizon_list = []

        if plot_online:
            # Turn on interactive mode
            # Initialize the figure and axes for the plot
            self.fig, (
                (self.ax1, self.ax2, self.ax3),
                (self.ax4, self.ax5, self.ax6),
                (self.ax7, self.ax8, self.ax9),
            ) = plt.subplots(3, 3, figsize=(15, 10))
            plt.ion()

        print("Solving...")
        while t[-1] < sim_time:
            # if in trajectory mode, update the target
            if self.follow_trajectory:
                self.update_traj(t[-1])

            # update and solve the optimal control problem
            self.opti.set_value(self.x_initial, x_list[-1])

            # solve the optimal control problem
            if t[-1] == 0:
                sol = self.opti.solve()
            else:
                # initial_guess = np.array(state_horizon_list)
                # print(initial_guess[-1])\
                self.opti.set_initial(self.x, state_horizon_list[-1])
                self.opti.set_initial(self.u, control_horizon_list[-1])
                sol = self.opti.solve()

            # retrieve the results
            u = sol.value(self.u)
            horizon = sol.value(self.x)

            # normalization of the angular parameters
            for i in range(4, 12, 2):
                for j in range(len(horizon)):
                    horizon[i][j] = horizon[i][j] / np.linalg.norm(
                        [horizon[i][j], horizon[i + 1][j]]
                    )
                    horizon[i + 1][j] = horizon[i + 1][j] / np.linalg.norm(
                        [horizon[i][j], horizon[i + 1][j]]
                    )

            # save the resutls
            x_list.append(
                [
                    horizon[0][1],
                    horizon[1][1],
                    horizon[2][1],
                    horizon[3][1],
                    horizon[4][1],
                    horizon[5][1],
                    horizon[6][1],
                    horizon[7][1],
                    horizon[8][1],
                    horizon[9][1],
                    horizon[10][1],
                    horizon[11][1],
                    horizon[12][1],
                    horizon[13][1],
                ]
            )

            state_horizon_list.append(horizon)
            control_horizon_list.append(u)
            thrust.append(u[0, 0])
            torque.append(u[1, 0])
            t.append(t[-1] + self.dt)

            # compute actual error
            px_target = self.linear_spline(
                t[-1], self.trajectory_params["t"], self.trajectory_params["x"]
            )
            py_target = self.linear_spline(
                t[-1], self.trajectory_params["t"], self.trajectory_params["y"]
            )

            pos_error = np.array([x_list[-1][0], x_list[-1][2]]) - np.array(
                [px_target, py_target]
            )
            self.epos_list.append(pos_error)

            e1bx_target = self.linear_spline(
                t[-1], self.trajectory_params["t"], self.trajectory_params["e1bx"]
            )
            e1by_target = self.linear_spline(
                t[-1], self.trajectory_params["t"], self.trajectory_params["e1by"]
            )
            e2bx_target = self.linear_spline(
                t[-1], self.trajectory_params["t"], self.trajectory_params["e2bx"]
            )
            e2by_target = self.linear_spline(
                t[-1], self.trajectory_params["t"], self.trajectory_params["e2by"]
            )

            er = (
                0.5 * x_list[-1][4] * e2bx_target
                + 0.5 * x_list[-1][5] * e2by_target
                - 0.5 * e1bx_target * x_list[-1][6]
                - 0.5 * e1by_target * x_list[-1][7]
            )
            self.er_list.append(er)

            if plot_online and t[-1] > self.dt:
                self.plot_horizon_online(
                    t, np.array(x_list), [thrust, torque], u, horizon
                )

        print("Simulation finished")

        return (
            np.array(t),
            np.array(x_list),
            np.array([thrust, torque]),
            np.array(state_horizon_list),
            np.array(control_horizon_list),
        )

    def plot_simulation(self, t, x, u):
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 12))

        # plot 1: x, and y
        ax1.plot(t, x[:, 0])
        ax1.plot(t, x[:, 1])
        ax1.plot(t, x[:, 2])
        ax1.plot(t, x[:, 3])
        ax1.legend(["$x$", "$v_x$", "$y$", "$v_y$"])
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Position (m)")
        ax1.set_title("Position vs Time")
        ax1.grid()

        thrust_bounds = self.controller_params["thrust_bounds"]
        delta_tvc_bounds = self.controller_params["delta_tvc_bounds"]

        gamma = np.angle(x[:, 4] + 1j * x[:, 5])
        tvc_angle = np.angle(x[:, 8] + 1j * x[:, 9]) - gamma

        # plot 2: thrust
        ax2.plot(t, x[:, 13])
        ax2.plot(t, [thrust_bounds[0]] * len(t), "--", color="black")
        ax2.plot(t, [thrust_bounds[1]] * len(t), "--", color="black")
        ax2.legend(["$f$", "$f_{min}$", "$f_{max}$"])
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Thrust (N)")
        ax2.set_title("Thrust vs Time")
        ax2.grid()

        # plot 3: epos
        self.epos_list = np.array(self.epos_list)
        ax3.plot(t, self.epos_list[:, 0], label="$e_{x}$")
        ax3.plot(t, self.epos_list[:, 1], label="$e_{y}$")
        ax3.legend()
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Position error")
        ax3.set_title("Position error vs Time")
        ax3.grid()

        # plot 4: gamma
        ax4.plot(t, np.rad2deg(gamma), label="$\\gamma$")
        ax4.legend()
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Angle (degrees)")
        ax4.set_title("Angle vs Time")
        ax4.grid()

        # plot 5: delta_tvc
        ax5.plot(t, np.rad2deg(tvc_angle))
        ax5.plot(t, [np.rad2deg(delta_tvc_bounds[0])] * len(t), "--", color="black")
        ax5.plot(t, [np.rad2deg(delta_tvc_bounds[1])] * len(t), "--", color="black")
        ax5.legend(
            ["$\\delta_{tvc}$", "$\\delta_{tvc_{min}}$", "$\\delta_{tvc_{max}}$"],
            loc="upper right",
        )
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("$\\delta_{tvc}$ (degrees)")
        ax5.set_title("$\\delta_{tvc}$ vs Time")
        ax5.grid()

        # plot 5: er
        ax6.plot(t, self.er_list, label="$e_{r}$")
        ax6.legend()
        ax6.set_xlabel("Time (s)")
        ax6.set_ylabel("Anglular error")
        ax6.set_title("Angular error vs Time")
        ax6.grid()

        plt.tight_layout()
        fig.show()

    def plot_tracking_results(self, t, x):
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 12))
        gamma = np.angle(x[:, 4] + 1j * x[:, 5])

        ax1.plot(t, x[:, 0], label="x")
        ax1.plot(
            self.trajectory_params["t"], self.trajectory_params["x"], label="x_ref"
        )
        ax1.legend()
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("X position (m)")
        ax1.set_title("X position vs Time")
        ax1.set_xlim([t[0], t[-1]])
        ax1.grid()

        ax2.plot(t, x[:, 2], label="y")
        ax2.plot(
            self.trajectory_params["t"], self.trajectory_params["y"], label="y_ref"
        )
        ax2.legend()
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Y position (m)")
        ax2.set_title("Y position vs Time")
        ax2.set_xlim([t[0], t[-1]])
        ax2.grid()

        ax3.plot(t, np.rad2deg(gamma), label="gamma")
        ax3.plot(
            self.trajectory_params["t"],
            np.rad2deg(
                np.arctan2(
                    self.trajectory_params["e1by"], self.trajectory_params["e1bx"]
                )
            ),
            label="gamma_ref",
        )
        ax3.legend()
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Gamma angle (deg)")
        ax3.set_title("Gamma angle vs Time")
        ax3.set_xlim([t[0], t[-1]])
        ax3.grid()

        ax4.plot(t, x[:, 1], label="vx")
        ax4.plot(
            self.trajectory_params["t"], self.trajectory_params["vx"], label="vx_ref"
        )
        ax4.legend()
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("X velocity (m/s)")
        ax4.set_title("X velocity vs Time")
        ax4.set_xlim([t[0], t[-1]])
        ax4.grid()

        ax5.plot(t, x[:, 3], label="vy")
        ax5.plot(
            self.trajectory_params["t"], self.trajectory_params["vy"], label="vy_ref"
        )
        ax5.legend()
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Y velocity (m/s)")
        ax5.set_title("Y velocity vs Time")
        ax5.set_xlim([t[0], t[-1]])
        ax5.grid()

        ax6.plot(t, x[:, 4], label="$e_{1bx}$")
        ax6.plot(t, x[:, 5], label="$e_{1by}$")
        ax6.plot(t, x[:, 6], label="$e_{2bx}$")
        ax6.plot(t, x[:, 7], label="$e_{2by}$")
        ax6.legend()
        ax6.set_xlabel("Time (s)")
        ax6.set_ylabel("Orientation parameters (-)")
        ax6.set_title("Orientation parameters vs Time")
        ax6.set_xlim([t[0], t[-1]])
        ax6.grid()

        plt.tight_layout()
        fig.show()

    def plot_horizon_online(self, t, x, last_u, u, horizon):
        # Clear the previous plot
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.ax5.clear()
        self.ax6.clear()
        self.ax7.clear()
        self.ax8.clear()
        self.ax9.clear()

        N = self.controller_params["N"]
        t_list = np.linspace(t[-2], t[-2] + N * self.dt, N + 1, endpoint=True)
        t_full = np.concatenate((t, t_list))
        horizon = np.array(horizon)

        thrust_deriv_list = u[0]
        delta_tvc_dot_list = u[1]
        thrust_dot_bounds = self.controller_params["thrust_dot_bounds"]
        delta_tvc_dot_bounds = self.controller_params["delta_tvc_dot_bounds"]
        thrust_bounds = self.controller_params["thrust_bounds"]
        delta_tvc_bounds = self.controller_params["delta_tvc_bounds"]
        last_thrust_deriv_list = last_u[0]
        last_delta_tvc_dot_list = last_u[1]

        # Plot the new data
        self.ax1.scatter(np.array(x)[:, 0], np.array(x)[:, 2], s=5)
        self.ax1.scatter(horizon[0, :], horizon[2, :], s=5)
        self.ax1.legend(["$xy$", "$xy_{hor}$"])
        self.ax1.set_xlabel("x position (m)")
        self.ax1.set_ylabel("y position (m)")
        self.ax1.set_title("Position of the hopper")
        self.ax1.axis("equal")
        self.ax1.grid()

        self.ax2.scatter(t, np.array(x)[:, 13], s=5)
        self.ax2.scatter(t_list, horizon[13, :], s=5)
        self.ax2.plot(t_full, [thrust_bounds[0]] * len(t_full), "--", color="black")
        self.ax2.plot(t_full, [thrust_bounds[1]] * len(t_full), "--", color="black")
        self.ax2.legend(["$f$", "$f_{ref}$"])
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("Thrust (N)")
        self.ax2.set_title("Thrust vs Time")
        self.ax2.grid()

        self.ax3.scatter(t[: len(t) - 1], last_thrust_deriv_list, s=5)
        self.ax3.scatter(t_list[: len(t_list) - 1], thrust_deriv_list, s=5)
        self.ax3.plot(t_full, [thrust_dot_bounds[0]] * len(t_full), "--", color="black")
        self.ax3.plot(t_full, [thrust_dot_bounds[1]] * len(t_full), "--", color="black")
        self.ax3.legend(["$\\dot{f}$", "$\\dot{f}_{ref}$"])
        self.ax3.set_xlabel("Time (s)")
        self.ax3.set_ylabel("Thrust derivative (N/s)")
        self.ax3.set_title("Thrust derivative vs Time")
        self.ax3.grid()

        gamma = np.arctan2(np.array(x)[:, 5], np.array(x)[:, 4])
        gamma_horizon = np.arctan2(horizon[5, :], horizon[4, :])
        self.ax4.scatter(t, np.rad2deg(gamma), s=5)
        self.ax4.scatter(t_list, np.rad2deg(gamma_horizon), s=5)
        self.ax4.legend(
            ["$\\gamma$", "$\\gamma_{hor}$"]  # , "$\\gamma_{min}$", "$\\gamma_{max}$"]
        )
        self.ax4.set_xlabel("Time (s)")
        self.ax4.set_ylabel("Gamma (degrees)")
        self.ax4.set_title("Gamma vs Time")
        self.ax4.grid()

        delta_tvc = np.arctan2(np.array(x)[:, 9], np.array(x)[:, 8]) - gamma
        delta_tvc_horizon = np.arctan2(horizon[9, :], horizon[8, :]) - gamma_horizon
        self.ax5.scatter(t, np.rad2deg(delta_tvc), s=5)
        self.ax5.scatter(t_list, np.rad2deg(delta_tvc_horizon), s=5)
        self.ax5.plot(
            t_full, [np.rad2deg(delta_tvc_bounds[0])] * len(t_full), "--", color="black"
        )
        self.ax5.plot(
            t_full, [np.rad2deg(delta_tvc_bounds[1])] * len(t_full), "--", color="black"
        )
        self.ax5.legend(
            [
                "$\\delta_{tvc}$",
                "$\\delta_{tvc_{horizon}}$",
            ],
            loc="upper right",
        )
        self.ax5.set_xlabel("Time (s)")
        self.ax5.set_ylabel("$\\delta_{tvc}$ (degrees)")
        self.ax5.set_title("$\\delta_{tvc}$ vs Time")
        self.ax5.grid()

        self.ax6.scatter(t[: len(t) - 1], last_delta_tvc_dot_list, s=5)
        self.ax6.scatter(t_list[: len(t_list) - 1], delta_tvc_dot_list, s=5)
        self.ax6.plot(
            t_full, [delta_tvc_dot_bounds[0]] * len(t_full), "--", color="black"
        )
        self.ax6.plot(
            t_full, [delta_tvc_dot_bounds[1]] * len(t_full), "--", color="black"
        )
        self.ax6.legend(
            [
                "$\\dot{\\delta_{tvc}}$",
                "$\\dot{\\delta_{tvc}}_{horizon}$",
            ],
            loc="upper right",
        )
        self.ax6.set_xlabel("Time (s)")
        self.ax6.set_ylabel("$\\dot{\\delta_{tvc}}$ (rad/s)")
        self.ax6.set_title("$\\dot{\\delta_{tvc}}$ vs Time")
        self.ax6.grid()

        self.ax7.scatter(t, np.array(x)[:, 5], s=5)
        self.ax7.scatter(t_list, horizon[5, :], s=5)
        self.ax7.plot(t_full, [1] * len(t_full), "--", color="black")
        self.ax7.plot(t_full, [-1] * len(t_full), "--", color="black")
        self.ax7.legend(["$e_{1bx} / e_{2by}$", "$e_{1bx_hor}$"])
        self.ax7.set_xlabel("Time (s)")
        self.ax7.set_ylabel("Orientation parameter (-)")
        self.ax7.set_title("Orientation parameter vs Time")
        self.ax7.grid()

        self.ax8.scatter(t, np.array(x)[:, 6], s=5)
        self.ax8.scatter(t_list, horizon[6, :], s=5)
        self.ax8.plot(t_full, [1] * len(t_full), "--", color="black")
        self.ax8.plot(t_full, [-1] * len(t_full), "--", color="black")
        self.ax8.legend(["$e_{1by}$", "$e_{1by_hor}$"])
        self.ax8.set_xlabel("Time (s)")
        self.ax8.set_ylabel("Orientation parameter (-)")
        self.ax8.set_title("Orientation parameter vs Time")
        self.ax8.grid()

        self.ax9.scatter(t, np.array(x)[:, 7], s=5)
        self.ax9.scatter(t_list, horizon[7, :], s=5)
        self.ax9.plot(t_full, [1] * len(t_full), "--", color="black")
        self.ax9.plot(t_full, [-1] * len(t_full), "--", color="black")
        self.ax9.legend(["$e_{2bx}$", "$e_{2bx_hor}$"])
        self.ax9.set_xlabel("Time (s)")
        self.ax9.set_ylabel("Orientation parameter (-)")
        self.ax9.set_title("Orientation parameter vs Time")
        self.ax9.grid()

        # Redraw the plot and flush the events
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Display the plot
        self.fig.show()
