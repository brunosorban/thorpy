import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from RK4 import RK4


class MPC_controller:
    """
    Class to implement the MPC controller. The controller is implemented as a
    class so that the controller can be updated with a new target position.
    """

    def __init__(self, m, C, g, T, N, u_bounds, t0, x0, x_target, Q, Qf, r):
        # rocket and environment parameters
        self.g = g
        self.m = m
        self.C = C

        # controller parameters
        self.dt = T / N
        self.T = T  # time hotizon
        self.N = N  # Number of control intervals
        self.u_bounds = u_bounds
        self.out = [0, 0]
        self.J_total = 0
        self.J_total_list = []
        self.horizon_list = []

        # state parameters
        self.t0_val = t0
        self.last_time_control = -self.dt
        self.x0_val = x0
        self.x_target = x_target

        self.Q = Q
        self.Qf = Qf
        self.r = r

        # initializing the state varibles
        x = ca.SX.sym("x")  # states
        x_dot = ca.SX.sym("x_dot")  # states
        y = ca.SX.sym("y")  # states
        y_dot = ca.SX.sym("y_dot")  # states
        gamma = ca.SX.sym("gamma")  # states
        gamma_dot = ca.SX.sym("gamma_dot")  # states

        # create a vector with variable names
        variables = ca.vertcat(x, x_dot, y, y_dot, gamma, gamma_dot)

        # initializing the control varibles
        thrust = ca.SX.sym("thrust")  # controls
        tau = ca.SX.sym("tau")  # controls

        # create a vector with variable names
        u = ca.vertcat(thrust, tau)

        # define the nonlinear ode
        ode = ca.vertcat(
            x_dot,
            thrust * ca.cos(gamma) / m,
            y_dot,
            thrust * ca.sin(gamma) / m - g,
            gamma_dot,
            -ca.power(gamma_dot, 2) + tau / C,
        )

        # define the linear ode
        # ode = ca.vertcat(
        #     x_dot,
        #     -g * (gamma - self.x0_val[4]),
        #     y_dot,
        #     thrust / m,
        #     gamma_dot,
        #     tau / C,
        # )

        # # DAE problem structure
        # dae = {"x": variables, "p": u, "ode": ode}

        # self.intg = ca.integrator(
        #     "intg",
        #     "rk",
        #     dae,
        #     t0,
        #     self.dt,
        #     # intg_options,
        # )

        # creating the ploblem statement function
        f = ca.Function(
            "f",
            [variables, u],
            [ode],
            ["[x, x_dot, y, y_dot, gamma, gamma_dot]", "[f, tau]"],
            ["ode"],
        )

        # integrator
        x_next = RK4(f, variables, u, self.dt)

        F = ca.Function(
            "F",
            [variables, u],
            [x_next],
            ["[x, x_dot, y, y_dot, gamma, gamma_dot]", "[f, tau]"],
            ["x_next"],
        )

        # building the optimal control problem
        self.opti = ca.Opti()

        # create the state, control and initial state varibles
        self.x = self.opti.variable(6, N + 1)
        self.u = self.opti.variable(2, N)
        self.x_initial = self.opti.parameter(6, 1)
        self.x_target = self.opti.parameter(6, 1)

        # define the cost function
        self.obj = 0
        for k in range(N):
            self.obj += ca.mtimes(
                [(self.x[:, k] - self.x_target).T, Q, (self.x[:, k] - self.x_target)]
            )

        self.obj += r * ca.sumsqr(self.u)
        self.obj += ca.mtimes(
            [(self.x[:, N] - self.x_target).T, Q, (self.x[:, N] - self.x_target)]
        )
        self.opti.minimize(self.obj)

        # apply the constraint that the state is defined by my linear system
        self.F = F
        for k in range(0, N):
            self.opti.subject_to(self.x[:, k + 1] == F(self.x[:, k], self.u[:, k]))
            # self.opti.subject_to(self.x[:, k + 1] == self.u[0, k])

        # apply bounds to the inputs
        # self.opti.subject_to(self.u >= u_min)
        # self.opti.subject_to(self.u <= u_max)

        # set the initial position
        self.opti.subject_to(self.x[:, 0] == self.x_initial)

        # set target
        self.opti.set_value(self.x_target, x_target)
        self.opti.set_value(self.x_initial, x0)

        # select the desired solver
        opts = {"ipopt.print_level": 0, "print_time": 0}  # hide solution output
        self.opti.solver("ipopt", opts)

    def update_target(self, new_target):
        self.opti.set_value(self.x_target, ca.vertcat(*new_target))

    def update(self, t, x):
        if t - self.last_time_control >= self.dt:  # update if time has reached
            # Calculate the control output
            self.current_state = x  # store current state

            self.opti.set_value(self.x_initial, x)

            # solve the optimal control problem
            sol = self.opti.solve()
            self.horizon_list.append((t, sol.value(self.x)))
            # print(sol.value(self.u))

            self.out = np.array(sol.value(self.u))[:, 0]

            self.J_total += sol.value(self.obj)  # add the cost payed on this solution
            self.J_total_list.append((t, self.J_total))
            self.last_time_control = t  # update last time it was controlled
        print("Controller output =", self.out)
        return self.out

    def simulate_inside(self, sim_time):
        print("Starting simulation")
        t = [0]
        x_list = [np.squeeze(self.x0_val)]  # each line contains a state
        thrust = []  # each line contains a control input
        torque = []  # each line contains a control input
        state_horizon_list = []
        control_horizon_list = []

        while t[-1] < sim_time:
            # update and solve the optimal control problem
            self.opti.set_value(self.x_initial, x_list[-1])
            sol = self.opti.solve()

            # retrieve the results
            u = sol.value(self.u)
            horizon = sol.value(self.x)

            # save the resutls
            x_list.append(
                [
                    horizon[0][1],
                    horizon[1][1],
                    horizon[2][1],
                    horizon[3][1],
                    horizon[4][1],
                    horizon[5][1],
                ]
            )
            state_horizon_list.append(horizon)
            control_horizon_list.append(u)
            thrust.append(u[0, 0])
            torque.append(u[1, 0])
            t.append(t[-1] + self.dt)

            # plot the results
            # self.plot_horizon(t, x, u, horizon)
            # input("Press Enter to continue...")
        print("Simulation finished")
        return (
            np.array(t),
            np.array(x_list),
            np.array([thrust, torque]),
            np.array(state_horizon_list),
            np.array(control_horizon_list),
        )

    def plot_simulation(self, t, x, u):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 4))

        # plot 1: x, and y
        ax1.plot(t, x[:, 0])
        ax1.plot(t, x[:, 2])
        ax1.legend(["$x$", "$y$"])
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Position (m)")
        ax1.set_title("Position vs Time")
        ax1.grid()

        f_list = u[0]
        tau_list = u[1]

        # plot 2: u
        ax2.plot(t[0 : len(t) - 1], f_list)
        ax2.legend(["$f$"])
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Thrust (N)")
        ax2.set_title("Thrust vs Time")
        ax2.grid()

        # plot 3: gamma
        ax3.plot(t, np.rad2deg(x[:, 4]))
        ax3.legend(["$\\gamma$"])
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Angle (degrees)")
        ax3.set_title("Angle vs Time")
        ax3.grid()

        # plot 4: u
        ax4.plot(t[0 : len(t) - 1], tau_list)
        ax4.legend(["$\\tau$"])
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Torque (Nm)")
        ax4.set_title("Torque vs Time")
        ax4.grid()

        plt.tight_layout()
        plt.show()

    def plot_horizon(self, t, x, u, horizon):
        t_list = np.linspace(t[-2], t[-2] + self.N * self.dt, self.N + 1, endpoint=True)
        horizon = np.array(horizon)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 4))

        # plot 1: x, x_hor, and vx_hor
        ax1.plot(np.array(x)[:, 0], np.array(x)[:, 2])
        ax1.plot(horizon[0, :], horizon[2, :])
        # ax1.plot(horizon[1, :], horizon[3, :])
        ax1.legend(["$x$", "$x_{hor}$"])
        ax1.grid()

        f_list = u[0]
        tau_list = u[1]

        # plot 2: u
        ax2.step(t_list[0 : len(t_list) - 1], f_list)
        ax2.legend(["$f$"])
        ax2.grid()

        # plot 3: gamma, gamma_hor, and gamma_dot_hor
        ax3.plot(t, np.rad2deg(np.array(x)[:, 4]))
        ax3.plot(t_list, np.rad2deg(horizon[4, :]))
        # ax3.plot(t_list, horizon[5, :])
        ax3.legend(["$gamma$", "$gamma_{hor}$"])  # , "$\dot{gamma}_{hor}$"])
        ax3.grid()

        # plot 4: u
        ax4.step(t_list[0 : len(t_list) - 1], tau_list)
        ax4.legend(["$tau$"])
        ax4.grid()

        plt.show()
