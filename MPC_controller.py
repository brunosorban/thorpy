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
        normalization_params_x,
        normalization_params_u,
    ):
        # store the parameters
        self.env_params = env_params
        self.rocket_params = rocket_params
        self.controller_params = controller_params
        self.normalization_params_x = normalization_params_x
        self.normalization_params_u = normalization_params_u

        # rocket and environment parameters
        g = env_params["g"]
        m = rocket_params["m"]
        C = rocket_params["C"]
        K_tvc = rocket_params["K_tvc"]
        T_tvc = rocket_params["T_tvc"]
        l_tvc = rocket_params["l_tvc"]

        # controller parameters
        dt = controller_params["dt"]
        T = controller_params["T"]  # time hotizon
        N = controller_params["N"]  # Number of control intervals
        u_bounds = controller_params["u_bounds"]
        gamma_bounds = controller_params["gamma_bounds"]

        # state parameters
        t0_val = controller_params["t0"]
        x0_val = controller_params["x0"]
        x_target = controller_params["x_target"]

        Q = controller_params["Q"]
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

        # initializing the state varibles
        x = ca.SX.sym("x")  # states
        x_dot = ca.SX.sym("x_dot")  # states
        y = ca.SX.sym("y")  # states
        y_dot = ca.SX.sym("y_dot")  # states
        gamma = ca.SX.sym("gamma")  # states
        gamma_dot = ca.SX.sym("gamma_dot")  # states
        delta_tvc = ca.SX.sym("delta_tvc")  # states

        # create a vector with variable names
        variables = ca.vertcat(x, x_dot, y, y_dot, gamma, gamma_dot, delta_tvc)

        # initializing the control varibles
        thrust = ca.SX.sym("thrust")  # controls
        delta_tvc_ref = ca.SX.sym("delta_tvc_ref")  # controls

        # create a vector with variable names
        u = ca.vertcat(thrust, delta_tvc_ref)

        # define the nonlinear ode
        ode = ca.vertcat(
            x_dot,
            thrust * ca.cos(gamma - delta_tvc) / m,
            y_dot,
            thrust * ca.sin(gamma - delta_tvc) / m - g,
            gamma_dot,
            -ca.power(gamma_dot, 2) - thrust * l_tvc * ca.sin(delta_tvc) / C,
            1 / T_tvc * (K_tvc * delta_tvc_ref - delta_tvc),
        )

        # define the linear ode
        # ode = ca.vertcat(
        #     x_dot,
        #     -g * (gamma - self.x0_val[4]),
        #     y_dot,
        #     thrust / m,
        #     gamma_dot,
        #     delta_tvc_ref / C,
        # )

        # creating the ploblem statement function
        f = ca.Function(
            "f",
            [variables, u],
            [ode],
            [
                "[x, x_dot, y, y_dot, gamma, gamma_dot, delta_tvc]",
                "[f, delta_tvc_ref]",
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
                "[x, x_dot, y, y_dot, gamma, gamma_dot, delta_tvc]",
                "[f, delta_tvc_ref]",
            ],
            ["x_next"],
        )

        # building the optimal control problem
        self.opti = ca.Opti()

        # create the state, control and initial state varibles
        self.x = ca.repmat(
            ca.vertcat(*normalization_params_x), 1, N + 1
        ) * self.opti.variable(7, N + 1)

        self.u = ca.repmat(
            ca.vertcat(*normalization_params_u), 1, N
        ) * self.opti.variable(2, N)

        self.x_initial = self.opti.parameter(7, 1)
        self.x_target = self.opti.parameter(7, 1)

        # define the cost function
        self.obj = 0
        for k in range(N):
            self.obj += ca.mtimes(
                [(self.x[:, k] - self.x_target).T,
                 Q, (self.x[:, k] - self.x_target)]
            ) + ca.mtimes([self.u[:, k].T, R, self.u[:, k]])

        self.obj += ca.mtimes(
            [(self.x[:, N] - self.x_target).T, Q, (self.x[:, N] - self.x_target)]
        )
        self.opti.minimize(self.obj)

        # apply the constraint that the state is defined by my system       
        # set the initial position
        self.opti.subject_to(self.x[:, 0] == self.x_initial)
        
        # set the dynamics
        self.F = F
        for k in range(0, N):
            # apply the dynamics as constraints
            self.opti.subject_to(
                self.x[:, k + 1] == F(self.x[:, k], self.u[:, k]))

            # apply bounds to the inputs
            self.opti.subject_to(self.u[0, k] >= u_bounds[0][0])
            self.opti.subject_to(self.u[0, k] <= u_bounds[0][1])
            self.opti.subject_to(self.u[1, k] >= u_bounds[1][0])
            self.opti.subject_to(self.u[1, k] <= u_bounds[1][1])

            # # apply bounds to the states
            self.opti.subject_to(self.x[4, k] >= gamma_bounds[0])
            self.opti.subject_to(self.x[4, k] <= gamma_bounds[1])

        # set target
        self.opti.set_value(self.x_target, x_target)
        self.opti.set_value(self.x_initial, x0_val)

        # select the desired solver
        # hide solution output
        opts = {"ipopt.print_level": 0, "print_time": 0}
        self.opti.solver("ipopt" , opts)

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

            # add the cost payed on this solution
            self.J_total += sol.value(self.obj)
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

            if t[-1] == 0:
                sol = self.opti.solve()
            else:
                self.opti.set_initial(self.x, last_sol.value(self.x))
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
                    horizon[6][1],
                ]
            )
            state_horizon_list.append(horizon)
            control_horizon_list.append(u)
            thrust.append(u[0, 0])
            torque.append(u[1, 0])
            t.append(t[-1] + self.dt)
            last_sol = sol

            # plot the results
            # self.plot_horizon(t, x_list, u, horizon)
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
        ax1.plot(t, x[:, 1])
        ax1.plot(t, x[:, 2])
        ax1.plot(t, x[:, 3])
        ax1.legend(["$x$", "$v_x$", "$y$", "$v_y$"])
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Position (m)")
        ax1.set_title("Position vs Time")
        ax1.grid()

        f_list = u[0]
        delta_tvc_ref_list = u[1]

        # plot 2: u
        ax2.plot(t[0: len(t) - 1], f_list)
        ax2.legend(["$f$"])
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Thrust (N)")
        ax2.set_title("Thrust vs Time")
        ax2.grid()

        gamma_bounds = self.controller_params["gamma_bounds"]
        # plot 3: gamma
        ax3.plot(t, np.rad2deg(x[:, 4]))
        ax3.plot(t, [np.rad2deg(gamma_bounds[0])] * len(t), "--", color="black")
        ax3.plot(t, [np.rad2deg(gamma_bounds[1])] * len(t), "--", color="black")
        ax3.legend(["$\\gamma$", "$\\gamma_{min}$", "$\\gamma_{max}$"])
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Angle (degrees)")
        ax3.set_title("Angle vs Time")
        ax3.grid()

        u_bounds = self.controller_params["u_bounds"]
        # plot 4: u
        ax4.plot(t[0: len(t) - 1], np.rad2deg(delta_tvc_ref_list))
        ax4.plot(t, np.rad2deg(x[:, 6]))
        ax4.plot(t, [np.rad2deg(u_bounds[1][0])] * len(t), "--", color="black")
        ax4.plot(t, [np.rad2deg(u_bounds[1][1])] * len(t), "--", color="black")
        ax4.legend(["$\\delta_{tvc_{ref}}$", "$\\delta_{tvc}$", "$\\delta_{tvc_{min}}$", "$\\delta_{tvc_{max}}$"], loc='upper right')
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("$\\delta_{tvc}$ (degrees)")
        ax4.set_title("$\\delta_{tvc}$ vs Time")
        ax4.grid()

        plt.tight_layout()
        plt.show()

    def plot_horizon(self, t, x, u, horizon):
        N = self.controller_params["N"]
        t_list = np.linspace(t[-2], t[-2] + N * self.dt, N + 1, endpoint=True)
        horizon = np.array(horizon)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # plot 1: x, x_hor, and vx_hor
        ax1.plot(np.array(x)[:, 0], np.array(x)[:, 2])
        ax1.plot(horizon[0, :], horizon[2, :])
        # ax1.plot(horizon[1, :], horizon[3, :])
        ax1.legend(["$x$", "$x_{hor}$"])
        ax1.grid()

        f_list = u[0]
        delta_tvc_ref_list = u[1]

        # plot 2: u
        ax2.scatter(t_list[0: len(t_list) - 1], f_list)
        ax2.legend(["$f$"])
        ax2.grid()

        # plot 3: gamma, gamma_hor, and gamma_dot_hor
        ax3.plot(t, np.rad2deg(np.array(x)[:, 4]))
        ax3.plot(t_list, np.rad2deg(horizon[4, :]))
        # ax3.plot(t_list, horizon[5, :])
        ax3.legend(["$gamma$", "$gamma_{hor}$"])  # , "$\dot{gamma}_{hor}$"])
        ax3.grid()

        # plot 4: u
        ax4.scatter(t_list[0: len(t_list) - 1], np.rad2deg(delta_tvc_ref_list))
        ax4.scatter(t, np.rad2deg(np.array(x)[:, 6]))
        ax4.scatter(t_list, np.rad2deg(horizon[6, :]))
        ax4.legend(["$delta_{tvc_{ref}}$", "$delta_{tvc}$", "$delta_{tvc_{hor}}$"])
        ax4.grid()

        plt.show()
