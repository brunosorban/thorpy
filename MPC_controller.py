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
        trajectory_params=None
    ):
        # store the parameters
        self.env_params = env_params
        self.rocket_params = rocket_params
        self.controller_params = controller_params
        self.normalization_params_x = normalization_params_x
        self.normalization_params_u = normalization_params_u
        self.trajectory_params = trajectory_params
        
        if trajectory_params is not None:
            self.follow_trajectory = True

        # rocket and environment parameters
        g = env_params["g"]
        m = rocket_params["m"]
        C = rocket_params["C"]
        K_tvc = rocket_params["K_tvc"]
        T_tvc = rocket_params["T_tvc"]
        l_tvc = rocket_params["l_tvc"]
        K_thrust = rocket_params["K_thrust"]
        T_thrust = rocket_params["T_thrust"]
        
        # controller parameters
        dt = controller_params["dt"]
        T = controller_params["T"]  # time hotizon
        N = controller_params["N"]  # Number of control intervals
        u_bounds = controller_params["u_bounds"]
        gamma_bounds = controller_params["gamma_bounds"]
        thrust_bounds = controller_params["thrust_bounds"]
        delta_tvc_bounds = controller_params["delta_tvc_bounds"]

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
        thrust = ca.SX.sym("thrust")  # states
        delta_tvc = ca.SX.sym("delta_tvc")  # states

        # create a vector with variable names
        variables = ca.vertcat(x, x_dot, y, y_dot, gamma, gamma_dot, thrust, delta_tvc)

        # initializing the control varibles
        thrust_dot = ca.SX.sym("thrust_dot")  # controls
        delta_tvc_dot = ca.SX.sym("delta_tvc_dot")  # controls

        # create a vector with variable names
        u = ca.vertcat(thrust_dot, delta_tvc_dot)

        # define the nonlinear ode
        ode = ca.vertcat(
            x_dot,
            thrust * ca.cos(gamma - delta_tvc) / m,
            y_dot,
            thrust * ca.sin(gamma - delta_tvc) / m - g,
            gamma_dot,
            -ca.power(gamma_dot, 2) - thrust * l_tvc * ca.sin(delta_tvc) / C,
            thrust_dot,
            delta_tvc_dot,
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
                "[x, x_dot, y, y_dot, gamma, gamma_dot, delta_tvc, thrust]",
                "[thrust, delta_tvc_ref]",
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
                "[x, x_dot, y, y_dot, gamma, gamma_dot, delta_tvc, thrust]",
                "[thrust, delta_tvc_ref]",
            ],
            ["x_next"],
        )

        # building the optimal control problem
        self.opti = ca.Opti()

        # create the state, control and initial state varibles
        self.x = ca.repmat(
            ca.vertcat(*normalization_params_x), 1, N + 1
        ) * self.opti.variable(8, N + 1)

        self.u = ca.repmat(
            ca.vertcat(*normalization_params_u), 1, N
        ) * self.opti.variable(2, N)

        self.x_initial = self.opti.parameter(8, 1)
        self.x_target = self.opti.parameter(8, 1)

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
        for k in range(0, N+1):
            if k < N:
                # apply the dynamics as constraints
                self.opti.subject_to(
                    self.x[:, k + 1] == F(self.x[:, k], self.u[:, k]))

                # apply bounds to the inputs
                self.opti.subject_to(self.u[0, k] >= u_bounds[0][0])
                self.opti.subject_to(self.u[0, k] <= u_bounds[0][1])
                self.opti.subject_to(self.u[1, k] >= u_bounds[1][0])
                self.opti.subject_to(self.u[1, k] <= u_bounds[1][1])

            # # apply bounds to the states
            # gamma = self.x[4, k]
            self.opti.subject_to(self.x[4, k] >= gamma_bounds[0])
            self.opti.subject_to(self.x[4, k] <= gamma_bounds[1])
            
            # thrust = self.x[6, k]
            self.opti.subject_to(self.x[6, k] >= thrust_bounds[0])
            self.opti.subject_to(self.x[6, k] <= thrust_bounds[1])
            
            # delta_tvc = self.x[7, k]
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
    
    def update_traj(self, time):
        # [x, x_dot, y, y_dot, gamma, gamma_dot, thrust, delta_tvc]
        px = self.trajectory_params["x"]
        py = self.trajectory_params["y"]
        vx = self.trajectory_params["vx"]
        vy = self.trajectory_params["vy"]
        
        t_index = np.searchsorted(self.trajectory_params["t"], time)
        
        if t_index >= len(self.trajectory_params["t"]):
            t_index = len(self.trajectory_params["t"]) - 1
            
        target = np.array([px[t_index], vx[t_index], py[t_index], vy[t_index], 0, 0, 0, 0])
        self.update_target(target)

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
            self.fig, ((self.ax1, self.ax2, self.ax3), (self.ax4, self.ax5, self.ax6)) = plt.subplots(2, 3, figsize=(15, 10))
            plt.ion()
        
        while t[-1] < sim_time:
            # if in trajectory mode, update the target
            if self.follow_trajectory:
                self.update_traj(t[-1])
                
            # update and solve the optimal control problem
            self.opti.set_value(self.x_initial, x_list[-1])

            if t[-1] == 0:
                sol = self.opti.solve()
            else:
                self.opti.set_initial(self.x, last_sol.value(self.x))
                self.opti.set_initial(self.u, last_sol.value(self.u))
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
                    horizon[7][1],
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
            
            if plot_online and t[-1] > self.dt:
                self.plot_horizon_online(t, np.array(x_list), [thrust, torque], u, horizon)
                
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

        thrust_deriv_list = u[0]
        delta_tvc_ref_deriv_list = u[1]
        u_bounds = self.controller_params["u_bounds"]
        gamma_bounds = self.controller_params["gamma_bounds"]
        thrust_bounds = self.controller_params["thrust_bounds"]
        delta_tvc_bounds = self.controller_params["delta_tvc_bounds"]


        # plot 2: thrust
        ax2.plot(t, x[:, 6])
        ax2.plot(t, [thrust_bounds[0]] * len(t), "--", color="black")
        ax2.plot(t, [thrust_bounds[1]] * len(t), "--", color="black")
        ax2.legend(["$f$", "$f_{min}$", "$f_{max}$"])
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Thrust (N)")
        ax2.set_title("Thrust vs Time")
        ax2.grid()

        # plot 3: thrust derivative
        ax3.plot(t[0: len(t) - 1], thrust_deriv_list)
        ax3.plot(t, [u_bounds[0][0]] * len(t), "--", color="black")
        ax3.plot(t, [u_bounds[0][1]] * len(t), "--", color="black")
        ax3.legend(["$\dot{f}$", "$\dot{f}_{ref}$", "$\dot{f}_{max}$"])
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Thrust derivative (N/s)")
        ax3.set_title("Thrust derivative vs Time")
        ax3.grid()
        
        # plot 4: gamma
        ax4.plot(t, np.rad2deg(x[:, 4]))
        ax4.plot(t, [np.rad2deg(gamma_bounds[0])] * len(t), "--", color="black")
        ax4.plot(t, [np.rad2deg(gamma_bounds[1])] * len(t), "--", color="black")
        ax4.legend(["$\\gamma$", "$\\gamma_{min}$", "$\\gamma_{max}$"])
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Angle (degrees)")
        ax4.set_title("Angle vs Time")
        ax4.grid()

        # plot 5: delta_tvc
        ax5.plot(t, np.rad2deg(x[:, 7]))
        ax5.plot(t, [np.rad2deg(delta_tvc_bounds[0])] * len(t), "--", color="black")
        ax5.plot(t, [np.rad2deg(delta_tvc_bounds[1])] * len(t), "--", color="black")
        ax5.legend(["$\\delta_{tvc}$", "$\\delta_{tvc_{min}}$", "$\\delta_{tvc_{max}}$"], loc='upper right')
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("$\\delta_{tvc}$ (degrees)")
        ax5.set_title("$\\delta_{tvc}$ vs Time")
        ax5.grid()
        
        # plot 5: delta_tvc
        ax6.plot(t[0: len(t) - 1], np.rad2deg(delta_tvc_ref_deriv_list))
        ax6.plot(t, [np.rad2deg(u_bounds[1][0])] * len(t), "--", color="black")
        ax6.plot(t, [np.rad2deg(u_bounds[1][1])] * len(t), "--", color="black")
        ax6.legend(["$\\dot{\\delta}_{tvc}$", "$\\dot{\\delta}_{tvc_{min}}$", "$\\dot{\\delta}_{tvc_{max}}$"], loc='upper right')
        ax6.set_xlabel("Time (s)")
        ax6.set_ylabel("$\\dot{\\delta}_{tvc}$ (degrees)")
        ax6.set_title("$\\dot{\\delta}_{tvc}$ vs Time")
        ax6.grid()

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
        ax4.scatter(t, np.rad2deg(np.array(x)[:, 7]))
        ax4.scatter(t_list, np.rad2deg(horizon[7, :]))
        ax4.legend(["$delta_{tvc_{ref}}$", "$delta_{tvc}$", "$delta_{tvc_{hor}}$"])
        ax4.grid()

        plt.show()

       
    def plot_horizon_online(self, t, x, last_u, u, horizon):
        # Clear the previous plot
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.ax5.clear()
        self.ax6.clear()

        N = self.controller_params["N"]
        t_list = np.linspace(t[-2], t[-2] + N * self.dt, N + 1, endpoint=True)
        t_full = np.concatenate((t, t_list))
        horizon = np.array(horizon)
        
        thrust_deriv_list = u[0]
        delta_tvc_ref_deriv_list = u[1]
        u_bounds = self.controller_params["u_bounds"]
        gamma_bounds = self.controller_params["gamma_bounds"]
        thrust_bounds = self.controller_params["thrust_bounds"]
        delta_tvc_bounds = self.controller_params["delta_tvc_bounds"]
        last_thrust_deriv_list = last_u[0]
        last_delta_tvc_ref_deriv_list = last_u[1]

        # Plot the new data
        self.ax1.scatter(np.array(x)[:, 0], np.array(x)[:, 2])
        self.ax1.scatter(horizon[0, :], horizon[2, :])
        self.ax1.legend(["$x$", "$x_{hor}$"])
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Position (m)")
        self.ax1.set_title("Position vs Time")
        self.ax1.axis('equal')
        self.ax1.grid()
        
        self.ax2.scatter(t, np.array(x)[:, 6])
        self.ax2.scatter(t_list, horizon[6, :])
        self.ax2.plot(t_full, [thrust_bounds[0]] * len(t_full), "--", color="black")
        self.ax2.plot(t_full, [thrust_bounds[1]] * len(t_full), "--", color="black")
        self.ax2.legend(["$f$", "$f_{ref}$", "$f_{min}$", "$f_{max}$"])
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("Thrust (N)")
        self.ax2.set_title("Thrust vs Time")
        self.ax2.grid()
        
        self.ax3.scatter(t[:len(t)-1], last_thrust_deriv_list)
        self.ax3.scatter(t_list[:len(t_list)-1], thrust_deriv_list)
        self.ax3.plot(t_full, [u_bounds[0][0]] * len(t_full), "--", color="black")
        self.ax3.plot(t_full, [u_bounds[0][1]] * len(t_full), "--", color="black")
        self.ax3.legend(["$\\dot{f}$", "$\\dot{f}_{ref}$", "$\\dot{f}_{min}$", "$\\dot{f}_{max}$"])
        self.ax3.set_xlabel("Time (s)")
        self.ax3.set_ylabel("Thrust (N)")
        self.ax3.set_title("Thrust vs Time")
        self.ax3.grid()
        
        self.ax4.scatter(t, np.rad2deg(np.array(x)[:, 4]))
        self.ax4.scatter(t_list, np.rad2deg(horizon[4, :]))
        self.ax4.plot(t_full, [np.rad2deg(gamma_bounds[0])] * len(t_full), "--", color="black")
        self.ax4.plot(t_full, [np.rad2deg(gamma_bounds[1])] * len(t_full), "--", color="black")
        self.ax4.legend(["$gamma$", "$gamma_{hor}$", "$\\gamma_{min}$", "$\\gamma_{max}$"])
        self.ax4.set_xlabel("Time (s)")
        self.ax4.set_ylabel("Angle (degrees)")
        self.ax4.set_title("Angle vs Time")
        self.ax4.grid()
        
        self.ax5.scatter(t, np.rad2deg(np.array(x)[:, 7]))
        self.ax5.scatter(t_list, np.rad2deg(horizon[7, :]))
        self.ax5.plot(t_full, [np.rad2deg(delta_tvc_bounds[0])] * len(t_full), "--", color="black")
        self.ax5.plot(t_full, [np.rad2deg(delta_tvc_bounds[1])] * len(t_full), "--", color="black")
        self.ax5.legend(["$\\delta_{tvc}$", "$\\delta_{tvc_{horizon}}$", "$\\delta_{tvc_{ref}}$", "$\\delta_{tvc_{min}}$", "$\\delta_{tvc_{max}}$"], loc='upper right')
        self.ax5.set_xlabel("Time (s)")
        self.ax5.set_ylabel("$\\delta_{tvc}$ (degrees)")
        self.ax5.set_title("$\\delta_{tvc}$ vs Time")
        self.ax5.grid()
        
        self.ax6.scatter(t[:len(t)-1], np.rad2deg(last_delta_tvc_ref_deriv_list))
        self.ax6.scatter(t_list[:len(t_list)-1], np.rad2deg(delta_tvc_ref_deriv_list))
        self.ax6.plot(t_full, [np.rad2deg(u_bounds[1][0])] * len(t_full), "--", color="black")
        self.ax6.plot(t_full, [np.rad2deg(u_bounds[1][1])] * len(t_full), "--", color="black")
        self.ax6.legend(["$\\dot{\\delta}_{tvc}$", "$\\dot{\\delta}_{tvc_{horizon}}$", "$\\dot{delta}_{tvc_{min}}$", "$\\delta_{tvc_{max}}$"], loc='upper right')
        self.ax6.set_xlabel("Time (s)")
        self.ax6.set_ylabel("$\\delta_{tvc}$ (degrees)")
        self.ax6.set_title("$\\delta_{tvc}$ vs Time")
        self.ax6.grid()


        # Redraw the plot and flush the events
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Display the plot
        plt.show()