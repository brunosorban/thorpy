import numpy as np
import casadi as ca
from math_tools.RK4 import RK4
from controller.plot_simulation import plot_simulation
from math_tools.data_handler_class import DataHandler

class MPC_controller:
    """
    Class to implement the MPC controller.
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

        self.epos_list = [np.array([0, 0, 0])]
        self.er_list = [np.array([0, 0, 0])]

        if trajectory_params is not None:
            self.follow_trajectory = True

        # rocket and environment parameters
        g = env_params["g"]
        m = rocket_params["m"]
        J_1 = rocket_params["J_1"]
        J_2 = rocket_params["J_2"]
        J_3 = rocket_params["J_3"]
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
        R = controller_params["R"]

        # create variables used during operation (update function)
        self.dt = T / N
        self.x0_val = x0_val

        # initializing the state varibles
        x = ca.SX.sym("x")  # states
        x_dot = ca.SX.sym("x_dot")  # states
        y = ca.SX.sym("y")  # states
        y_dot = ca.SX.sym("y_dot")  # states
        z = ca.SX.sym("z")  # states
        z_dot = ca.SX.sym("z_dot")  # states
        e1bx = ca.SX.sym("e1bx")  # states
        e1by = ca.SX.sym("e1by")  # states
        e1bz = ca.SX.sym("e1bz")  # states
        e2bx = ca.SX.sym("e2bx")  # states
        e2by = ca.SX.sym("e2by")  # states
        e2bz = ca.SX.sym("e2bz")  # states
        e3bx = ca.SX.sym("e3bx")  # states
        e3by = ca.SX.sym("e3by")  # states
        e3bz = ca.SX.sym("e3bz")  # states
        e1tx = ca.SX.sym("e1tx")  # states
        e1ty = ca.SX.sym("e1ty")  # states
        e1tz = ca.SX.sym("e1tz")  # states
        e2tx = ca.SX.sym("e2tx")  # states
        e2ty = ca.SX.sym("e2ty")  # states
        e2tz = ca.SX.sym("e2tz")  # states
        e3tx = ca.SX.sym("e3tx")  # states
        e3ty = ca.SX.sym("e3ty")  # states
        e3tz = ca.SX.sym("e3tz")  # states
        omega_x = ca.SX.sym("omega_x")  # states
        omega_y = ca.SX.sym("omega_y")  # states
        omega_z = ca.SX.sym("omega_z")  # states
        thrust = ca.SX.sym("thrust")  # states

        # create a vector with variable names
        variables = ca.vertcat(
            x,  # 0
            x_dot,  # 1
            y,  # 2
            y_dot,  # 3
            z,  # 4
            z_dot,  # 5
            e1bx,  # 6
            e1by,  # 7
            e1bz,  # 8
            e2bx,  # 9
            e2by,  # 10
            e2bz,  # 11
            e3bx,  # 12
            e3by,  # 13
            e3bz,  # 14
            e1tx,  # 15
            e1ty,  # 16
            e1tz,  # 17
            e2tx,  # 18
            e2ty,  # 19
            e2tz,  # 20
            e3tx,  # 21
            e3ty,  # 22
            e3tz,  # 23
            omega_x,  # 24
            omega_y,  # 25
            omega_z,  # 26
            thrust,  # 27
        )

        # initializing the control varibles
        thrust_dot = ca.SX.sym("thrust_dot")  # controls
        delta_tvc_dot_x = ca.SX.sym("delta_tvc_dot_x")  # controls
        delta_tvc_dot_y = ca.SX.sym("delta_tvc_dot_y")  # controls

        # create a vector with variable names
        u = ca.vertcat(thrust_dot, delta_tvc_dot_x, delta_tvc_dot_y)

        # define the nonlinear ode
        ode = ca.vertcat(
            x_dot,  # x
            thrust / m * e3tx,  # v_x
            y_dot,  # y
            thrust / m * e3ty,  # v_y
            z_dot,  # z
            thrust / m * e3tz - g,  # v_z
            -omega_y * e3bx + omega_z * e2bx,  # e1bx
            -omega_y * e3by + omega_z * e2by,  # e1by
            -omega_y * e3bz + omega_z * e2bz,  # e1bz
            omega_x * e3bx - omega_z * e1bx,  # e2bx
            omega_x * e3by - omega_z * e1by,  # e2by
            omega_x * e3bz - omega_z * e1bz,  # e2bz
            -omega_x * e2bx + omega_y * e1bx,  # e3bx
            -omega_x * e2by + omega_y * e1by,  # e3by
            -omega_x * e2bz + omega_y * e1bz,  # e3bz
            -(omega_y + delta_tvc_dot_y) * e3tx + omega_z * e2tx,  # e1tx
            -(omega_y + delta_tvc_dot_y) * e3ty + omega_z * e2ty,  # e1ty
            -(omega_y + delta_tvc_dot_y) * e3tz + omega_z * e2tz,  # e1tz
            (omega_x + delta_tvc_dot_x) * e3tx - omega_z * e1tx,  # e2tx
            (omega_x + delta_tvc_dot_x) * e3ty - omega_z * e1ty,  # e2ty
            (omega_x + delta_tvc_dot_x) * e3tz - omega_z * e1tz,  # e2tz
            -(omega_x + delta_tvc_dot_x) * e2tx + (omega_y + delta_tvc_dot_y) * e1tx,  # e3tx
            -(omega_x + delta_tvc_dot_x) * e2ty + (omega_y + delta_tvc_dot_y) * e1ty,  # e3ty
            -(omega_x + delta_tvc_dot_x) * e2tz + (omega_y + delta_tvc_dot_y) * e1tz,  # e3tz
            (thrust * l_tvc * (e3tx * e2bx + e3ty * e2by + e3tz * e2bz) - (-J_2 + J_3) * omega_y * omega_z)
            / J_1,  # omega x
            (-thrust * l_tvc * (e3tx * e1bx + e3ty * e1by + e3tz * e1bz) - (J_1 - J_3) * omega_x * omega_z)
            / J_2,  # omega y
            (-(-J_1 + J_2) * omega_x * omega_y) / J_3,  # omega z
            thrust_dot,  # thrust
        )

        # creating the ploblem statement function
        f = ca.Function(
            "f",
            [variables, u],
            [ode],
            [
                "[x, x_dot, y, y_dot, z, z_dot, e1bx, e1by, e1bz, e2bx, e2by, e2bz, e3bx, e3by, e3bz, e1tx, e1ty, e1tz, e2tx, e2ty, e2tz, e3tx, e3ty, e3tz, omega_x, omega_y, omega_z, thrust]",
                "[thrust, delta_tvc_dot_y, delta_tvc_dot_z]",
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
                "[x, x_dot, y, y_dot, z, z_dot, e1bx, e1by, e1bz, e2bx, e2by, e2bz, e3bx, e3by, e3bz, e1tx, e1ty, e1tz, e2tx, e2ty, e2tz, e3tx, e3ty, e3tz, omega_x, omega_y, omega_z, thrust]",
                "[thrust, delta_tvc_dot_y, delta_tvc_dot_z]",
            ],
            ["x_next"],
        )

        # building the optimal control problem
        self.opti = ca.Opti()

        # create the state, control and initial state varibles
        self.x = self.opti.variable(28, N + 1)
        self.u = self.opti.variable(3, N)
        self.x_initial = self.opti.parameter(28, 1)
        self.x_target = self.opti.parameter(6, N + 1)

        # define the cost function
        self.obj = 0
        for k in range(N):
            # position and velocity tracking error
            tracking_error = self.x[0:6, k] - self.x_target[0:6, k]
            self.obj += ca.mtimes([tracking_error.T, Q[0:6, 0:6], tracking_error])

            # control effort
            self.obj += ca.mtimes([self.u[:, k].T, R, self.u[:, k]])

        # position and velocity tracking error
        tracking_error = self.x[0:6, N] - self.x_target[0:6, N]
        self.obj += ca.mtimes([tracking_error.T, Q[0:6, 0:6], tracking_error])

        self.opti.minimize(self.obj)

        ##### apply the constraint that the state is defined by my system ######
        # set the initial position
        self.opti.subject_to(self.x[:, 0] == self.x_initial)

        # set the dynamics
        for k in range(0, N + 1):
            if k < N:
                # apply the dynamics as constraints
                self.opti.subject_to(self.x[:, k + 1] == F(self.x[:, k], self.u[:, k]))

                # apply bounds to the inputs
                self.opti.subject_to(self.u[0, k] >= thrust_dot_bounds[0])
                self.opti.subject_to(self.u[0, k] <= thrust_dot_bounds[1])
                self.opti.subject_to(self.u[1, k] >= delta_tvc_dot_bounds[0])
                self.opti.subject_to(self.u[1, k] <= delta_tvc_dot_bounds[1])
                self.opti.subject_to(self.u[2, k] >= delta_tvc_dot_bounds[0])
                self.opti.subject_to(self.u[2, k] <= delta_tvc_dot_bounds[1])

            # add the thrust constraint
            self.opti.subject_to(self.x[27, k] >= thrust_bounds[0])
            self.opti.subject_to(self.x[27, k] <= thrust_bounds[1])

            # TODO: Update to 3D with rotation matrix
            e1b = [self.x[6, k], self.x[7, k], self.x[8, k]]
            e2b = [self.x[9, k], self.x[10, k], self.x[11, k]]
            e3t = [self.x[21, k], self.x[22, k], self.x[23, k]]

            delta_tvc_x, delta_tvc_y = self.compute_rotation_angles(e1b, e2b, e3t)
            self.opti.subject_to(delta_tvc_x >= delta_tvc_bounds[0])
            self.opti.subject_to(delta_tvc_x <= delta_tvc_bounds[1])
            self.opti.subject_to(delta_tvc_y >= delta_tvc_bounds[0])
            self.opti.subject_to(delta_tvc_y <= delta_tvc_bounds[1])

        # set target
        if self.follow_trajectory:
            self.update_traj(t0_val)
        else:
            self.opti.set_value(self.x_target, x_target)

        # set initial state
        self.opti.set_value(self.x_initial, x0_val)

        # configure the solver
        opts = {"ipopt.print_level": 0, "print_time": 0}
        self.opti.solver("ipopt", opts)

    @staticmethod
    def compute_rotation_angles(e1b, e2b, e3t):
        # Inline function for dot product
        def dot_product(v1, v2):
            return sum(x * y for x, y in zip(v1, v2))

        # Calculate dot products
        p_e1b = dot_product(e3t, e1b)
        p_e2b = dot_product(e3t, e2b)

        # Calculate angles in radians
        theta_e1b = p_e2b  # Rotation around e1b
        theta_e2b = -p_e1b  # Rotation around e2b

        return theta_e1b, theta_e2b

    def update_target(self, new_target):
        # TODO: optimize this function
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
        px = self.trajectory_params["x"]
        py = self.trajectory_params["y"]
        pz = self.trajectory_params["z"]
        vx = self.trajectory_params["vx"]
        vy = self.trajectory_params["vy"]
        vz = self.trajectory_params["vz"]

        N = self.controller_params["N"]
        t_list = np.linspace(time, time + N * self.dt, N + 1, endpoint=True)

        px_list = np.zeros(len(t_list))
        py_list = np.zeros(len(t_list))
        pz_list = np.zeros(len(t_list))
        vx_list = np.zeros(len(t_list))
        vy_list = np.zeros(len(t_list))
        vz_list = np.zeros(len(t_list))

        for i in range(len(t_list)):
            px_list[i] = self.linear_spline(t_list[i], self.trajectory_params["t"], px)
            py_list[i] = self.linear_spline(t_list[i], self.trajectory_params["t"], py)
            pz_list[i] = self.linear_spline(t_list[i], self.trajectory_params["t"], pz)
            vx_list[i] = self.linear_spline(t_list[i], self.trajectory_params["t"], vx)
            vy_list[i] = self.linear_spline(t_list[i], self.trajectory_params["t"], vy)
            vz_list[i] = self.linear_spline(t_list[i], self.trajectory_params["t"], vz)

        target = np.array(
            [
                px_list,
                vx_list,
                py_list,
                vy_list,
                pz_list,
                vz_list,
            ]
        )
        self.update_target(target)

    def simulate_inside(self, sim_time, plot_online=False):
        """This method simulates the system using the MPC controller. The simulations is performed
        using the Runge-Kutta 4th order method and the time step is defined to the same as the
        controller.

        Args:
            sim_time (float): Total simulation time.
            plot_online (bool, optional): If True, a monitor to the solution status will be plotted. Defaults to False.

        Returns:
            time: array with the time points
            x: array with the states
            u: array with the control inputs
            state_horizon_list: list with the state horizon
            control_horizon_list: list with the control horizon
        """
        print("Starting simulation")
        t = [0]
        x_list = [np.squeeze(self.x0_val)]  # each line contains a state
        thrust = []  # each line contains a control input
        delta_tvc_y = []  # each line contains a control input
        delta_tvc_z = []  # each line contains a control input
        state_horizon_list = []
        control_horizon_list = []

        # if plot_online:
        #     # Turn on interactive mode
        #     # Initialize the figure and axes for the plot
        #     self.fig, (
        #         (self.ax1, self.ax2, self.ax3),
        #         (self.ax4, self.ax5, self.ax6),
        #         (self.ax7, self.ax8, self.ax9),
        #     ) = plt.subplots(3, 3, figsize=(15, 10))
        #     plt.ion()

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
                self.opti.set_initial(self.x, state_horizon_list[-1])
                self.opti.set_initial(self.u, control_horizon_list[-1])
                sol = self.opti.solve()

            # retrieve the results
            u = sol.value(self.u)
            horizon = sol.value(self.x)

            # normalization of the angular parameters
            for i in range(6, 24, 3):
                for j in range(len(horizon[0])):
                    norm = np.linalg.norm([horizon[i][j], horizon[i + 1][j], horizon[i + 2][j]])

                    horizon[i][j] = horizon[i][j] / norm
                    horizon[i + 1][j] = horizon[i + 1][j] / norm
                    horizon[i + 2][j] = horizon[i + 2][j] / norm

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
                    horizon[14][1],
                    horizon[15][1],
                    horizon[16][1],
                    horizon[17][1],
                    horizon[18][1],
                    horizon[19][1],
                    horizon[20][1],
                    horizon[21][1],
                    horizon[22][1],
                    horizon[23][1],
                    horizon[24][1],
                    horizon[25][1],
                    horizon[26][1],
                    horizon[27][1],
                ]
            )

            state_horizon_list.append(horizon)
            control_horizon_list.append(u)
            thrust.append(u[0, 0])
            delta_tvc_y.append(u[1, 0])
            delta_tvc_z.append(u[2, 0])
            t.append(t[-1] + self.dt)

            # compute actual error
            # TODO: create get target function
            px_target = self.linear_spline(t[-1], self.trajectory_params["t"], self.trajectory_params["x"])
            py_target = self.linear_spline(t[-1], self.trajectory_params["t"], self.trajectory_params["y"])
            pz_target = self.linear_spline(t[-1], self.trajectory_params["t"], self.trajectory_params["z"])

            pos_error = np.array([x_list[-1][0], x_list[-1][2], x_list[-1][4]]) - np.array(
                [px_target, py_target, pz_target]
            )
            self.epos_list.append(pos_error)

            if plot_online and t[-1] > self.dt:
                self.plot_horizon_online(t, np.array(x_list), [thrust, delta_tvc_y, delta_tvc_z], u, horizon)

        print("Simulation finished")
        
        self.t = t
        self.x_list = x_list
        self.thrust = thrust
        self.delta_tvc_y = delta_tvc_y
        self.delta_tvc_z = delta_tvc_z
        self.state_horizon_list = state_horizon_list
        self.control_horizon_list = control_horizon_list

        return (
            np.array(t),
            np.array(x_list),
            np.array([thrust, delta_tvc_y, delta_tvc_z]),
            np.array(state_horizon_list),
            np.array(control_horizon_list),
        )
    
    def plot_simulation(self):
        plot_simulation(self.t, self.x_list, [self.thrust, self.delta_tvc_y, self.delta_tvc_z], self.trajectory_params, self.controller_params, self.epos_list)
        
    def export_simulation(self, folder_path: str, file_name: str):
        sol_dict = {
            "t": self.t,
            "x": self.x_list,
            "u": [self.thrust, self.delta_tvc_y, self.delta_tvc_z],
            "state_horizon_list": self.state_horizon_list,
            "control_horizon_list": self.control_horizon_list,
        }

        DataHandler.save(folder_path, file_name, sol_dict)