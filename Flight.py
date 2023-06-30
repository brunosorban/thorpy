# -*- coding: utf-8 -*-

__author__ = "Bruno Abdulklech Sorban"
__copyright__ = "Copyright 2023, Bruno Sorban"
__license__ = "MIT"

import numpy as np
from Function import Function
from scipy import integrate


class Flight:
    def __init__(
        self,
        rocket_params,
        environment,
        controller,
        initial_solution=None,
        max_time=60,
        rtol=1e-6,
        atol=1e-6,
        max_step=1e-3,
    ):
        self.rocket_params = rocket_params
        self.controller = controller
        self.environment = environment

        self.max_time = max_time
        self.initial_solution = (
            [0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, rocket_params["m"] * environment.g] if initial_solution is None else initial_solution
        )
        self.t0 = 0
        self.atol = atol
        self.rtol = rtol
        self.max_step = max_step
        self.time_count = 0

        # initilize variables
        self.x = [self.initial_solution[0]]
        self.vx = [self.initial_solution[1]]
        self.ax = [0]
        self.y = [self.initial_solution[2]]
        self.vy = [self.initial_solution[3]]
        self.ay = [0]
        self.e1bx = [np.cos(self.initial_solution[4])]
        self.e1by = [np.sin(self.initial_solution[5])]
        self.e2bx = [np.cos(self.initial_solution[6])]
        self.e2by = [np.sin(self.initial_solution[7])]
        self.e1tx = [np.cos(self.initial_solution[8])]
        self.e1ty = [np.sin(self.initial_solution[9])]
        self.e2tx = [np.cos(self.initial_solution[10])]
        self.e2ty = [np.sin(self.initial_solution[11])]
        self.omega_z = [self.initial_solution[12]]
        self.omega_z_dot = [0]
        self.thrust = [self.initial_solution[13]]
        self.thrust_dot = [0]
        self.delta_tvc_dot = [0]

    def uDot(self, t, sol, post_process=False):
        """Calculate the state space derivatives"""

        # if t - self.time_count > 0.1:
        #     # Clear the previous output
        #     clear_output(wait=True)

        #     # Display the current simulation time
        #     display("Simulation time: {:.1f} seconds".format(t))
        #     self.time_count += 0.1

        [x, vx, y, vy, e1bx, e1by, e2bx, e2by, e1tx, e1ty, e2tx, e2ty, omega_z, thrust] = sol
        f_dot, delta_tvc_dot = self.controller.update(t, sol)
        
        f1 = vx,  # x
        f2 = thrust / self.rocket_params["m"] * e1tx,  # v_x
        f3 = vy,  # y
        f4 = thrust / self.rocket_params["m"] * e1ty - self.environment.g,  # v_y
        f5 = omega_z * e2bx,  # e1bx
        f6 = omega_z * e2by,  # e1by
        f7 = -omega_z * e1bx,  # e2bx
        f8 = -omega_z * e1by,  # e2by
        f9 = (delta_tvc_dot + omega_z) * e2tx,  # e1tx
        f10 = (delta_tvc_dot + omega_z) * e2ty,  # e1ty
        f11 = -(delta_tvc_dot + omega_z) * e1tx,  # e2tx
        f12 = -(delta_tvc_dot + omega_z) * e1ty,  # e2ty
        f13 = -thrust * self.rocket_params["l_tvc"] * (e1tx * e2bx + e1ty * e2by) / self.rocket_params["J_z"],  # omega
        f14 = f_dot,  # f

        uDot = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14]

        # Get control variables
        self.x.append(x)
        self.y.append(y)
        self.vx.append(vx)
        self.vy.append(vy)
        self.ax.append(f2)
        self.ay.append(f4)
        self.e1bx.append(e1bx)
        self.e1by.append(e1by)
        self.e2bx.append(e2bx)
        self.e2by.append(e2by)
        self.e1tx.append(e1tx)
        self.e1ty.append(e1ty)
        self.e2tx.append(e2tx)
        self.e2ty.append(e2ty)
        self.omega_z.append(omega_z)
        self.omega_z_dot.append(f13)
        self.thrust.append(thrust)
        self.thrust_dot.append(f_dot)
        self.delta_tvc_dot.append(delta_tvc_dot)
        
        return uDot

    def solve_system(self):
        # Initialize the data
        self.time = [self.t0]
        self.solution = [self.initial_solution]

        # Create the solver
        self.solver = integrate.LSODA(
            self.uDot,
            t0=0,
            y0=self.initial_solution,
            t_bound=self.max_time,
            max_step=self.max_step,
            rtol=self.rtol,
            atol=self.atol,
        )

        # Iterate until max_time is reached
        while self.solver.status != "finished":
            self.solver.step()
            self.time.append(self.solver.t)
            self.solution.append(self.solver.y)
        print("Solution Finished")

    def post_process(self):
        # Create function objects from the retrieved data
        gamma = np.arctan2(self.e1by, self.e1bx)
        delta_tvc = np.arctan2(self.e2ty, self.e2tx) - gamma
        
        self.r = Function(
            self.time,
            np.sqrt(np.array(self.x) ** 2 + np.array(self.y) ** 2),
            xlabel="Time (s)",
            ylabel="Distance from the origin (m)",
            name="r",
        )
        self.v = Function(
            self.time,
            np.sqrt(np.array(self.vx) ** 2 + np.array(self.vy) ** 2),
            xlabel="Time (s)",
            ylabel="Velocity (m/s)",
            name="v",
        )
        self.xy = Function(
            self.x,
            self.y,
            xlabel="Position in inertial frame x axis (m)",
            ylabel="Position in inertial frame y axis (m)",
            name="xy",
        )
        self.x = Function(
            self.time,
            self.x,
            xlabel="Time (s)",
            ylabel="Position in inertial frame x axis (m)",
            name="x",
        )
        self.y = Function(
            self.time,
            self.y,
            xlabel="Time (s)",
            ylabel="Position in inertial frame y axis (m)",
            name="y",
        )
        self.vx = Function(
            self.time,
            self.vx,
            xlabel="Time (s)",
            ylabel="Velocity in inertial frame x axis (m/s)",
            name="vx",
        )
        self.vy = Function(
            self.time,
            self.vy,
            xlabel="Time (s)",
            ylabel="Velocity in inertial frame y axis (m/s)",
            name="vy",
        )
        self.ax = Function(
            self.time,
            self.ax,
            xlabel="Time (s)",
            ylabel="Acceleration in inertial frame x axis (m/s²)",
            name="ax",
        )
        self.ay = Function(
            self.time,
            self.ay,
            xlabel="Time (s)",
            ylabel="Acceleration in inertial frame y axis (m/s²)",
            name="ay",
        )
        self.gamma = Function(
            self.time,
            np.rad2deg(gamma),
            xlabel="Time (s)",
            ylabel="Yaw angle (degrees)",
            name="gamma",
        )
        self.omega_z = Function(
            self.time,
            self.omega_z,
            xlabel="Time (s)",
            ylabel="Yaw angular velocity (rad/s)",
            name="omega_z",
        )
        self.omega_z_dot = Function(
            self.time,
            self.omega_z_dot,
            xlabel="Time (s)",
            ylabel="Yaw angular acceleration (rad/s²)",
            name="omega_z_dot",
        )
        self.thrust = Function(
            self.time,
            self.thrust,
            xlabel="Time (s)",
            ylabel="Thrust force (N)",
            name="f",
        )
        self.thrust_dot = Function(
            self.time,
            self.thrust_dot,
            xlabel="Time (s)",
            ylabel="Thrust force derivative (N/s)",
            name="thrust_dot",
        )
        self.e1bx = Function(
            self.time,
            self.e1bx,
            xlabel="Time (s)",
            ylabel="e1bx",
            name="e1bx",
        )
        self.e1by = Function(
            self.time,
            self.e1by,
            xlabel="Time (s)",
            ylabel="e1by",
            name="e1by",
        )
        self.e2bx = Function(
            self.time,
            self.e2bx,
            xlabel="Time (s)",
            ylabel="e2bx",
            name="e2bx",
        )
        self.e2by = Function(
            self.time,
            self.e2by,
            xlabel="Time (s)",
            ylabel="e2by",
            name="e2by",
        )

        self.e1tx = Function(
            self.time,
            self.e1tx,
            xlabel="Time (s)",
            ylabel="e1tx",
            name="e1tx",
        )
        self.e1ty = Function(
            self.time,
            self.e1ty,
            xlabel="Time (s)",
            ylabel="e1ty",
            name="e1ty",
        )
        self.e2tx = Function(
            self.time,
            self.e2tx,
            xlabel="Time (s)",
            ylabel="e2tx",
            name="e2tx",
        )
        self.e2ty = Function(
            self.time,
            self.e2ty,
            xlabel="Time (s)",
            ylabel="e2ty",
            name="e2ty",
        )
        self.delta_tvc = Function(
            self.time,
            np.rad2deg(delta_tvc),
            xlabel="Time (s)",
            ylabel="TVC angle (degrees)",
            name="delta_tvc",
        )
        self.delta_tvc_dot = Function(
            self.time,
            self.delta_tvc_dot,
            xlabel="Time (s)",
            ylabel="TVC angle derivative (degrees/s)",
            name="delta_tvc_dot",
        )
        
        