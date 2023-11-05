import os
import sys
import numpy as np
import matplotlib.pyplot as plt

current_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.dirname(current_path)

sys.path.append(parent_path)
sys.path.append("../trajectory_planning")

from math_tools.RK4 import RK4

class DriftChecker:
    def __init__(self, env_params, rocket_params, trajectory_params):
        # Initialize class attributes for environment and rocket parameters
        self.env_params = env_params
        self.m = rocket_params["m"]
        self.l_tvc = rocket_params["l_tvc"]
        self.J_1 = rocket_params["J_1"]
        self.J_2 = rocket_params["J_2"]
        self.J_3 = rocket_params["J_3"]
        self.trajectory_params = trajectory_params
        
        self.drift_checker()

    def _plot_states(self):
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(15, 10))

        ax1.plot(self.time, self.sim_states[0, :])
        ax1.plot(self.trajectory_params["t"], self.trajectory_params["x"])
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("x")
        ax1.set_title("Trajectory")
        ax1.grid()
        ax1.legend(["RK4", "Analytical"])

        ax2.plot(self.time, self.sim_states[2, :])
        ax2.plot(self.trajectory_params["t"], self.trajectory_params["y"])
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("y")
        ax2.set_title("y position")
        ax2.grid()
        ax2.legend(["RK4", "Analytical"])

        ax3.plot(self.time, self.sim_states[4, :])
        ax3.plot(self.trajectory_params["t"], self.trajectory_params["z"])
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("z")
        ax3.set_title("z position")
        ax3.grid()
        ax3.legend(["RK4", "Analytical"])

        ax4.plot(self.time, self.sim_states[1, :])
        ax4.plot(self.trajectory_params["t"], self.trajectory_params["vx"])
        ax4.set_xlabel("t")
        ax4.set_ylabel("vx")
        ax4.set_title("Velocity in x")
        ax4.grid()
        ax4.legend(["RK4", "Analytical"])

        ax5.plot(self.time, self.sim_states[3, :])
        ax5.plot(self.trajectory_params["t"], self.trajectory_params["vy"])
        ax5.set_xlabel("t")
        ax5.set_ylabel("vy")
        ax5.set_title("Velocity in y")
        ax5.grid()
        ax5.legend(["RK4", "Analytical"])

        ax6.plot(self.time, self.sim_states[5, :])
        ax6.plot(self.trajectory_params["t"], self.trajectory_params["vz"])
        ax6.set_xlabel("t")
        ax6.set_ylabel("vz")
        ax6.set_title("Velocity in z")
        ax6.grid()
        ax6.legend(["RK4", "Analytical"])

        ax7.plot(self.time, self.sim_states[15, :])
        ax7.plot(self.trajectory_params["t"], self.trajectory_params["omega"][0, :])
        ax7.set_xlabel("t")
        ax7.set_ylabel("Angular velocities in x (rad/s)")
        ax7.set_title("Angular velocities in x (rad/s)")
        ax7.grid()
        ax7.legend(["wx RK4", "wx Analytical"])

        ax8.plot(self.time, self.sim_states[16, :])
        ax8.plot(self.trajectory_params["t"], self.trajectory_params["omega"][1, :])
        ax8.set_xlabel("t")
        ax8.set_ylabel("Angular velocities in y (rad/s)")
        ax8.set_title("Angular velocities in y (rad/s)")
        ax8.grid()
        ax8.legend(["wy RK4", "wy Analytical"])

        ax9.plot(self.time, self.sim_states[17, :])
        ax9.plot(self.trajectory_params["t"], self.trajectory_params["omega"][2, :])
        ax9.set_ylim([-1e-1, 1e-1])
        ax9.set_xlabel("t")
        ax9.set_ylabel("Angular velocities in z (rad/s)")
        ax9.set_title("Angular velocities in z (rad/s)")
        ax9.grid()
        ax9.legend(["wz RK4", "wz Analytical"])

        fig_2, (
            (ax1_2, ax2_2, ax3_2),
            (ax4_2, ax5_2, ax6_2),
            (ax7_2, ax8_2, ax9_2),
        ) = plt.subplots(3, 3, figsize=(15, 10))

        ax1_2.plot(self.time, self.sim_states[6, :])
        ax1_2.plot(self.trajectory_params["t"], self.trajectory_params["e1bx"])
        ax1_2.set_xlabel("t")
        ax1_2.set_ylabel("e1bx")
        ax1_2.set_title("Euler parameter e1bx")
        ax1_2.grid()
        ax1_2.legend(["e1bx RK4", "e1bx Analytical"])

        ax2_2.plot(self.time, self.sim_states[7, :])
        ax2_2.plot(self.trajectory_params["t"], self.trajectory_params["e1by"])
        ax2_2.set_xlabel("t")
        ax2_2.set_ylabel("e1by")
        ax2_2.set_title("Euler parameter e1by")
        ax2_2.grid()
        ax2_2.legend(["e1by RK4", "e1by Analytical"])

        ax3_2.plot(self.time, self.sim_states[8, :])
        ax3_2.plot(self.trajectory_params["t"], self.trajectory_params["e1bz"])
        ax3_2.set_xlabel("t")
        ax3_2.set_ylabel("e1bz")
        ax3_2.set_title("Euler parameter e1bz")
        ax3_2.grid()
        ax3_2.legend(["e1bz RK4", "e1bz Analytical"])

        ax4_2.plot(self.time, self.sim_states[9, :])
        ax4_2.plot(self.trajectory_params["t"], self.trajectory_params["e2bx"])
        ax4_2.set_xlabel("t")
        ax4_2.set_ylabel("e2bx")
        ax4_2.set_title("Euler parameter e2bx")
        ax4_2.grid()
        ax4_2.legend(["e2bx RK4", "e2bx Analytical"])

        ax5_2.plot(self.time, self.sim_states[10, :])
        ax5_2.plot(self.trajectory_params["t"], self.trajectory_params["e2by"])
        ax5_2.set_xlabel("t")
        ax5_2.set_ylabel("e2by")
        ax5_2.set_title("Euler parameter e2by")
        ax5_2.grid()
        ax5_2.legend(["e2by RK4", "e2by Analytical"])

        ax6_2.plot(self.time, self.sim_states[11, :])
        ax6_2.plot(self.trajectory_params["t"], self.trajectory_params["e2bz"])
        ax6_2.set_xlabel("t")
        ax6_2.set_ylabel("e2bz")
        ax6_2.set_title("Euler parameter e2bz")
        ax6_2.grid()
        ax6_2.legend(["e2bz RK4", "e2bz Analytical"])

        ax7_2.plot(self.time, self.sim_states[12, :])
        ax7_2.plot(self.trajectory_params["t"], self.trajectory_params["e3bx"])
        ax7_2.set_xlabel("t")
        ax7_2.set_ylabel("e3bx")
        ax7_2.set_title("Euler parameter e3bx")
        ax7_2.grid()
        ax7_2.legend(["e3bx RK4", "e3bx Analytical"])

        ax8_2.plot(self.time, self.sim_states[13, :])
        ax8_2.plot(self.trajectory_params["t"], self.trajectory_params["e3by"])
        ax8_2.set_xlabel("t")
        ax8_2.set_ylabel("e3by")
        ax8_2.set_title("Euler parameter e3by")
        ax8_2.grid()
        ax8_2.legend(["e3by RK4", "e3by Analytical"])

        ax9_2.plot(self.time, self.sim_states[14, :])
        ax9_2.plot(self.trajectory_params["t"], self.trajectory_params["e3bz"])
        ax9_2.set_xlabel("t")
        ax9_2.set_ylabel("e3bz")
        ax9_2.set_title("Euler parameter e3bz")
        ax9_2.grid()
        ax9_2.legend(["e3bz RK4", "e3bz Analytical"])

        yaw_ana = np.arctan2(self.trajectory_params["e1by"], self.trajectory_params["e1bx"])
        yaw_num = np.arctan2(self.sim_states[7, :], self.sim_states[6, :])

        plt.figure()
        plt.plot(self.time, yaw_num)
        plt.plot(self.trajectory_params["t"], yaw_ana)
        plt.xlabel("t")
        plt.ylabel("yaw")
        plt.title("Yaw")
        plt.grid()
        plt.legend(["yaw RK4", "yaw Analytical"])

        plt.show()


    ############################ Vee map operation #################################
    def vee(self, in_mat):
        """
        Calculates the vee operator for a 3x3 matrix.

        Parameters:
            in_mat (array-like): 3x3 matrix.

        Returns:
            out_vec (array-like): 3-dimensional vector.
        """
        out_vec = np.array([in_mat[2, 1], in_mat[0, 2], in_mat[1, 0]])
        return out_vec


    ######################### Creating the EOM ##############################
    def ode(self, last_sol, u):
        [
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
            omega_x,  # 24
            omega_y,  # 25
            omega_z,  # 26
        ] = last_sol

        sol = np.array(
            [
                x_dot,  # x
                (u[0] * e1bx + u[1] * e2bx + u[2] * e3bx) / self.m,  # v_x
                y_dot,  # y
                (u[0] * e1by + u[1] * e2by + u[2] * e3by) / self.m,  # v_y
                z_dot,  # z
                (u[0] * e1bz + u[1] * e2bz + u[2] * e3bz) / self.m - self.env_params["g"],  # v_z
                -omega_y * e3bx + omega_z * e2bx,  # e1bx
                -omega_y * e3by + omega_z * e2by,  # e1by
                -omega_y * e3bz + omega_z * e2bz,  # e1bz
                omega_x * e3bx - omega_z * e1bx,  # e2bx
                omega_x * e3by - omega_z * e1by,  # e2by
                omega_x * e3bz - omega_z * e1bz,  # e2bz
                -omega_x * e2bx + omega_y * e1bx,  # e3bx
                -omega_x * e2by + omega_y * e1by,  # e3by
                -omega_x * e2bz + omega_y * e1bz,  # e3bz
                (u[1] * self.l_tvc - (-self.J_2 + self.J_3) * omega_y * omega_z) / self.J_1,  # omega x
                (-u[0] * self.l_tvc - (self.J_1 - self.J_3) * omega_x * omega_z) / self.J_2,  # omega y
                (-(-self.J_1 + self.J_2) * omega_x * omega_y) / self.J_3,  # omega z
            ]
        )
        return sol


    ########################## Numerical integration ###############################
    def drift_checker(self, plot=False):
        """This function checks if the drift of the trajectory is within the limits.

        Args:
            env_params (dict): dictionary containing the environment parameters. Namely gravity (g), maximum drift (max_drift) and maximum angular drift (max_angular_drift).
            trajectory_params (dict): dictionary containing the trajectory parameters. Namely the control inputs (f1, f2, f3), the euler parameters (e1bx, e1by, e1bz, e2bx, e2by, e2bz, e3bx, e3by, e3bz), the time (t) and the initial state (initial_state).
            plot (bool, optional): Plots graphs showing the position drift if true. Defaults to False.

        Raises:
            ValueError: The drift was too high.
        """
        print("Checking drift...")
        # retrieve control inputs
        f1 = self.trajectory_params["f1"]
        f2 = self.trajectory_params["f2"]
        f3 = self.trajectory_params["f3"]

        x = self.trajectory_params["x"]
        y = self.trajectory_params["y"]
        z = self.trajectory_params["z"]
        vx = self.trajectory_params["vx"]
        vy = self.trajectory_params["vy"]
        vz = self.trajectory_params["vz"]
        e1bx = self.trajectory_params["e1bx"]
        e1by = self.trajectory_params["e1by"]
        e1bz = self.trajectory_params["e1bz"]
        e2bx = self.trajectory_params["e2bx"]
        e2by = self.trajectory_params["e2by"]
        e2bz = self.trajectory_params["e2bz"]
        e3bx = self.trajectory_params["e3bx"]
        e3by = self.trajectory_params["e3by"]
        e3bz = self.trajectory_params["e3bz"]
        omega = self.trajectory_params["omega"]

        # time = np.linspace(0, total_time, len(f_dot), endpoint=True)
        time = self.trajectory_params["t"]
        dt = time[1] - time[0]

        initial_state = np.array(
            [
                x[0],
                vx[0],
                y[0],
                vy[0],
                z[0],
                vz[0],
                e1bx[0],
                e1by[0],
                e1bz[0],
                e2bx[0],
                e2by[0],
                e2bz[0],
                e3bx[0],
                e3by[0],
                e3bz[0],
                omega[0, 0],
                omega[1, 0],
                omega[2, 0],
            ]
        )

        sim_states = np.zeros((len(initial_state), len(time)))  # states in columns
        sim_states[:, 0] = initial_state

        for i in range(len(time) - 1):
            sim_states[:, i + 1] = RK4(self.ode, sim_states[:, i], [f1[i], f2[i], f3[i]], dt)

            # # normalization
            sim_states[6:9, i + 1] /= np.linalg.norm(sim_states[6:9, i + 1])
            sim_states[9:12, i + 1] /= np.linalg.norm(sim_states[9:12, i + 1])
            sim_states[12:15, i + 1] /= np.linalg.norm(sim_states[12:15, i + 1])

        pos_drift = np.max(
            np.abs(
                np.array([sim_states[0], sim_states[2], sim_states[4]])
                - np.array([self.trajectory_params["x"], self.trajectory_params["y"], self.trajectory_params["z"]])
            )
        )
        max_pos = np.linalg.norm(
            [
                np.max(np.abs(self.trajectory_params["x"])),
                np.max(np.abs(self.trajectory_params["y"])),
                np.max(np.abs(self.trajectory_params["z"])),
            ]
        )
        rel_pos_drift = np.linalg.norm(pos_drift) / max_pos

        angular_drift_x = np.zeros_like(e1bx)
        angular_drift_y = np.zeros_like(e1by)
        angular_drift_z = np.zeros_like(e1bz)

        # compute the abs angular drift
        for i in range(len(e1bx)):
            R = np.array([sim_states[6:9, i], sim_states[9:12, i], sim_states[12:15, i]]).T
            R_des = np.array(
                [
                    [e1bx[i], e2bx[i], e3bx[i]],
                    [e1by[i], e2by[i], e3by[i]],
                    [e1bz[i], e2bz[i], e3bz[i]],
                ]
            )

            er = 1 / 2 * self.vee(R_des.T @ R - R.T @ R_des)

            angular_drift_x[i] = er[0]
            angular_drift_y[i] = er[1]
            angular_drift_z[i] = er[2]

        max_angular_drift = np.linalg.norm(
            [
                np.max(np.abs(angular_drift_x)),
                np.max(np.abs(angular_drift_y)),
                np.max(np.abs(angular_drift_z)),
            ]
        )  # norm of the maximum values (very conservative)

        rel_angular_drift = max_angular_drift / (np.pi)  # assuming it goes from -pi to pi
        self.time = time
        self.sim_states = sim_states

        if plot:
            self._plot_states()

        if rel_pos_drift <= self.env_params["max_drift"] and rel_angular_drift <= self.env_params["max_angular_drift"]:
            print("Drift check passed!")

        elif rel_pos_drift > self.env_params["max_drift"]:
            raise ValueError(
                "Position drift too high!\nThe trajectory is not feasible from the numerical integration point of view. Please, check the trajectory and controller parameters."
            )

        else:
            raise ValueError(
                "Angular drift too high!\nThe trajectory is not feasible from the numerical integration point of view. Please, check the trajectory and controller parameters."
            )
