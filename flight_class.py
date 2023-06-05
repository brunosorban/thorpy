# -*- coding: utf-8 -*-

__author__ = "Bruno Abdulklech Sorban"
__copyright__ = "Copyright 2022, Bruno Sorban"
__license__ = "MIT"

import numpy as np
from Function import Function
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import sys, pygame  # For the animation function
import imageio  # For the animation function
from IPython.display import display, clear_output


class Flight:
    def __init__(
        self,
        m,
        C,
        g,
        controller,
        initial_solution=None,
        max_time=60,
        rtol=1e-6,
        atol=1e-6,
        max_step=1e-3,
    ):
        self.m = m
        self.C = C
        self.g = g
        self.controller = controller

        self.max_time = max_time
        self.initial_solution = (
            [0, 0, 0, 0, np.pi, 0] if initial_solution is None else initial_solution
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
        self.gamma = [self.initial_solution[4]]
        self.gamma_dot = [self.initial_solution[5]]
        self.gamma_dot_dot = [0]
        self.f = [m * g]
        self.tau = [0]
        self.J = [0]

    def uDot(self, t, sol, post_process=False):
        """Calculate the state space derivatives"""

        # if t - self.time_count > 0.1:
        #     # Clear the previous output
        #     clear_output(wait=True)

        #     # Display the current simulation time
        #     display("Simulation time: {:.1f} seconds".format(t))
        #     self.time_count += 0.1

        [x, vx, y, vy, gamma, omega] = sol
        [f, tau] = self.controller.update(t, np.array([x, vx, y, vy, gamma, omega]))
        # [f, tau] = [0, 0]
        J = self.controller.J_total

        f1 = vx
        f2 = f * np.cos(gamma) / self.m
        f3 = vy
        f4 = f * np.sin(gamma) / self.m - self.g
        f5 = omega
        f6 = tau / self.C - omega**2

        uDot = [f1, f2, f3, f4, f5, f6]

        # Get control variables
        self.x.append(x)
        self.y.append(y)
        self.vx.append(vx)
        self.vy.append(vy)
        self.ax.append(f2)
        self.ay.append(f4)
        self.gamma.append(gamma)
        self.gamma_dot.append(omega)
        self.gamma_dot_dot.append(f6)
        self.f.append(f)
        self.tau.append(tau)
        self.J.append(J)
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
            180 / np.pi * np.array(self.gamma),
            xlabel="Time (s)",
            ylabel="Yaw angle (degrees)",
            name="gamma",
        )
        self.gamma_dot = Function(
            self.time,
            self.gamma_dot,
            xlabel="Time (s)",
            ylabel="Yaw angular velocity (rad/s)",
            name="gamma_dot",
        )
        self.gamma_dot_dot = Function(
            self.time,
            self.gamma_dot_dot,
            xlabel="Time (s)",
            ylabel="Yaw angular acceleration (rad/s²)",
            name="gamma_dot_dot",
        )

        self.f = Function(
            self.time,
            self.f,
            xlabel="Time (s)",
            ylabel="Thrust force (N)",
            name="f",
        )
        self.tau = Function(
            self.time,
            self.tau,
            xlabel="Time (s)",
            ylabel="Steering torque (Nm)",
            name="tau",
        )

        # self.controller.J_total_list = np.array(self.controller.J_total_list)
        # self.J = Function(
        #     self.time,
        #     self.J,
        #     xlabel="Time (s)",
        #     ylabel="Thrust controller cost",
        #     name="J",
        # )

        # self.controller.J_total_list = np.array(self.controller.J_total_list)
        # self.J_clean = Function(
        #     self.controller.J_total_list[:, 0],
        #     self.controller.J_total_list[:, 1],
        #     xlabel="Time (s)",
        #     ylabel="Thrust controller cost (cleaned)",
        #     name="J_clean",
        # )

    # Animation
    def blitRotateCenter(self, surf, image, topleft, angle):
        # Auxiliar function to blit the animation image rotated
        rotated_image = pygame.transform.rotate(image, angle)
        new_rect = rotated_image.get_rect(center=image.get_rect(topleft=topleft).center)
        surf.blit(rotated_image, new_rect.topleft)

    def mapi(self, x, y, xlim_sup, ylim_sup, xlim_inf, ylim_inf, size, realScale=True):
        # Auxiliar function to map the trajectory to fit in view
        x_factor = size[0] / abs(xlim_sup - xlim_inf)
        y_factor = size[1] / abs(ylim_sup - ylim_inf)

        if realScale:  # select the lowest factor
            if x_factor > y_factor:
                x_factor = y_factor
            else:
                y_factor = x_factor

        # print("scale =", x_factor)
        return np.array([(x - xlim_inf) * x_factor, (y - ylim_inf) * y_factor])

    def animate(self, timeMax=False, save=False, scale=1, matplotlib=False):
        if not timeMax:
            timeMax = self.max_time

        # animation constants
        offset = 10
        dt = 1 / 60 / 2
        target_goal = (30, 30)

        # Initialization
        xlim_inf = abs(min(self.x.__Y_source__))
        xlim_sup = max(self.x.__Y_source__) + offset
        ylim_inf = abs(min(self.y.__Y_source__))
        ylim_sup = max(self.y.__Y_source__) + offset

        pygame.init()
        pygame.display.init()
        font = pygame.font.SysFont("Helvetica", 32)

        if save:
            w = imageio.get_writer("../Videos/rocket.mp4", format="FFMPEG", fps=60)

        # Creating animation screen
        # size = np.array([int(1920 / 1.25), int(1080 / 1.25)])
        size = np.array([int(3840 / 1.5), int(2160 / 1.5)])
        screen = pygame.display.set_mode(size)

        # Preparing images
        background = pygame.image.load("../Animation/sky.jpg")
        background = pygame.transform.scale(background, (size))

        rocket = pygame.image.load("../Animation/rocket.png")
        # rocket = pygame.transform.scale(rocket, (rocket.get_height() / scale, rocket.get_width() / scale))
        # rocket = pygame.transform.scale(rocket, (int(size[0] / 10), int((151 / 10))))
        rocket = pygame.transform.scale(
            rocket, (int(size[0] / 10), int((3840 / 1920) * 151 / 10))
        )

        target = pygame.image.load("../Animation/target.png")
        target = pygame.transform.scale(target, (40, 40))

        # Initialize position vectors
        initial_position = np.array([0, size[1] - 128])
        position = initial_position

        timeCount = 0
        time_list = [0]
        x_list = [self.x(0)]
        y_list = [self.y(0)]
        gamma_list = [self.gamma(0)]

        # retrive control horizons
        self.controller.horizon_list = np.array(self.controller.horizon_list)
        control_time = self.controller.horizon_list[:, 0]
        horizon_list_f = self.controller.horizon_list[:, 1]

        # Iteration in time
        while timeCount <= timeMax:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

            # Get position, current time and vehicle velocity
            position = initial_position + self.mapi(
                self.x(timeCount),
                -self.y(timeCount),
                xlim_sup,
                ylim_sup,
                xlim_inf,
                ylim_inf,
                size,
                realScale=True,
            )

            # Get position, current time and vehicle velocity
            target_pos = initial_position + self.mapi(
                target_goal[0],
                -target_goal[1],
                xlim_sup,
                ylim_sup,
                xlim_inf,
                ylim_inf,
                size,
                realScale=True,
            )

            # Compensate to be on rocket center
            target_pos += np.array(
                [rocket.get_width() / 2, rocket.get_height() / 2]
            ) - np.array([target.get_width() / 2, target.get_height() / 2])

            tempo = font.render(
                "Time : {:.1f}s".format(timeCount), True, (255, 255, 255)
            )

            velocidade = font.render(
                "Speed : {:.1f}m/s".format(self.v(timeCount)),
                True,
                (255, 255, 255),
            )

            escala = font.render("Scale : {:.1f}x".format(scale), True, (255, 255, 255))

            # Blit the data into the screen
            screen.blit(background, (0, 0))
            screen.blit(tempo, (5, 0))
            screen.blit(velocidade, (5, 35))

            screen.blit(escala, (5, 70))
            screen.blit(target, target_pos)
            self.blitRotateCenter(screen, rocket, position, self.gamma(timeCount))

            time_list.append(timeCount)
            x_list.append(self.x(timeCount))
            y_list.append(self.y(timeCount))
            gamma_list.append(self.gamma(timeCount))

            # if matplotlib:
            #     horizon_index = np.searchsorted(control_time, timeCount, side="left")

            #     if horizon_index > 0:
            #         horizon_index -= 1

            #     t_list = np.linspace(
            #         control_time[horizon_index],
            #         control_time[horizon_index]
            #         + self.controller_tau.N * self.controller_tau.dt,
            #         self.controller_tau.N + 1,
            #         endpoint=True,
            #     )

            #     # Plotting the two Matplotlib figures
            #     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
            #     ax1.plot(x_list, y_list)
            #     # print(horizon_list_f[horizon_index])
            #     # input("break")
            #     ax1.plot(
            #         horizon_list_tau[horizon_index][0, :],
            #         horizon_list_f[horizon_index][0, :],
            #     )
            #     ax1.set_xlabel("Horizontal Position (m)")
            #     ax1.set_ylabel("Vertical Position (m)")
            #     ax1.set_title("Rocket Trajectory")
            #     ax2.plot(time_list, gamma_list)
            #     ax2.plot(
            #         t_list, np.rad2deg(horizon_list_tau[horizon_index][2, :]) + np.pi
            #     )
            #     ax2.set_xlabel("Time (s)")
            #     ax2.set_ylabel("Yaw angle (°)")
            #     ax2.set_title("Rocket yaw angle")
            #     fig.tight_layout()

            #     # Convert the Matplotlib figure to an image
            #     canvas = agg.FigureCanvasAgg(fig)
            #     canvas.draw()
            #     renderer = canvas.get_renderer()
            #     raw_data = renderer.tostring_rgb()
            #     size = canvas.get_width_height()

            #     # Create a Pygame surface from the Matplotlib image and blit it onto the screen
            #     surf = pygame.image.fromstring(raw_data, size, "RGB")
            #     screen.blit(surf, (int(size[0] / 1.25) - 600, 100))

            pygame.display.flip()

            if save:
                pygame.image.save(screen, "../Videos/temp.jpg")
                w.append_data(imageio.imread("../Videos/temp.jpg"))

            timeCount += dt
            plt.close()

        pygame.display.quit()
        if save:
            w.close()
