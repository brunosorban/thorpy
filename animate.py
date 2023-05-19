import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import sys, pygame  # For the animation function
import imageio  # For the animation function
from Function import Function

# Animation
def blitRotateCenter(surf, image, topleft, angle):
    # Auxiliar function to blit the animation image rotated
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(topleft=topleft).center)
    surf.blit(rotated_image, new_rect.topleft)


def mapi(x, y, xlim_sup, ylim_sup, xlim_inf, ylim_inf, size, realScale=True):
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


def animate(
    t,
    x,
    y,
    gamma,
    delta_tvc,
    state_horizon_list,
    control_horizon_list=None,
    N=False,
    dt=False,
    target_goal=False,
    trajectory_params=False,
    save=False,
    scale=1,
    matplotlib=False,
):
    timeMax = t[-1]
    f_x = Function(t, x)
    f_y = Function(t, y)
    f_gamma = Function(t, np.rad2deg(gamma))
    f_delta_tvc_c = Function(t, np.rad2deg(delta_tvc))

    # animation constants
    offset = 0.2 * max(max(x), max(y))
    dt = 1 / 60 * 3
    target_goal = (30, 30) if target_goal == False else target_goal
    
    if trajectory_params is False:
        following_trajectory = False
    else:
        following_trajectory = True
        f_x_traj = Function(trajectory_params["t"], trajectory_params["x"])
        f_y_traj = Function(trajectory_params["t"], trajectory_params["y"])

    # Initialization
    xlim_inf = abs(min(x))
    xlim_sup = max(x) + offset
    ylim_inf = abs(min(y))
    ylim_sup = max(y) + offset
    gamma_lin_inf = np.rad2deg(abs(min(gamma))) - 5
    gamma_lin_sup = np.rad2deg(max(gamma)) + 5

    pygame.init()
    pygame.display.init()
    font = pygame.font.SysFont("Helvetica", 32)

    if save:
        w = imageio.get_writer("Videos/rocket.mp4", format="FFMPEG", fps=60)

    # Creating animation screen
    size = np.array([int(1920 / 1.25), int(1080 / 1.25)])
    # size = np.array([int(3840 / 1.5), int(2160 / 1.5)])
    screen = pygame.display.set_mode(size)

    # Preparing images
    background = pygame.image.load("Animation/sky.jpg")
    background = pygame.transform.scale(background, (size))

    rocket = pygame.image.load("Animation/rocket.png")
    rocket = pygame.transform.scale(rocket, (int(size[0] / 10), int((151 / 10))))
    # rocket = pygame.transform.scale(
    #     rocket, (int(size[0] / 10), int((3840 / 1920) * 151 / 10))
    # )

    target = pygame.image.load("Animation/target.png")
    target = pygame.transform.scale(target, (40, 40))

    # nozzle: 2000x731
    nozzle = pygame.image.load("Animation/nozzle.png")
    nozzle_scale = rocket.get_size()[1] / nozzle.get_size()[1]
    nozzle = pygame.transform.scale(
        nozzle,
        (
            int(nozzle_scale * nozzle.get_size()[0]),
            int(nozzle_scale * nozzle.get_size()[1]),
        ),
    )

    # flame: 2000x731
    flame = pygame.image.load("Animation/flame.png")
    flame_scale = rocket.get_size()[1] / flame.get_size()[1]
    flame = pygame.transform.scale(
        flame,
        (
            int(flame_scale * flame.get_size()[0]),
            int(flame_scale * flame.get_size()[1]),
        ),
    )

    # Initialize position vectors
    initial_position = np.array([0, size[1] - 128])
    position = initial_position

    timeCount = 0
    time_list = [0]
    x_list = [f_x(0)]
    y_list = [f_y(0)]
    gamma_list = [f_gamma(0)]

    # Iteration in time
    while timeCount <= timeMax:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        # Get position, current time and vehicle velocity
        position = initial_position + mapi(
            f_x(timeCount),
            -f_y(timeCount),
            xlim_sup,
            ylim_sup,
            xlim_inf,
            ylim_inf,
            size,
            realScale=True,
        )

        if following_trajectory:
            # Get position, current time and vehicle velocity
            target_pos = initial_position + mapi(
                f_x_traj(timeCount),
                -f_y_traj(timeCount),
                xlim_sup,
                ylim_sup,
                xlim_inf,
                ylim_inf,
                size,
                realScale=True,
            )
        else:
            # Get position, current time and vehicle velocity
            target_pos = initial_position + mapi(
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

        tempo = font.render("Time : {:.1f}s".format(timeCount), True, (255, 255, 255))

        # velocidade = font.render(
        #     "Speed : {:.1f}m/s".format(v(timeCount)),
        #     True,
        #     (255, 255, 255),
        # )

        # escala = font.render("Scale : {:.1f}x".format(scale), True, (255, 255, 255))

        # Blit the data into the screen
        screen.blit(background, (0, 0))
        screen.blit(tempo, (5, 0))
        # screen.blit(velocidade, (5, 35))

        # screen.blit(escala, (5, 70))
        screen.blit(target, target_pos)
        blitRotateCenter(screen, rocket, position, f_gamma(timeCount))
        # pos_nozzle = (
        #     position  # + np.array([rocket.get_width() / 2, -rocket.get_height()])
        # )
        # blitRotateCenter(screen, nozzle, pos_nozzle, f_gamma(timeCount) + 180)
        # pos_flame = position + np.array(
        #     [
        #         rocket.get_width() / 2 * np.cos(np.deg2rad(f_gamma(timeCount)))
        #         - rocket.get_height() * np.sin(np.deg2rad(f_gamma(timeCount))),
        #         rocket.get_width() / 2 * np.sin(np.deg2rad(f_gamma(timeCount)))
        #         + rocket.get_height() * np.cos(np.deg2rad(f_gamma(timeCount))),
        #     ]
        # )
        # blitRotateCenter(screen, flame, pos_flame, f_gamma(timeCount) + 180)

        time_list.append(timeCount)
        x_list.append(f_x(timeCount))
        y_list.append(f_y(timeCount))
        gamma_list.append(f_gamma(timeCount))

        if matplotlib:
            # retrive control horizons
            horizon_list_x = []
            horizon_list_y = []
            horizon_list_gamma = []

            for horizon in state_horizon_list:
                horizon_list_x.append(horizon[0, :])
                horizon_list_y.append(horizon[2, :])
                horizon_list_gamma.append(horizon[4, :])

            horizon_list_f = []
            horizon_list_tau = []

            for horizon in control_horizon_list:
                horizon_list_f.append(horizon[0, :])
                horizon_list_tau.append(horizon[1, :])

            horizon_list_x = np.array(horizon_list_x)
            horizon_list_y = np.array(horizon_list_y)
            horizon_list_gamma = np.array(horizon_list_gamma)
            horizon_list_f = np.array(horizon_list_f)
            horizon_list_tau = np.array(horizon_list_tau)

            horizon_index = np.searchsorted(t, timeCount, side="left")

            if horizon_index > 0:
                horizon_index -= 1

            t_list = np.linspace(
                t[horizon_index],
                t[horizon_index] + N * dt,
                N + 1,
                endpoint=True,
            )

            # Plotting the two Matplotlib figures
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
            ax1.plot(x_list, y_list)
            ax1.plot(
                horizon_list_x[horizon_index],
                horizon_list_y[horizon_index],
            )
            ax1.set_xlabel("Horizontal Position (m)")
            ax1.set_ylabel("Vertical Position (m)")
            ax1.set_title("Rocket Trajectory")
            ax1.set_xlim([xlim_inf, xlim_sup])
            ax1.set_ylim([ylim_inf, ylim_sup])
            ax1.axis("equal")

            ax2.plot(time_list, gamma_list)
            ax2.plot(t_list, np.rad2deg(horizon_list_gamma[horizon_index]))
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Yaw angle (°)")
            ax2.set_title("Rocket yaw angle")
            ax2.set_ylim([gamma_lin_inf, gamma_lin_sup])
            fig.tight_layout()

            # Convert the Matplotlib figure to an image
            canvas = agg.FigureCanvasAgg(fig)
            canvas.draw()
            renderer = canvas.get_renderer()
            raw_data = renderer.tostring_rgb()
            size_2 = canvas.get_width_height()

            # Create a Pygame surface from the Matplotlib image and blit it onto the screen
            surf = pygame.image.fromstring(raw_data, size_2, "RGB")
            screen.blit(surf, (int(size[0]) - 600, 100))

        pygame.display.flip()

        if save:
            pygame.image.save(screen, "Videos/temp.jpg")
            w.append_data(imageio.imread("Videos/temp.jpg"))

        timeCount += dt
        plt.close()

    pygame.display.quit()
    if save:
        w.close()
