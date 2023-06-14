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


def draw_vector(initial_point, length, arrow_length, ei, color, screen):
    """
    Draw a vector with initial point and length
    """
    pygame.draw.line(
        screen,
        color,
        initial_point,
        (initial_point[0] + length * ei[0], initial_point[1] + length * ei[1]),
        2,
    )

    arrow_points = [
        (initial_point[0] + length * ei[0], initial_point[1] + length * ei[1]),
        (
            initial_point[0]
            + length * ei[0]
            - arrow_length * ei[0]
            + arrow_length * ei[1],
            initial_point[1]
            + length * ei[1]
            - arrow_length * ei[1]
            - arrow_length * ei[0],
        ),
        (
            initial_point[0]
            + length * ei[0]
            - arrow_length * ei[0]
            - arrow_length * ei[1],
            initial_point[1]
            + length * ei[1]
            - arrow_length * ei[1]
            + arrow_length * ei[0],
        ),
    ]

    pygame.draw.polygon(screen, color, arrow_points)


def animate(
    t,
    x,
    y,
    gamma,
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

    # animation constants
    offset = 0.2 * max(max(abs(x)), max(abs(y)))
    dt = 1 / 60
    target_goal = (30, 30) if target_goal == False else target_goal

    # Initialization
    xlim_inf = min(x)
    xlim_sup = max(x) + offset
    ylim_inf = min(y)
    ylim_sup = max(y) + offset
    
    x -= xlim_inf
    y -= ylim_inf
    
    xlim_inf = -offset
    ylim_inf = 0
    xlim_sup = max(x) + offset
    ylim_sup = max(y) + offset

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

    # retrive control horizons
    horizon_list_x = []
    horizon_list_y = []
    horizon_list_gamma = []
    horizon_list_e1bx = []
    horizon_list_e1by = []
    horizon_list_e2bx = []
    horizon_list_e2by = []

    for horizon in state_horizon_list:
        horizon_list_x.append(horizon[0, :])
        horizon_list_y.append(horizon[2, :])
        horizon_list_gamma.append(np.arctan2(horizon[5, :], horizon[4, :]))
        horizon_list_e1bx.append(horizon[4, :])
        horizon_list_e1by.append(horizon[5, :])
        horizon_list_e2bx.append(horizon[6, :])
        horizon_list_e2by.append(horizon[7, :])

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
        blitRotateCenter(screen, rocket, position, f_gamma(timeCount))

        # plot the horizon as dots for the position in x and y
        if state_horizon_list is not None:
            horizon_index = np.searchsorted(t, timeCount, side="left")
            if horizon_index > 0:
                horizon_index -= 1

            for horizon in state_horizon_list:
                for i in range(len(horizon_list_x[horizon_index])):
                    horizon_pos = (
                        initial_position
                        + mapi(
                            horizon_list_x[horizon_index][i],
                            -horizon_list_y[horizon_index][i],
                            xlim_sup,
                            ylim_sup,
                            xlim_inf,
                            ylim_inf,
                            size,
                            realScale=True,
                        )
                        + np.array([151 / 2, 0])
                    )

                    pygame.draw.circle(screen, (255, 0, 0), horizon_pos[:], 2)

                    if i % 90 == 0:
                        vec_len = 25
                        arrow_len = 5
                        # initial_point, length, arrow_length, ei, color, screen
                        draw_vector(
                            horizon_pos,
                            vec_len,
                            arrow_len,
                            (
                                horizon_list_e1bx[horizon_index][i],
                                -horizon_list_e1by[horizon_index][i],
                            ),
                            (0, 255, 0),
                            screen,
                        )
                        draw_vector(
                            horizon_pos,
                            vec_len,
                            arrow_len,
                            (
                                horizon_list_e2bx[horizon_index][i],
                                -horizon_list_e2by[horizon_index][i],
                            ),
                            (0, 0, 255),
                            screen,
                        )

            # plot the trajectory as dots for the position in x and y
            t_traj = trajectory_params["t"]
            x_traj = trajectory_params["x"]
            y_traj = trajectory_params["y"]
            e1bx_traj = trajectory_params["e1bx"]
            e1by_traj = trajectory_params["e1by"]
            e2bx_traj = trajectory_params["e2bx"]
            e2by_traj = trajectory_params["e2by"]

            for i in range(len(t_traj)):
                traj_pos = (
                    initial_position
                    + mapi(
                        x_traj[i],
                        -y_traj[i],
                        xlim_sup,
                        ylim_sup,
                        xlim_inf,
                        ylim_inf,
                        size,
                        realScale=True,
                    )
                    + np.array([151 / 2, 0])
                )

                pygame.draw.circle(screen, (0, 0, 0), traj_pos[:], 2)

                if i % 90 == 0:
                    vec_len = 25
                    arrow_len = 5
                    # initial_point, length, arrow_length, ei, color, screen
                    draw_vector(
                        traj_pos,
                        vec_len,
                        arrow_len,
                        (
                            e1bx_traj[i],
                            -e1by_traj[i],
                        ),
                        (0, 255, 0),
                        screen,
                    )
                    draw_vector(
                        traj_pos,
                        vec_len,
                        arrow_len,
                        (
                            e2bx_traj[i],
                            -e2by_traj[i],
                        ),
                        (0, 0, 255),
                        screen,
                    )

        time_list.append(timeCount)
        x_list.append(f_x(timeCount))
        y_list.append(f_y(timeCount))
        gamma_list.append(f_gamma(timeCount))

        if matplotlib:
            horizon_list_x = np.array(horizon_list_x)
            horizon_list_y = np.array(horizon_list_y)
            horizon_list_gamma = np.array(horizon_list_gamma)

            horizon_index = np.searchsorted(t, timeCount, side="left")

            if horizon_index > 0:
                horizon_index -= 1

            # Plotting the two Matplotlib figures
            fig, ax1 = plt.subplots(1, 1, figsize=(6, 9))
            ax1.plot(x_list, y_list)
            ax1.scatter(
                horizon_list_x[horizon_index],
                horizon_list_y[horizon_index],
                s=2,
                color="orange",
            )

            # plot only every 8th vector
            for ind_plot in range(0, len(horizon_list_x[0]), 8):
                if ind_plot < len(horizon_list_x[0]):
                    ax1.quiver(
                        horizon_list_x[horizon_index][ind_plot],
                        horizon_list_y[horizon_index][ind_plot],
                        horizon_list_e1bx[horizon_index][ind_plot],
                        horizon_list_e1by[horizon_index][ind_plot],
                        color="r",
                        scale=15,
                    )
                    ax1.quiver(
                        horizon_list_x[horizon_index][ind_plot],
                        horizon_list_y[horizon_index][ind_plot],
                        horizon_list_e2bx[horizon_index][ind_plot],
                        horizon_list_e2by[horizon_index][ind_plot],
                        color="b",
                        scale=15,
                    )

            ax1.set_xlabel("Horizontal Position (m)")
            ax1.set_ylabel("Vertical Position (m)")
            ax1.set_title("Rocket Trajectory")
            ax1.set_xlim([xlim_inf, xlim_sup])
            ax1.set_ylim([ylim_inf, ylim_sup])
            ax1.axis("equal")

            # ax2.plot(time_list, gamma_list)
            # ax2.plot(t_list, np.rad2deg(horizon_list_gamma[horizon_index]))
            # ax2.set_xlabel("Time (s)")
            # ax2.set_ylabel("Yaw angle (°)")
            # ax2.set_title("Rocket yaw angle")
            # ax2.set_ylim([gamma_lin_inf, gamma_lin_sup])
            # fig.tight_layout()

            # Convert the Matplotlib figure to an image
            canvas = agg.FigureCanvasAgg(fig)
            canvas.draw()
            renderer = canvas.get_renderer()
            raw_data = renderer.tostring_rgb()
            size_2 = canvas.get_width_height()

            # Create a Pygame surface from the Matplotlib image and blit it onto the screen
            surf = pygame.image.fromstring(raw_data, size_2, "RGB")
            screen.blit(surf, (int(size[0]) - 600, -100))

        pygame.display.flip()

        if save:
            pygame.image.save(screen, "Videos/temp.jpg")
            w.append_data(imageio.imread("Videos/temp.jpg"))

        timeCount += dt
        plt.close()

    pygame.display.quit()
    if save:
        w.close()
