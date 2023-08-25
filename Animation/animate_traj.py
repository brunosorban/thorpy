import numpy as np
from mayavi import mlab
import imageio


def animate_traj(t, x, y, z, e1bx, e1by, e1bz, e2bx, e2by, e2bz, e3bx, e3by, e3bz, trajectory_params, duration=15, save=True):
    if save:
        w = imageio.get_writer("Videos/rocket.mp4", format="FFMPEG", fps=60)
        
    fps = 60
    time_list = np.linspace(0, t[-1], duration * fps)

    x_ref = trajectory_params["x"]
    y_ref = trajectory_params["y"]
    z_ref = trajectory_params["z"]
    e1bx_ref = trajectory_params["e1bx"]
    e1by_ref = trajectory_params["e1by"]
    e1bz_ref = trajectory_params["e1bz"]
    e2bx_ref = trajectory_params["e2bx"]
    e2by_ref = trajectory_params["e2by"]
    e2bz_ref = trajectory_params["e2bz"]
    e3bx_ref = trajectory_params["e3bx"]
    e3by_ref = trajectory_params["e3by"]
    e3bz_ref = trajectory_params["e3bz"]
    ############## Define animation parameters ##############
    # Define the rocket parameters
    height_nose = 1.5
    height_body = 10
    radius_body = 0.5

    # Create the parametric equations for the rocket body and nose

    # Nose (Cone)
    u_nose = np.linspace(0, height_nose, 100)
    v = np.linspace(0, 2 * np.pi, 100)

    U_nose, V = np.meshgrid(u_nose, v)
    X_nose = height_body + U_nose
    Y_nose = (height_nose - U_nose) / height_nose * radius_body * np.sin(V)
    Z_nose = (height_nose - U_nose) / height_nose * radius_body * np.cos(V)

    # Body (Cylinder)
    u_body = np.linspace(0, height_body, 100)
    U_body, V = np.meshgrid(u_body, v)
    X_body = U_body
    Y_body = radius_body * np.sin(V)
    Z_body = radius_body * np.cos(V)

    # Calculate the center of mass (z-coordinate) for the rocket assuming uniform density
    x_cm = height_body / 2
    
    X_nose = X_nose - x_cm
    X_body = X_body - x_cm

    f = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(1280, 978))

    # Ground plane
    ground_size = 10 * height_body
    ground_x, ground_y = np.mgrid[
        -ground_size:ground_size:0.5, -ground_size:ground_size:0.5
    ]
    ground_z = np.zeros_like(ground_x) - x_cm
    mlab.mesh(ground_x, ground_y, ground_z, color=(0.5, 0.5, 0.5))
    # Plot reference trajectory
    mlab.plot3d(x_ref, y_ref, z_ref, color=(0, 0, 1), tube_radius=0.1)

    # Plot reference orientation vectors sampled every 3 seconds
    step = max(1, 3 * int(len(x_ref) / len(x)))  # Assuming equal time steps for both trajectories
    mlab.quiver3d(x_ref[::step], y_ref[::step], z_ref[::step], 
                 e1bx_ref[::step], e1by_ref[::step], e1bz_ref[::step], 
                 color=(1, 0, 0), mode='arrow', scale_factor=2)
    mlab.quiver3d(x_ref[::step], y_ref[::step], z_ref[::step], 
                 e2bx_ref[::step], e2by_ref[::step], e2bz_ref[::step], 
                 color=(0, 1, 0), mode='arrow', scale_factor=2)
    mlab.quiver3d(x_ref[::step], y_ref[::step], z_ref[::step], 
                 e3bx_ref[::step], e3by_ref[::step], e3bz_ref[::step], 
                 color=(0, 0, 1), mode='arrow', scale_factor=2)


    # Create mesh representations for the rocket's components

    rocket_nose = mlab.mesh(X_nose, Y_nose, Z_nose, color=(1, 0, 0))
    rocket_body = mlab.mesh(X_body, Y_body, Z_body, color=(0, 0, 1))
    # Ball representing the center of gravity
    # ball_diameter = 1.2 * 2 * radius_body  # Define an appropriate radius for visualization
    # ball = mlab.points3d(0, 0, 0, scale_factor=ball_diameter, color=(0, 0, 0))

    # Initial camera view settings
    azimuth = 45  # rotation around the up axis
    elevation = 1.5 * np.rad2deg(
        np.arctan(1 / np.sqrt(2))
    )  # elevation angle (90 means top-down view)
    initial_distance = 2 * ground_size  # initial distance from the focal point

    mlab.view(azimuth=azimuth, elevation=elevation, distance=initial_distance)

    ############## Define auxiliar functions ##############
    # Function to apply rotation in the body frame around the center of mass
    def apply_center_of_mass_rotation(
        X, Y, Z, rotation_matrix, x_translation, y_translation, z_translation, x_cm
    ):
        # Translate so that center of mass is at the origin
        X_origin = X
        Y_origin = Y
        Z_origin = Z

        X_rotated_translated = np.zeros_like(X)
        Y_rotated_translated = np.zeros_like(Y)
        Z_rotated_translated = np.zeros_like(Z)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                coordinates = np.array([X_origin[i, j], Y_origin[i, j], Z_origin[i, j]])
                rotated_coordinates = rotation_matrix @ coordinates
                
                X_rotated_translated[i, j] = (
                    rotated_coordinates[0] + x_translation
                )
                Y_rotated_translated[i, j] = rotated_coordinates[1] + y_translation
                Z_rotated_translated[i, j] = rotated_coordinates[2] + z_translation

        return X_rotated_translated, Y_rotated_translated, Z_rotated_translated
    
    def linear_spline(x, x_source, y_source):
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

    # Animation loop
    for i in range(len(time_list)):
        # Update positions (moving in X, Y, and Z)
        x_translation = linear_spline(time_list[i], t, x)
        y_translation = linear_spline(time_list[i], t, y)
        z_translation = linear_spline(time_list[i], t, z)
        e1bx_i = linear_spline(time_list[i], t, e1bx)
        e1by_i = linear_spline(time_list[i], t, e1by)
        e1bz_i = linear_spline(time_list[i], t, e1bz)
        e2bx_i = linear_spline(time_list[i], t, e2bx)
        e2by_i = linear_spline(time_list[i], t, e2by)
        e2bz_i = linear_spline(time_list[i], t, e2bz)
        e3bx_i = linear_spline(time_list[i], t, e3bx)
        e3by_i = linear_spline(time_list[i], t, e3by)
        e3bz_i = linear_spline(time_list[i], t, e3bz)

        # Update rotation matrix from world to body frame
        rotation_matrix = np.array(
            [
                [e1bx_i, e2bx_i, e3bx_i],
                [e1by_i, e2by_i, e3by_i],
                [e1bz_i, e2bz_i, e3bz_i],
            ]
        )

        # Normalize vectors
        rotation_matrix[0, :] = rotation_matrix[0, :] / np.linalg.norm(
            rotation_matrix[0, :]
        )
        rotation_matrix[1, :] = rotation_matrix[1, :] / np.linalg.norm(
            rotation_matrix[1, :]
        )
        rotation_matrix[2, :] = rotation_matrix[2, :] / np.linalg.norm(
            rotation_matrix[2, :]
        )

        # Update ball position
        # Update the ball position according to the center of gravity


        # Apply rotation around center of mass and translations
        X_nose_rotated, Y_nose_rotated, Z_nose_rotated = apply_center_of_mass_rotation(
            X_nose,
            Y_nose,
            Z_nose,
            rotation_matrix,
            x_translation,
            y_translation,
            z_translation,
            x_cm,
        )
        X_body_rotated, Y_body_rotated, Z_body_rotated = apply_center_of_mass_rotation(
            X_body,
            Y_body,
            Z_body,
            rotation_matrix,
            x_translation,
            y_translation,
            z_translation,
            x_cm,
        )

        # Update rocket meshes with new coordinates
        rocket_nose.mlab_source.set(
            x=X_nose_rotated, y=Y_nose_rotated, z=Z_nose_rotated
        )
        rocket_body.mlab_source.set(
            x=X_body_rotated, y=Y_body_rotated, z=Z_body_rotated
        )
        # ball.mlab_source.set(x=x_translation, y=y_translation, z=z_translation)

        mlab.process_ui_events()
        
        if save:
            mlab.savefig(filename="Videos/temp.jpg")
            w.append_data(imageio.imread("Videos/temp.jpg"))
            
    if save:
        w.close()
