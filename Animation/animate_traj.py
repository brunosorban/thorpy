import numpy as np
from mayavi import mlab


def animate_traj(x, y, z, e1bx, e1by, e1bz, e2bx, e2by, e2bz, e3bx, e3by, e3bz):
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
    # x_cm = (height_nose * (height_nose / 2) + height_body * (height_body / 2 + height_nose)) / (height_nose + height_body)
    x_cm = height_body / 2

    f = mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(800, 800))

    # Ground plane
    ground_size = 10 * height_body
    ground_x, ground_y = np.mgrid[
        -ground_size:ground_size:0.5, -ground_size:ground_size:0.5
    ]
    ground_z = np.zeros_like(ground_x) - x_cm
    mlab.mesh(ground_x, ground_y, ground_z, color=(0.5, 0.5, 0.5))

    # Create mesh representations for the rocket's components
    rocket_nose = mlab.mesh(X_nose, Y_nose, Z_nose, color=(1, 0, 0))
    rocket_body = mlab.mesh(X_body, Y_body, Z_body, color=(0, 0, 1))

    # Initial camera view settings
    azimuth = 45  # rotation around the up axis
    elevation = 1.5 * np.rad2deg(
        np.arctan(1 / np.sqrt(2))
    )  # elevation angle (90 means top-down view)
    initial_distance = 3 * ground_size  # initial distance from the focal point

    mlab.view(azimuth=azimuth, elevation=elevation, distance=initial_distance)

    ############## Define auxiliar functions ##############
    # Function to apply rotation in the body frame around the center of mass
    def apply_center_of_mass_rotation(
        X, Y, Z, rotation_matrix, x_translation, y_translation, z_translation, x_cm
    ):
        # Translate so that center of mass is at the origin
        X_origin = X - x_cm
        Y_origin = Y
        Z_origin = Z

        # print("X_origin: ", X_origin)
        # print("X_origin.ravel(): ", X_origin.ravel())
        # # Apply rotation
        # coordinates = np.array([X_origin.ravel(), Y_origin.ravel(), Z_origin.ravel()])
        # rotated_coordinates = rotation_matrix @ coordinates

        # # Translate back
        # X_rotated_translated = rotated_coordinates[0].reshape(X.shape) + x_translation
        # Y_rotated_translated = rotated_coordinates[1].reshape(Y.shape) + y_translation
        # Z_rotated_translated = rotated_coordinates[2].reshape(Z.shape) + z_cm + z_translation

        X_rotated_translated = np.zeros_like(X)
        Y_rotated_translated = np.zeros_like(Y)
        Z_rotated_translated = np.zeros_like(Z)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                coordinates = np.array([X_origin[i, j], Y_origin[i, j], Z_origin[i, j]])
                rotated_coordinates = rotation_matrix @ coordinates
                X_rotated_translated[i, j] = (
                    rotated_coordinates[0] + x_cm + x_translation
                )
                Y_rotated_translated[i, j] = rotated_coordinates[1] + y_translation
                Z_rotated_translated[i, j] = rotated_coordinates[2] + z_translation

        return X_rotated_translated, Y_rotated_translated, Z_rotated_translated

    # Animation loop
    for i in range(len(x)):
        # Update positions (moving in X, Y, and Z)
        x_translation = x[i]
        y_translation = y[i]  # Example translation in Y
        z_translation = z[i]  # Translation in Z (moving upwards)

        # Update rotation matrix from world to body frame
        rotation_matrix = np.array(
            [
                [e1bx[i], e2bx[i], e3bx[i]],
                [e1by[i], e2by[i], e3by[i]],
                [e1bz[i], e2bz[i], e3bz[i]],
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

        mlab.process_ui_events()
