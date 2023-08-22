def compute_omega_dot(
    x_dot_dot, y_dot_dot, z_dot_dot, x_3dot, y_3dot, z_3dot, x_4dot, y_4dot, z_4dot, g
):
    """
    This function computes the omega_dot vector written in the world frame.

    Args:
        x_dot_dot (float): x acceleration.
        y_dot_dot (float): y acceleration.
        z_dot_dot (float): z acceleration.
        x_3dot (float): x jerk.
        y_3dot (float): y jerk.
        z_3dot (float): z jerk.
        x_4dot (float): x snap.
        y_4dot (float): y snap.
        z_4dot (float): z snap.
        g (float): gravity.

    Returns:
        e (list): omega_dot vector."""
    e = [
        2
        * g**2
        * y_3dot
        * z_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        + 2
        * g
        * x_dot_dot
        * y_3dot
        * x_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        + 2
        * g
        * y_3dot
        * y_dot_dot
        * y_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        + 4
        * g
        * y_3dot
        * z_dot_dot
        * z_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        - 2
        * g
        * y_dot_dot
        * z_3dot
        * z_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        - g
        * y_4dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        + 2
        * x_dot_dot
        * y_3dot
        * z_dot_dot
        * x_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        - 2
        * x_dot_dot
        * y_dot_dot
        * z_3dot
        * x_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        + 2
        * y_3dot
        * y_dot_dot
        * z_dot_dot
        * y_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        + 2
        * y_3dot
        * z_dot_dot**2
        * z_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        - 2
        * y_dot_dot**2
        * z_3dot
        * y_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        - 2
        * y_dot_dot
        * z_3dot
        * z_dot_dot
        * z_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        - y_3dot
        * z_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        + y_dot_dot
        * z_4dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        + z_3dot
        * y_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        - z_dot_dot
        * y_4dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        ),
        -2
        * g**2
        * x_3dot
        * z_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        - 2
        * g
        * x_3dot
        * x_dot_dot
        * x_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        - 2
        * g
        * x_3dot
        * y_dot_dot
        * y_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        - 4
        * g
        * x_3dot
        * z_dot_dot
        * z_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        + 2
        * g
        * x_dot_dot
        * z_3dot
        * z_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        + g
        * x_4dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        - 2
        * x_3dot
        * x_dot_dot
        * z_dot_dot
        * x_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        - 2
        * x_3dot
        * y_dot_dot
        * z_dot_dot
        * y_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        - 2
        * x_3dot
        * z_dot_dot**2
        * z_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        + 2
        * x_dot_dot**2
        * z_3dot
        * x_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        + 2
        * x_dot_dot
        * y_dot_dot
        * z_3dot
        * y_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        + 2
        * x_dot_dot
        * z_3dot
        * z_dot_dot
        * z_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        + x_3dot
        * z_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        - x_dot_dot
        * z_4dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        - z_3dot
        * x_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        + z_dot_dot
        * x_4dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        ),
        2
        * g
        * x_3dot
        * y_dot_dot
        * z_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        - 2
        * g
        * x_dot_dot
        * y_3dot
        * z_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        + 2
        * x_3dot
        * x_dot_dot
        * y_dot_dot
        * x_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        + 2
        * x_3dot
        * y_dot_dot**2
        * y_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        + 2
        * x_3dot
        * y_dot_dot
        * z_dot_dot
        * z_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        - 2
        * x_dot_dot**2
        * y_3dot
        * x_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        - 2
        * x_dot_dot
        * y_3dot
        * y_dot_dot
        * y_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        - 2
        * x_dot_dot
        * y_3dot
        * z_dot_dot
        * z_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        ** 2
        - x_3dot
        * y_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        + x_dot_dot
        * y_4dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        + y_3dot
        * x_3dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        )
        - y_dot_dot
        * x_4dot
        / (
            g**2
            + 2 * g * z_dot_dot
            + x_dot_dot**2
            + y_dot_dot**2
            + z_dot_dot**2
        ),
    ]

    return e
