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

    beta_dot = (
        -2
        * (x_dot_dot * x_3dot + y_dot_dot * y_3dot + (z_dot_dot + g) * z_3dot)
        / (x_dot_dot**2 + y_dot_dot**2 + (z_dot_dot + g) ** 2) ** 2
    )
    wx_dot = -(z_dot_dot + g) * y_4dot + y_dot_dot * z_4dot
    wy_dot = (z_dot_dot + g) * x_4dot - x_dot_dot * z_4dot
    wz_dot = -y_dot_dot * x_4dot + x_dot_dot * y_4dot
    e = [beta_dot * wx_dot, beta_dot * wy_dot, beta_dot * wz_dot]

    return e
