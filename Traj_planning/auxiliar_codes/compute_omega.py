def compute_omega(x_dot_dot, y_dot_dot, z_dot_dot, x_3dot, y_3dot, z_3dot, g):
    """
    This function computes the omega_dot vector written in the world frame.

    Args:
        x_dot_dot (float): x acceleration.
        y_dot_dot (float): y acceleration.
        z_dot_dot (float): z acceleration.
        x_3dot (float): x jerk.
        y_3dot (float): y jerk.
        z_3dot (float): z jerk.
        g (float): gravity.

    Returns:
        e (list): omega vector."""

    temp = (g**2 + 2 * g * z_dot_dot + x_dot_dot**2 + y_dot_dot**2 + z_dot_dot**2)
    wx = (-g * y_3dot - y_3dot * z_dot_dot + y_dot_dot * z_3dot)
    wy = (g * x_3dot + x_3dot * z_dot_dot - x_dot_dot * z_3dot)
    wz = (-x_3dot * y_dot_dot + x_dot_dot * y_3dot)
    e = [wx / temp, wy / temp, wz / temp]
    return e
