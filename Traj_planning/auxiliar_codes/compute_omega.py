import casadi as ca
import numpy as np


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

    t = ca.vertcat(x_dot_dot, y_dot_dot, z_dot_dot + g)

    wx = -g * y_3dot - y_3dot * z_dot_dot + y_dot_dot * z_3dot
    wy = g * x_3dot + x_3dot * z_dot_dot - x_dot_dot * z_3dot
    wz = -x_3dot * y_dot_dot + x_dot_dot * y_3dot
    e = ca.vertcat(
        wx / ca.norm_2(t) ** 2, wy / ca.norm_2(t) ** 2, wz / ca.norm_2(t) ** 2
    )

    return e


def compute_omega_np(x_dot_dot, y_dot_dot, z_dot_dot, x_3dot, y_3dot, z_3dot, g):
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

    t = np.array([x_dot_dot, y_dot_dot, z_dot_dot + g])

    wx = -g * y_3dot - y_3dot * z_dot_dot + y_dot_dot * z_3dot
    wy = g * x_3dot + x_3dot * z_dot_dot - x_dot_dot * z_3dot
    wz = -x_3dot * y_dot_dot + x_dot_dot * y_3dot
    e = np.array(
        [
            wx / np.linalg.norm(t) ** 2,
            wy / np.linalg.norm(t) ** 2,
            wz / np.linalg.norm(t) ** 2,
        ]
    )
    return e
