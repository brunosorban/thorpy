import casadi as ca
import numpy as np


def compute_omega_dot(
    x_dot_dot, y_dot_dot, z_dot_dot, x_3dot, y_3dot, z_3dot, x_4dot, y_4dot, z_4dot, g
):
    """
    This function computes the angular acceleration (omega_dot) vector written in the world frame.

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
        e (list): angular acceleration (omega_dot) vector."""

    t = ca.vertcat(x_dot_dot, y_dot_dot, z_dot_dot + g)

    beta_dot = (
        -2
        * (x_dot_dot * x_3dot + y_dot_dot * y_3dot + (z_dot_dot + g) * z_3dot)
        / ca.norm_2(t) ** 4
    )

    wx = -g * y_3dot - y_3dot * z_dot_dot + y_dot_dot * z_3dot
    wy = g * x_3dot + x_3dot * z_dot_dot - x_dot_dot * z_3dot
    wz = -x_3dot * y_dot_dot + x_dot_dot * y_3dot

    wx_dot = -(z_dot_dot + g) * y_4dot + y_dot_dot * z_4dot
    wy_dot = (z_dot_dot + g) * x_4dot - x_dot_dot * z_4dot
    wz_dot = -y_dot_dot * x_4dot + x_dot_dot * y_4dot

    e = ca.vertcat(
        wx_dot / ca.norm_2(t) ** 2 + wx * beta_dot,
        wy_dot / ca.norm_2(t) ** 2 + wy * beta_dot,
        wz_dot / ca.norm_2(t) ** 2 + wz * beta_dot,
    )

    return e


def compute_omega_dot_np(
    x_dot_dot, y_dot_dot, z_dot_dot, x_3dot, y_3dot, z_3dot, x_4dot, y_4dot, z_4dot, g
):
    """
    This function computes the angular acceleration (omega_dot) vector written in the world frame.
    Numpy version.

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
        e (list): angular acceleration (omega_dot) vector."""

    t = np.array([x_dot_dot, y_dot_dot, z_dot_dot + g])

    beta_dot = (
        -2
        * (x_dot_dot * x_3dot + y_dot_dot * y_3dot + (z_dot_dot + g) * z_3dot)
        / np.linalg.norm(t) ** 4
    )

    wx = -g * y_3dot - y_3dot * z_dot_dot + y_dot_dot * z_3dot
    wy = g * x_3dot + x_3dot * z_dot_dot - x_dot_dot * z_3dot
    wz = -x_3dot * y_dot_dot + x_dot_dot * y_3dot

    wx_dot = -(z_dot_dot + g) * y_4dot + y_dot_dot * z_4dot
    wy_dot = (z_dot_dot + g) * x_4dot - x_dot_dot * z_4dot
    wz_dot = -y_dot_dot * x_4dot + x_dot_dot * y_4dot
    e = [
        wx_dot / np.linalg.norm(t) ** 2 + wx * beta_dot,
        wy_dot / np.linalg.norm(t) ** 2 + wy * beta_dot,
        wz_dot / np.linalg.norm(t) ** 2 + wz * beta_dot,
    ]

    return e
