import casadi as ca
import numpy as np


def compute_omega_dot_dot(
    x_dot_dot,
    y_dot_dot,
    z_dot_dot,
    x_3dot,
    y_3dot,
    z_3dot,
    x_4dot,
    y_4dot,
    z_4dot,
    x_5dot,
    y_5dot,
    z_5dot,
    g,
):
    """
    This function computes the angular velocity second derivative (omega_dot_dot) vector written in the world frame.

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
        x_5dot (float): x crackle.
        y_5dot (float): y crackle.
        z_5dot (float): z crackle.
        g (float): gravity.

    Returns:
        e (list): angular velocity second derivative (omega_dot_dot) vector."""

    t = ca.vertcat(x_dot_dot, y_dot_dot, z_dot_dot + g)

    beta_dot = -2 * (x_dot_dot * x_3dot + y_dot_dot * y_3dot + (z_dot_dot + g) * z_3dot) / ca.norm_2(t) ** 4

    beta_dot_dot = (
        2
        * (
            4 * (x_dot_dot**2 * x_3dot + y_dot_dot**2 * y_3dot + (z_dot_dot + g) ** 2 * z_3dot) ** 2
            - (x_dot_dot**2 + y_dot_dot**2 + (z_dot_dot + g) ** 2)
            * (
                x_dot_dot * x_4dot
                + y_dot_dot * y_4dot
                + (z_dot_dot + g) * z_4dot
                + x_3dot**2
                + y_3dot**2
                + z_3dot**2
            )
        )
    ) / ca.norm_2(t) ** 6

    wx_dot_dot = (-z_3dot * y_4dot - (z_dot_dot + g) * y_5dot + y_3dot * z_4dot + y_dot_dot * z_5dot) * beta_dot + (
        -(z_dot_dot + g) * y_4dot + y_dot_dot * z_4dot
    ) * beta_dot_dot
    wy_dot_dot = (z_3dot * x_4dot + (z_dot_dot + g) * x_5dot - x_3dot * z_4dot - x_dot_dot * z_5dot) * beta_dot + (
        (z_dot_dot + g) * x_4dot - x_dot_dot * z_4dot
    ) * beta_dot_dot
    wz_dot_dot = (-y_3dot * x_4dot - y_dot_dot * x_5dot + x_3dot * y_4dot + x_dot_dot * y_5dot) * beta_dot + (
        -y_dot_dot * x_4dot + x_dot_dot * y_4dot
    ) * beta_dot_dot

    e = ca.vertcat(beta_dot_dot * wx_dot_dot, beta_dot_dot * wy_dot_dot, beta_dot_dot * wz_dot_dot)

    return e


def compute_omega_dot_dot_np(
    x_dot_dot,
    y_dot_dot,
    z_dot_dot,
    x_3dot,
    y_3dot,
    z_3dot,
    x_4dot,
    y_4dot,
    z_4dot,
    x_5dot,
    y_5dot,
    z_5dot,
    g,
):
    """
    This function computes the angular velocity second derivative (omega_dot_dot) vector written in the world frame.
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
        x_5dot (float): x crackle.
        y_5dot (float): y crackle.
        z_5dot (float): z crackle.
        g (float): gravity.

    Returns:
        e (list): angular velocity second derivative (omega_dot_dot) vector."""

    t = np.array([x_dot_dot, y_dot_dot, z_dot_dot + g])

    beta_dot = -2 * (x_dot_dot * x_3dot + y_dot_dot * y_3dot + (z_dot_dot + g) * z_3dot) / np.linalg.norm(t) ** 4

    beta_dot_dot = (
        2
        * (
            4 * (x_dot_dot**2 * x_3dot + y_dot_dot**2 * y_3dot + (z_dot_dot + g) ** 2 * z_3dot) ** 2
            - (x_dot_dot**2 + y_dot_dot**2 + (z_dot_dot + g) ** 2)
            * (
                x_dot_dot * x_4dot
                + y_dot_dot * y_4dot
                + (z_dot_dot + g) * z_4dot
                + x_3dot**2
                + y_3dot**2
                + z_3dot**2
            )
        )
    ) / np.linalg.norm(t) ** 6

    wx_dot_dot = (-z_3dot * y_4dot - (z_dot_dot + g) * y_5dot + y_3dot * z_4dot + y_dot_dot * z_5dot) * beta_dot + (
        -(z_dot_dot + g) * y_4dot + y_dot_dot * z_4dot
    ) * beta_dot_dot
    wy_dot_dot = (z_3dot * x_4dot + (z_dot_dot + g) * x_5dot - x_3dot * z_4dot - x_dot_dot * z_5dot) * beta_dot + (
        (z_dot_dot + g) * x_4dot - x_dot_dot * z_4dot
    ) * beta_dot_dot
    wz_dot_dot = (-y_3dot * x_4dot - y_dot_dot * x_5dot + x_3dot * y_4dot + x_dot_dot * y_5dot) * beta_dot + (
        -y_dot_dot * x_4dot + x_dot_dot * y_4dot
    ) * beta_dot_dot

    e = np.array(
        [
            beta_dot_dot * wx_dot_dot,
            beta_dot_dot * wy_dot_dot,
            beta_dot_dot * wz_dot_dot,
        ]
    )

    return e
