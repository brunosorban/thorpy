import numpy as np


def get_pos(coefs, t):
    return (
        coefs[0] * t**6
        + coefs[1] * t**5
        + coefs[2] * t**4
        + coefs[3] * t**3
        + coefs[4] * t**2
        + coefs[5] * t
        + coefs[6]
    )


def get_vel(coefs, t):
    return (
        6 * coefs[0] * t**5
        + 5 * coefs[1] * t**4
        + 4 * coefs[2] * t**3
        + 3 * coefs[3] * t**2
        + 2 * coefs[4] * t
        + coefs[5]
    )


def get_acc(coefs, t):
    return (
        30 * coefs[0] * t**4
        + 20 * coefs[1] * t**3
        + 12 * coefs[2] * t**2
        + 6 * coefs[3] * t
        + 2 * coefs[4]
    )


def get_jerk(coefs, t):
    return (
        120 * coefs[0] * t**3
        + 60 * coefs[1] * t**2
        + 24 * coefs[2] * t
        + 6 * coefs[3]
    )


def get_snap(coefs, t):
    return 360 * coefs[0] * t**2 + 120 * coefs[1] * t + 24 * coefs[2]


def get_gamma_3dot(states):
    g = states["g"]
    x_dot_dot = states["ax_o"]
    y_dot_dot = states["ay_o"]
    x_3dot = states["jx_o"]
    y_3dot = states["jy_o"]
    x_4dot = states["sx_o"]
    y_4dot = states["sy_o"]
    x_5dot = states["cx_o"]
    y_5dot = states["cy_o"]
    gamma_3dot = (
        -(-(g + y_dot_dot) * x_3dot + x_dot_dot * y_3dot)
        * (-4 * (g + y_dot_dot) * y_3dot - 4 * x_dot_dot * x_3dot)
        * ((2 * g + 2 * y_dot_dot) * y_3dot + 2 * x_3dot * x_dot_dot)
        / ((g + y_dot_dot) ** 2 + x_dot_dot**2) ** 3
        - (-(g + y_dot_dot) * x_3dot + x_dot_dot * y_3dot)
        * (
            (2 * g + 2 * y_dot_dot) * y_4dot
            + 2 * x_3dot * x_3dot
            + 2 * x_dot_dot * x_4dot
            + 2 * y_3dot * y_3dot
        )
        / ((g + y_dot_dot) ** 2 + x_dot_dot**2) ** 2
        + (-(g + y_dot_dot) * x_4dot + x_dot_dot * y_4dot)
        * (-2 * (g + y_dot_dot) * y_3dot - 2 * x_dot_dot * x_3dot)
        / ((g + y_dot_dot) ** 2 + x_dot_dot**2) ** 2
        - ((2 * g + 2 * y_dot_dot) * y_3dot + 2 * x_3dot * x_dot_dot)
        * (
            -(g + y_dot_dot) * x_4dot
            - x_3dot * y_3dot
            + x_dot_dot * y_4dot
            + y_3dot * x_3dot
        )
        / ((g + y_dot_dot) ** 2 + x_dot_dot**2) ** 2
        + (
            -(g + y_dot_dot) * x_5dot
            - x_4dot * y_3dot
            + x_dot_dot * y_5dot
            + y_4dot * x_3dot
        )
        / ((g + y_dot_dot) ** 2 + x_dot_dot**2)
    )
    return gamma_3dot
