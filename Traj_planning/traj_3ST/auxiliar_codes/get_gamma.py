from Traj_planning.traj_3ST.auxiliar_codes.coeffs2derivatives import (
    get_acc,
    get_jerk,
    get_snap,
    get_crackle,
)


def get_gamma_2dot(t, coeffs_x, coeffs_y, g):
    x_dot_dot = get_acc(coeffs_x, t)
    y_dot_dot = get_acc(coeffs_y, t)
    x_3dot = get_jerk(coeffs_x, t)
    y_3dot = get_jerk(coeffs_y, t)
    x_4dot = get_snap(coeffs_x, t)
    y_4dot = get_snap(coeffs_y, t)

    return (y_4dot * x_dot_dot - x_4dot * (y_dot_dot + g)) / (
        x_dot_dot**2 + (y_dot_dot + g) ** 2
    ) - (y_3dot * x_dot_dot - (y_dot_dot + g) * x_3dot) * (
        2 * x_dot_dot * x_3dot + 2 * (y_dot_dot + g) * y_3dot
    ) / (
        x_dot_dot**2 + (y_dot_dot + g) ** 2
    ) ** 2


def get_gamma_3dot(t, coeffs_x, coeffs_y, g):
    x_dot_dot = get_acc(coeffs_x, t)
    y_dot_dot = get_acc(coeffs_y, t)
    x_3dot = get_jerk(coeffs_x, t)
    y_3dot = get_jerk(coeffs_y, t)
    x_4dot = get_snap(coeffs_x, t)
    y_4dot = get_snap(coeffs_y, t)
    x_5dot = get_crackle(coeffs_x, t)
    y_5dot = get_crackle(coeffs_y, t)
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
