def get_gamma_2dot(x_dot_dot, y_dot_dot, x_3dot, y_3dot, x_4dot, y_4dot, params):
    g = params["g"]
    return (y_4dot * x_dot_dot - x_4dot * (y_dot_dot + g)) / (
        x_dot_dot**2 + (y_dot_dot + g) ** 2
    ) - (y_3dot * x_dot_dot - (y_dot_dot + g) * x_3dot) * (
        2 * x_dot_dot * x_3dot + 2 * (y_dot_dot + g) * y_3dot
    ) / (
        x_dot_dot**2 + (y_dot_dot + g) ** 2
    ) ** 2


def get_gamma_3dot(
    x_dot_dot, y_dot_dot, x_3dot, y_3dot, x_4dot, y_4dot, x_5dot, y_5dot, params
):
    g = params["g"]
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
