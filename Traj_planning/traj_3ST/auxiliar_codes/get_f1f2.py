import casadi as ca
from Traj_planning.traj_3ST.auxiliar_codes.get_gamma import (
    get_gamma_2dot,
    get_gamma_3dot,
)
from Traj_planning.traj_3ST.auxiliar_codes.coeffs2derivatives import (
    get_acc,
    get_jerk,
)


def get_f1f2(t, coeffs_x, coeffs_y, params):
    g = params["g"]
    x_dot_dot = get_acc(coeffs_x, t)
    y_dot_dot = get_acc(coeffs_y, t)
    gamma_dot_dot = get_gamma_2dot(t, coeffs_x, coeffs_y, g)

    m = params["m"]
    J_z = params["J_z"]
    l_tvc = params["l_tvc"]

    f1 = ca.sqrt(
        (m * x_dot_dot) ** 2
        + (m * (y_dot_dot + g)) ** 2
        - (-J_z * gamma_dot_dot / l_tvc) ** 2
    )

    f2 = -J_z * gamma_dot_dot / l_tvc

    return f1, f2


def get_f1f2_dot(t, coeffs_x, coeffs_y, params):
    g = params["g"]
    x_dot_dot = get_acc(coeffs_x, t)
    y_dot_dot = get_acc(coeffs_y, t)
    x_3dot = get_jerk(coeffs_x, t)
    y_3dot = get_jerk(coeffs_y, t)
    gamma_dot_dot = get_gamma_2dot(t, coeffs_x, coeffs_y, g)
    gamma_3dot = get_gamma_3dot(t, coeffs_x, coeffs_y, g)

    m = params["m"]
    J_z = params["J_z"]
    l_tvc = params["l_tvc"]

    f1_dot = (
        -(J_z**2) * gamma_dot_dot * gamma_3dot / l_tvc**2
        + m**2 * (g + y_dot_dot) * y_3dot
        + m**2 * x_dot_dot * x_3dot
    ) / ca.sqrt(
        -(J_z**2) * gamma_dot_dot**2 / l_tvc**2
        + m**2 * (g + y_dot_dot) ** 2
        + m**2 * x_dot_dot**2
    )

    f2_dot = -J_z * gamma_3dot / l_tvc

    return f1_dot, f2_dot
