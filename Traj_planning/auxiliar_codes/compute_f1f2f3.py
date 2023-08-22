import casadi as ca
from Traj_planning.auxiliar_codes.compute_omega import *
from Traj_planning.auxiliar_codes.compute_omega_dot import *


def compute_f1f2f3(
    x_dot_dot,
    y_dot_dot,
    z_dot_dot,
    x_3dot,
    y_3dot,
    z_3dot,
    x_4dot,
    y_4dot,
    z_4dot,
    params,
):
    m = params["m"]
    J_2 = params["J_2"]
    J_3 = params["J_3"]
    l_tvc = params["l_tvc"]
    g = params["g"]
    omega = compute_omega(x_dot_dot, y_dot_dot, z_dot_dot, x_3dot, y_3dot, z_3dot, g)
    omega_dot = compute_omega_dot(
        x_dot_dot,
        y_dot_dot,
        z_dot_dot,
        x_3dot,
        y_3dot,
        z_3dot,
        x_4dot,
        y_4dot,
        z_4dot,
        g,
    )

    omega_2 = omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]
    omega_dot_2 = (
        omega_dot[0] * omega_dot[0]
        + omega_dot[1] * omega_dot[1]
        + omega_dot[2] * omega_dot[2]
    )

    f1 = (
        m * ca.sqrt((x_dot_dot**2 + y_dot_dot**2 + (z_dot_dot + g) ** 2))
        + (J_2 + J_3) / (2 * l_tvc) * omega_2
    )
    f23 = ((J_2 + J_3) / 2) * ca.sqrt(omega_dot_2) / l_tvc

    return f1, f23
