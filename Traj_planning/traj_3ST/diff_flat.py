import numpy as np
import matplotlib.pyplot as plt
from Traj_planning.traj_3ST.auxiliar_codes.compute_gamma_3dot import compute_gamma_3dot
from Traj_planning.traj_3ST.auxiliar_codes.pol_processor import *

def diff_flat_traj(
    Px_coeffs, Py_coeffs, Pz_coeffs, t, env_params, rocket_params, controller_params
):
    """This function computes the trajectory parameters for the 2D case using differential flatness.

    Args:
        Px_coeffs (list): List of coefficients for the x position.
        Py_coeffs (list): List of coefficients for the y position.
        Pz_coeffs (List): List of coefficients for the z position.
        t (list): List of time points.
        env_params (dict): Dictionary containing the environment parameters. Currently only g is used.
        rocket_params (dict): Dictionary containing the rocket parameters. Currently only m, l_tvc and J_z are used.
        controller_params (dict): List of controller parameters. Currently only dt is used.

    Returns:
        trajectory_params (dict): Dictionary containing the trajectory parameters.
    """

    print("Starting differential flatness trajectory planning...")
    g = env_params["g"]
    m = rocket_params["m"]
    l_tvc = rocket_params["l_tvc"]
    J_z = rocket_params["J_z"]
    dt = controller_params["dt"]

    t_list = np.arange(0, t[-1], dt)

    x_o = np.zeros_like(t_list)
    y_o = np.zeros_like(t_list)
    z_o = np.zeros_like(t_list)
    vx_o = np.zeros_like(t_list)
    vy_o = np.zeros_like(t_list)
    vz_o = np.zeros_like(t_list)
    ax_o = np.zeros_like(t_list)
    ay_o = np.zeros_like(t_list)
    az_o = np.zeros_like(t_list)
    jx_o = np.zeros_like(t_list)
    jy_o = np.zeros_like(t_list)
    sx_o = np.zeros_like(t_list)
    sy_o = np.zeros_like(t_list)
    cx_o = np.zeros_like(t_list)
    cy_o = np.zeros_like(t_list)
    f1 = np.zeros_like(t_list)
    f1_dot = np.zeros_like(t_list)

    for i in range(len(t_list)):
        idx = np.searchsorted(t, t_list[i])
        if idx > 0:
            idx -= 1

        x_o[i] = pos_processor(Px_coeffs[:, idx], [t[idx], t[idx+1]], t_list[i])
        y_o[i] = pos_processor(Py_coeffs[:, idx], [t[idx], t[idx+1]], t_list[i])
        z_o[i] = pos_processor(Pz_coeffs[:, idx], [t[idx], t[idx+1]], t_list[i])

        vx_o[i] = vel_processor(Px_coeffs[:, idx], [t[idx], t[idx+1]], t_list[i])
        vy_o[i] = vel_processor(Py_coeffs[:, idx], [t[idx], t[idx+1]], t_list[i])
        vz_o[i] = vel_processor(Pz_coeffs[:, idx], [t[idx], t[idx+1]], t_list[i])

        ax_o[i] = acc_processor(Px_coeffs[:, idx], [t[idx], t[idx+1]], t_list[i])
        ay_o[i] = acc_processor(Py_coeffs[:, idx], [t[idx], t[idx+1]], t_list[i])
        az_o[i] = acc_processor(Pz_coeffs[:, idx], [t[idx], t[idx+1]], t_list[i])

        jx_o[i] = jerk_processor(Px_coeffs[:, idx], [t[idx], t[idx+1]], t_list[i])
        jy_o[i] = jerk_processor(Py_coeffs[:, idx], [t[idx], t[idx+1]], t_list[i])

        sx_o[i] = snap_processor(Px_coeffs[:, idx], [t[idx], t[idx+1]], t_list[i])
        sy_o[i] = snap_processor(Py_coeffs[:, idx], [t[idx], t[idx+1]], t_list[i])

        cx_o[i] = crackle_processor(Px_coeffs[:, idx], [t[idx], t[idx+1]], t_list[i])
        cy_o[i] = crackle_processor(Py_coeffs[:, idx], [t[idx], t[idx+1]], t_list[i])

    temp_states = {
        "x_o": x_o,
        "y_o": y_o,
        "z_o": z_o,
        "vx_o": vx_o,
        "vy_o": vy_o,
        "vz_o": vz_o,
        "ax_o": ax_o,
        "ay_o": ay_o,
        "az_o": az_o,
        "jx_o": jx_o,
        "jy_o": jy_o,
        "sx_o": sx_o,
        "sy_o": sy_o,
        "cx_o": cx_o,
        "cy_o": cy_o,
        "g": g,
    }

    gamma = np.arctan2(ay_o + g, ax_o)
    gamma_dot = (jy_o * ax_o - (ay_o + g) * jx_o) / (ax_o**2 + (ay_o + g) ** 2)
    gamma_dot_dot = (sy_o * ax_o - (ay_o + g) * sx_o) / (
        ax_o**2 + (ay_o + g) ** 2
    ) - (jy_o * ax_o - (ay_o + g) * jx_o) * (
        2 * ax_o * jx_o + 2 * (ay_o + g) * jy_o
    ) / (
        ax_o**2 + (ay_o + g) ** 2
    ) ** 2
    gamma_3dot = compute_gamma_3dot(temp_states)

    e1bx = np.zeros_like(t_list)
    e1by = np.zeros_like(t_list)
    e2bx = np.zeros_like(t_list)
    e2by = np.zeros_like(t_list)

    # estimate the trajectory parameters for the 3D case
    for i in range(len(t_list)):
        t = np.array([ax_o[i], ay_o[i] + g, 0])

        e1bx[i] = t[0] / np.linalg.norm(t)
        e1by[i] = t[1] / np.linalg.norm(t)
        e2bx[i] = -t[1] / np.linalg.norm(t)
        e2by[i] = t[0] / np.linalg.norm(t)

    x_g = x_o - J_z / (m * l_tvc) * e1bx
    y_g = y_o - J_z / (m * l_tvc) * e1by
    z_g = z_o

    vx_g = vx_o - J_z / (m * l_tvc) * gamma_dot * e2bx
    vy_g = vy_o - J_z / (m * l_tvc) * gamma_dot * e2by
    vz_g = vz_o

    ax_g = (
        ax_o
        - J_z / (m * l_tvc) * gamma_dot_dot * e2bx
        + J_z / (m * l_tvc) * (gamma_dot**2) * e1bx
    )
    ay_g = (
        ay_o
        - J_z / (m * l_tvc) * gamma_dot_dot * e2by
        + J_z / (m * l_tvc) * (gamma_dot**2) * e1by
    )
    az_g = az_o

    r_go = J_z / (m * l_tvc)

    f2 = -(J_z + m * r_go**2) * gamma_dot_dot / (l_tvc + r_go)  # dot(m*t, yb)
    f2_dot = -(J_z + m * r_go**2) * gamma_3dot / (l_tvc + r_go)

    for i in range(len(t_list)):
        t = np.array([ax_o[i], ay_o[i] + g, 0])
        t_dot = np.array([jx_o[i], jy_o[i], 0])

        f1[i] = (
            m * (t[0] * e1bx[i] + t[1] * e1by[i]) + J_z * gamma_dot[i] ** 2 / l_tvc
        )  # dot(m*t, xb)

        # compute the derivatives
        f1_dot[i] = m * (t_dot[0] * e1bx[i] + t_dot[1] * e1by[i]) + gamma_dot[i] * (
            -t[0] * e1by[i] + t[1] * e1bx[i]
        )  # dot(m*t_dot, xb) + dot(t, cross(gamma_dot, xb))

    f = np.sqrt(f1**2 + f2**2)
    f_dot = (f1 * f1_dot + f2 * f2_dot) / f

    delta_tvc = np.arctan2(f2, f1)
    delta_tvc_dot = (f1 * f2_dot - f2 * f1_dot) / (f1**2 + f2**2)

    trajectory_params = {
        "t": t_list,
        "x": x_g,
        "y": y_g,
        "z": z_g,
        "vx": vx_g,
        "vy": vy_g,
        "vz": vz_g,
        "ax": ax_g,
        "ay": ay_g,
        "az": az_g,
        "e1bx": e1bx,
        "e1by": e1by,
        "e2bx": e2bx,
        "e2by": e2by,
        "gamma": gamma,
        "gamma_dot": gamma_dot,
        "gamma_dot_dot": gamma_dot_dot,
        "gamma_3dot": gamma_3dot,
        "f1": f1,
        "f2": f2,
        "f1_dot": f1_dot,
        "f2_dot": f2_dot,
        "delta_tvc": delta_tvc,
        "delta_tvc_dot": delta_tvc_dot,
        "f": f,
        "f_dot": f_dot,
    }

    print("Differential flatness trajectory planning done.")
    return trajectory_params
