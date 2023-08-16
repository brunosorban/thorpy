import numpy as np
from Traj_planning.auxiliar_codes.get_gamma import *
from Traj_planning.auxiliar_codes.compute_gamma_3dot import compute_gamma_3dot
from Traj_planning.auxiliar_codes.pol_processor import *
from Traj_planning.auxiliar_codes.get_f1f2 import *


def diff_flat_traj(
    Px_coeffs, Py_coeffs, Pz_coeffs, t, env_params, rocket_params, controller_params
):
    """This function computes the trajectory parameters for the 2D case using differential flatness.
        One shall mind that the polinomials are for the trajectory of the oscillation point, but the
        returned variables are converted for the center of gravity of the rocket.

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
            The variables were computed for the center of gravity of the rocket,
            but the polinomials are for the oscillation point.
    """

    print("Starting differential flatness trajectory planning...")
    g = env_params["g"]
    m = rocket_params["m"]
    l_tvc = rocket_params["l_tvc"]
    J_z = rocket_params["J_z"]
    dt = controller_params["dt"]

    params = {}
    params["g"] = g
    params["m"] = m
    params["l_tvc"] = l_tvc
    params["J_z"] = J_z

    t_list = np.linspace(t[0], t[-1], int((t[-1] - t[0]) / dt) + 1, endpoint=True)

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

    for i in range(len(t) - 1):
        t0 = t[i]
        tf = t[i + 1]
        i0 = np.where(t_list == t0)[0][0]
        i1 = np.where(t_list == tf)[0][0] + 1

        x_o[i0:i1] = pos_processor(Px_coeffs[:, i], [t[i], t[i + 1]], t_list[i0:i1])
        y_o[i0:i1] = pos_processor(Py_coeffs[:, i], [t[i], t[i + 1]], t_list[i0:i1])
        z_o[i0:i1] = pos_processor(Pz_coeffs[:, i], [t[i], t[i + 1]], t_list[i0:i1])

        vx_o[i0:i1] = vel_processor(Px_coeffs[:, i], [t[i], t[i + 1]], t_list[i0:i1])
        vy_o[i0:i1] = vel_processor(Py_coeffs[:, i], [t[i], t[i + 1]], t_list[i0:i1])
        vz_o[i0:i1] = vel_processor(Pz_coeffs[:, i], [t[i], t[i + 1]], t_list[i0:i1])

        ax_o[i0:i1] = acc_processor(Px_coeffs[:, i], [t[i], t[i + 1]], t_list[i0:i1])
        ay_o[i0:i1] = acc_processor(Py_coeffs[:, i], [t[i], t[i + 1]], t_list[i0:i1])
        az_o[i0:i1] = acc_processor(Pz_coeffs[:, i], [t[i], t[i + 1]], t_list[i0:i1])

        jx_o[i0:i1] = jerk_processor(Px_coeffs[:, i], [t[i], t[i + 1]], t_list[i0:i1])
        jy_o[i0:i1] = jerk_processor(Py_coeffs[:, i], [t[i], t[i + 1]], t_list[i0:i1])

        sx_o[i0:i1] = snap_processor(Px_coeffs[:, i], [t[i], t[i + 1]], t_list[i0:i1])
        sy_o[i0:i1] = snap_processor(Py_coeffs[:, i], [t[i], t[i + 1]], t_list[i0:i1])

        cx_o[i0:i1] = crackle_processor(
            Px_coeffs[:, i], [t[i], t[i + 1]], t_list[i0:i1]
        )
        cy_o[i0:i1] = crackle_processor(
            Py_coeffs[:, i], [t[i], t[i + 1]], t_list[i0:i1]
        )

    e1bx = np.zeros_like(t_list)
    e1by = np.zeros_like(t_list)
    e2bx = np.zeros_like(t_list)
    e2by = np.zeros_like(t_list)

    # estimate the trajectory parameters for the 3D case
    for i in range(len(t_list)):
        t = np.array([ax_o[i], ay_o[i] + g, 0])
        temp = np.linalg.norm(t)

        e1bx[i] = t[0] / temp
        e1by[i] = t[1] / temp
        e2bx[i] = -t[1] / temp
        e2by[i] = t[0] / temp

    # estimate the gamma, gamma_dot, gamma_dot_dot, gamma_3dot and forces
    gamma = np.arctan2(ay_o + g, ax_o)
    gamma_dot = (jy_o * ax_o - (ay_o + g) * jx_o) / (ax_o**2 + (ay_o + g) ** 2)
    gamma_dot_dot = get_gamma_2dot(ax_o, ay_o, jx_o, jy_o, sx_o, sy_o, params)
    gamma_3dot = get_gamma_3dot(ax_o, ay_o, jx_o, jy_o, sx_o, sy_o, cx_o, cy_o, params)

    f1, f2 = get_f1f2_np(ax_o, ay_o, jx_o, jy_o, sx_o, sy_o, params)
    f1_dot, f2_dot = get_f1f2_dot_np(
        ax_o, ay_o, jx_o, jy_o, sx_o, sy_o, cx_o, cy_o, params
    )

    f = np.sqrt(f1**2 + f2**2)
    f_dot = (f1 * f1_dot + f2 * f2_dot) / f

    delta_tvc = np.arctan2(f2, f1)
    delta_tvc_dot = (f1 * f2_dot - f2 * f1_dot) / (f1**2 + f2**2)

    # compute the states for the CG
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

    trajectory_params = {
        "t": t_list,
        "x": x_o,
        "y": y_o,
        "z": z_o,
        "vx": vx_o,
        "vy": vy_o,
        "vz": vz_o,
        "ax": ax_o,
        "ay": ay_o,
        "az": az_o,
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
