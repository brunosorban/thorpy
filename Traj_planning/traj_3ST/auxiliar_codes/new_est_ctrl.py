import numpy as np
from copy import deepcopy
from Traj_planning.traj_3ST.auxiliar_codes.coeffs2derivatives import (
    get_pos,
    get_vel,
    get_acc,
    get_jerk,
    get_snap,
    get_crackle,
)


def estimate_control(
    Px_coeffs, Py_coeffs, Pz_coeffs, t, env_params, rocket_params, controller_params
):
    # estimate the trajectory parameters for the 2D case
    print("Estimating the trajectory parameters...")

    g = env_params["g"]
    m = rocket_params["m"]
    l_tvc = rocket_params["l_tvc"]
    J_z = rocket_params["J_z"]
    # dt = controller_params["dt"]
    dt = 1e-3

    params = deepcopy(rocket_params)
    params["g"] = g

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
    f1_o = np.zeros_like(t_list)
    f2_o = np.zeros_like(t_list)
    f1_dot_o = np.zeros_like(t_list)
    f2_dot_o = np.zeros_like(t_list)

    for i in range(len(t_list)):
        idx = np.searchsorted(t, t_list[i])
        if idx > 0:
            idx -= 1

        x_o[i] = get_pos(Px_coeffs[:, idx], t_list[i])
        y_o[i] = get_pos(Py_coeffs[:, idx], t_list[i])
        z_o[i] = get_pos(Pz_coeffs[:, idx], t_list[i])

        vx_o[i] = get_vel(Px_coeffs[:, idx], t_list[i])
        vy_o[i] = get_vel(Py_coeffs[:, idx], t_list[i])
        vz_o[i] = get_vel(Pz_coeffs[:, idx], t_list[i])

        ax_o[i] = get_acc(Px_coeffs[:, idx], t_list[i])
        ay_o[i] = get_acc(Py_coeffs[:, idx], t_list[i])
        az_o[i] = get_acc(Pz_coeffs[:, idx], t_list[i])

        jx_o[i] = get_jerk(Px_coeffs[:, idx], t_list[i])
        jy_o[i] = get_jerk(Py_coeffs[:, idx], t_list[i])

        sx_o[i] = get_snap(Px_coeffs[:, idx], t_list[i])
        sy_o[i] = get_snap(Py_coeffs[:, idx], t_list[i])

        cx_o[i] = get_crackle(Px_coeffs[:, idx], t_list[i])
        cy_o[i] = get_crackle(Py_coeffs[:, idx], t_list[i])

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

    e1bx = np.zeros_like(t_list)
    e1by = np.zeros_like(t_list)
    e2bx = np.zeros_like(t_list)
    e2by = np.zeros_like(t_list)

    # estimate the trajectory parameters for the 3D case
    for i in range(len(t_list)):
        t = np.array([ax_o, ay_o + g, 0])
        
        e1bx[i] = t[0] / np.linalg.norm(t)
        e1by[i] = t[1] / np.linalg.norm(t)
        e2bx[i] = -t[1] / np.linalg.norm(t)
        e2by[i] = t[0] / np.linalg.norm(t)
        
        f1_o[i] = m * (t[0] * e1bx[i] + t[1] * e1by[i]) # dot(m*t, xb)
        f2_o[i] = m * (t[0] * e2bx[i] + t[1] * e2by[i]) # dot(m*t, yb)

    estimated_states = {
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
        "f1": f1_o,
        "f2": f2_o,
        "f1_dot": f1_dot_o,
        "f2_dot": f2_dot_o,
        "delta_tvc": delta_tvc_o,
        "delta_tvc_dot": delta_tvc_dot_o,
        "f": f_o,
        "f_dot": f_dot_o,
        "e1tx": e1tx,
        "e1ty": e1ty,
        "e2tx": e2tx,
        "e2ty": e2ty,
    }

    # print("Trajectory parameters estimated.")

    return estimated_states
