import numpy as np
from Traj_planning.traj_3ST.get_gamma_3dot import get_gamma_3dot


def get_pos(coefs, t):
    return (
        coefs[0] * t ** 7
        + coefs[1] * t ** 6
        + coefs[2] * t ** 5
        + coefs[3] * t ** 4
        + coefs[4] * t ** 3
        + coefs[5] * t ** 2
        + coefs[6] * t
        + coefs[7]
    )
    
def get_vel(coefs, t):
    return (
        7 * coefs[0] * t ** 6
        + 6 * coefs[1] * t ** 5
        + 5 * coefs[2] * t ** 4
        + 4 * coefs[3] * t ** 3
        + 3 * coefs[4] * t ** 2
        + 2 * coefs[5] * t
        + coefs[6]
    )
    
def get_acc(coefs, t):
    return (
        42 * coefs[0] * t ** 5
        + 30 * coefs[1] * t ** 4
        + 20 * coefs[2] * t ** 3
        + 12 * coefs[3] * t ** 2
        + 6 * coefs[4] * t
        + 2 * coefs[5]
    )
    
def get_jerk(coefs, t):
    return (
        210 * coefs[0] * t ** 4
        + 120 * coefs[1] * t ** 3
        + 60 * coefs[2] * t ** 2
        + 24 * coefs[3] * t
        + 6 * coefs[4]
    )
    
def get_snap(coefs, t):
    return (
        840 * coefs[0] * t ** 3
        + 360 * coefs[1] * t ** 2
        + 120 * coefs[2] * t
        + 24 * coefs[3]
    )
    
def get_crackle(coefs, t):
    return (
        2520 * coefs[0] * t ** 2
        + 720 * coefs[1] * t
        + 120 * coefs[2]
    )
    
def get_pop(coefs, t):
    return (
        5040 * coefs[0] * t
        + 720 * coefs[1]
    )

# def get_pos(coefs, t):
#     return (
#         coefs[0] * t ** 8
#         + coefs[1] * t ** 7
#         + coefs[2] * t ** 6
#         + coefs[3] * t ** 5
#         + coefs[4] * t ** 4
#         + coefs[5] * t ** 3
#         + coefs[6] * t ** 2
#         + coefs[7] * t
#         + coefs[8]
#     )
    
# def get_vel(coefs, t):
#     return (
#         56 * coefs[0] * t ** 7
#         + 42 * coefs[1] * t ** 6
#         + 30 * coefs[2] * t ** 5
#         + 20 * coefs[3] * t ** 4
#         + 12 * coefs[4] * t ** 3
#         + 6 * coefs[5] * t ** 2
#         + 2 * coefs[6] * t
#         + coefs[7]
#     )
    
# def get_acc(coefs, t):
#     return (
#         336 * coefs[0] * t ** 6
#         + 252 * coefs[1] * t ** 5
#         + 150 * coefs[2] * t ** 4
#         + 80 * coefs[3] * t ** 3
#         + 36 * coefs[4] * t ** 2
#         + 12 * coefs[5] * t
#         + 2 * coefs[6]
#     )
    
# def get_jerk(coefs, t):
#     return (
#         1680 * coefs[0] * t ** 5
#         + 1260 * coefs[1] * t ** 4
#         + 600 * coefs[2] * t ** 3
#         + 240 * coefs[3] * t ** 2
#         + 72 * coefs[4] * t
#         + 12 * coefs[5]
#     )
    
# def get_snap(coefs, t):
#     return (
#         6720 * coefs[0] * t ** 4
#         + 5040 * coefs[1] * t ** 3
#         + 1800 * coefs[2] * t ** 2
#         + 480 * coefs[3] * t
#         + 72 * coefs[4]
#     )
    
# def get_crackle(coefs, t):
#     return (
#         26880 * coefs[0] * t ** 3
#         + 15120 * coefs[1] * t ** 2
#         + 3600 * coefs[2] * t
#         + 480 * coefs[3]
#     )
    
# def get_pop(coefs, t):
#     return (
#         80640 * coefs[0] * t ** 2
#         + 30240 * coefs[1] * t
#         + 3600 * coefs[2]
#     )


def get_derivative(t, f):
    """Calculate the derivative of the array f."""
    df = np.zeros_like(f)

    h = t[1] - t[0]
    df[0] = (-f[2] + 4 * f[1] - 3 * f[0]) / (2 * h)

    # Treat right border
    h = t[-1] - t[-2]
    df[-1] = (3 * f[-1] - 4 * f[-2] + f[-3]) / (2 * h)

    # center of the array
    for i in range(1, len(f) - 1):
        df[i] = (f[i + 1] - f[i - 1]) / (2 * h)

    return df


def diff_flat_traj(
    Px_coeffs, Py_coeffs, Pz_coeffs, t, env_params, rocket_params, controller_params
):
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

    gamma = np.arctan2(ay_o + g, ax_o)
    gamma_dot = (jy_o * ax_o - (ay_o + g) * jx_o) / (ax_o**2 + (ay_o + g) ** 2)
    gamma_dot_dot = (sy_o * ax_o - (ay_o + g) * sx_o) / (
        ax_o**2 + (ay_o + g) ** 2
    ) - (jy_o * ax_o - (ay_o + g) * jx_o) * (
        2 * ax_o * jx_o + 2 * (ay_o + g) * jy_o
    ) / (
        ax_o**2 + (ay_o + g) ** 2
    ) ** 2
    gamma_3dot = get_gamma_3dot(temp_states)

    e1bx = np.cos(gamma)
    e1by = np.sin(gamma)
    e2bx = -np.sin(gamma)
    e2by = np.cos(gamma)

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

    # compute the control inputs of the system
    f1 = np.sqrt(
        (m * ax_o) ** 2 + (m * (ay_o + g)) ** 2 + -((-J_z * gamma_dot_dot / l_tvc) ** 2)
    )
    f2 = -J_z * gamma_dot_dot / l_tvc

    # compute the control inputs dervatives of the system
    f1_dot = (
        m**2 * (ax_o * jx_o + (ay_o + g) * jy_o)
        - J_z**2 * gamma_dot_dot * gamma_3dot / l_tvc**2
    ) / f1
    f2_dot = -J_z * gamma_3dot / l_tvc

    delta_tvc = np.arctan2(f2, f1)
    delta_tvc_dot = (f1 * f2_dot - f2 * f1_dot) / (f1**2 + f2**2)

    f = np.sqrt(f1**2 + f2**2)
    f_dot = (f1 * f1_dot + f2 * f2_dot) / f

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
