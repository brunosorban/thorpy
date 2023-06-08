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
    J = rocket_params["J"]
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

    gamma = np.arctan2(ay_o + g, ax_o)
    gamma_dot_o = get_derivative(t_list, gamma)
    gamma_dot_dot_o = get_derivative(t_list, gamma_dot_o)

    e1bx = np.cos(gamma)
    e1by = np.sin(gamma)
    e2bx = -np.sin(gamma)
    e2by = np.cos(gamma)

    x_g = x_o - J / (m * l_tvc) * e1bx
    y_g = y_o - J / (m * l_tvc) * e1by
    z_g = z_o

    vx_g = vx_o - J / (m * l_tvc) * gamma_dot_o * e2bx
    vy_g = vy_o - J / (m * l_tvc) * gamma_dot_o * e2by
    vz_g = vz_o

    ax_g = (
        ax_o
        - J / (m * l_tvc) * gamma_dot_dot_o * e2bx
        + J / (m * l_tvc) * (gamma_dot_o**2) * e1bx
    )
    ay_g = (
        ay_o
        - J / (m * l_tvc) * gamma_dot_dot_o * e2by
        + J / (m * l_tvc) * (gamma_dot_o**2) * e1by
    )
    az_g = az_o

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
        "gamma_dot": gamma_dot_o,
        "gamma_dot_dot": gamma_dot_dot_o,
    }

    print("Differential flatness trajectory planning done.")
    return trajectory_params
