import numpy as np


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


def estimate_control(env_params, trajectory_params, rocket_params):
    # estimate the trajectory parameters for the 2D case
    print("Estimating the trajectory parameters...")

    # extract the trajectory parameters
    t = trajectory_params["t"]
    x = trajectory_params["x"]
    y = trajectory_params["y"]
    z = trajectory_params["z"]
    vx = trajectory_params["vx"]
    vy = trajectory_params["vy"]
    vz = trajectory_params["vz"]
    ax = trajectory_params["ax"]
    ay = trajectory_params["ay"]
    az = trajectory_params["az"]
    gamma = trajectory_params["gamma"]
    gamma_dot = trajectory_params["gamma_dot"]
    gamma_dot_dot = trajectory_params["gamma_dot_dot"]
    e1bx = trajectory_params["e1bx"]
    e1by = trajectory_params["e1by"]
    e2bx = trajectory_params["e2bx"]
    e2by = trajectory_params["e2by"]

    m = rocket_params["m"]
    J_z = rocket_params["J_z"]
    l_tvc = rocket_params["l_tvc"]

    g = env_params["g"]

    thrust = m * np.sqrt(ax**2 + (ay + g) ** 2)
    delta_tvc = np.arcsin(J_z * gamma_dot_dot / (thrust * l_tvc))

    thrust_dot = get_derivative(t, thrust)
    delta_tvc_dot = get_derivative(t, delta_tvc)

    delta_tvc_w = gamma + delta_tvc

    e1tx = np.cos(delta_tvc_w)
    e1ty = np.sin(delta_tvc_w)
    e2tx = -np.sin(delta_tvc_w)
    e2ty = np.cos(delta_tvc_w)

    estimated_states = {
        "t": t,
        "x": x,
        "y": y,
        "z": z,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "ax": ax,
        "ay": ay,
        "az": az,
        "gamma": gamma,
        "gamma_dot": gamma_dot,
        "gamma_dot_dot": gamma_dot_dot,
        "e1bx": e1bx,
        "e1by": e1by,
        "e2bx": e2bx,
        "e2by": e2by,
        "thrust": thrust,
        "delta_tvc": delta_tvc,
        "thrust_dot": thrust_dot,
        "delta_tvc_dot": delta_tvc_dot,
        "e1tx": e1tx,
        "e1ty": e1ty,
        "e2tx": e2tx,
        "e2ty": e2ty,
    }

    print("Trajectory parameters estimated.")

    return estimated_states
