from Traj_planning.traj_3ST.pol_interpolation import *
from Traj_planning.traj_3ST.diff_flat import *
from Traj_planning.traj_3ST.auxiliar_codes.plot_traj import *

from copy import deepcopy


def trajenerator_3ST(states, constraints, env_params, rocket_params, controller_params):
    """Function that takes the desired states and constraints and returns the trajectory parameters.

    Args:
        states (dict): Dictionary containing the desired states. The path points shall be in x, y and z lists. One may
            also add the desired velocities in vx, vy and vz lists. If the velocities are not provided, they will be
            calculated using the minimum snap trajectory. Nonetheless, the desired initial and final velocities must
            always be provided. It is assumed that the initial and final accelerations, are zero.
        constraints (dict): Dictionary containing the constraints. Currently t_final, max_vel, max_acc are used.
        env_params (dict): Dictionary containing the environment parameters. Currently g is used.
        rocket_params (dict): Dictionary containing the rocket parameters. Currently m, l_tvc and J_z are used.
        controller_params (dict): Dictionary containing the controller parameters. Currently dt is used.

    Returns:
        trajectory_parameters (dict): Dictionary containing the trajectory parameters.
    """
    # interpolate polinomials for minimum snap trajectory
    states_x = deepcopy(states)
    constraints_x = deepcopy(constraints)

    states_x["pos"] = states["x"]
    constraints_x["max_v"] = constraints["max_vx"]
    constraints_x["min_v"] = constraints["min_vx"]
    constraints_x["max_a"] = constraints["max_ax"]
    constraints_x["min_a"] = constraints["min_ax"]
    Px_coeffs, t = pol_interpolation(states_x, constraints_x)

    states_y = deepcopy(states)
    constraints_y = deepcopy(constraints)

    states_y["pos"] = states["y"]
    constraints_y["max_v"] = constraints["max_vy"]
    constraints_y["min_v"] = constraints["min_vy"]
    constraints_y["max_a"] = constraints["max_ay"]
    constraints_y["min_a"] = constraints["min_ay"]
    Py_coeffs, t = pol_interpolation(states_y, constraints_y)

    states_z = deepcopy(states)
    constraints_z = deepcopy(constraints)

    states_z["pos"] = states["z"]
    constraints_z["max_v"] = constraints["max_vz"]
    constraints_z["min_v"] = constraints["min_vz"]
    constraints_z["max_a"] = constraints["max_az"]
    constraints_z["min_a"] = constraints["min_az"]
    Pz_coeffs, t = pol_interpolation(states_z, constraints_z)

    # Px_coeffs, Py_coeffs, Pz_coeffs, t = min_snap_traj(
    #     states, constraints, rocket_params, controller_params
    # )

    # calculate gamma using differential flatness
    trajectory_params = diff_flat_traj(
        Px_coeffs, Py_coeffs, Pz_coeffs, t, env_params, rocket_params, controller_params
    )

    plot_trajectory(states, trajectory_params, "Diff flat trajectory")

    return trajectory_params
