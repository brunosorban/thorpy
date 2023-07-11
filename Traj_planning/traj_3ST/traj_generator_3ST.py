from Traj_planning.traj_3ST.coupled_pol_interpolation import *
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

    Px_coeffs, Py_coeffs, Pz_coeffs, t = coupled_pol_interpolation(
        states, constraints, rocket_params, controller_params
    )

    # calculate gamma using differential flatness
    trajectory_params = diff_flat_traj(
        Px_coeffs, Py_coeffs, Pz_coeffs, t, env_params, rocket_params, controller_params
    )

    plot_trajectory(states, trajectory_params, "Diff flat trajectory")

    return trajectory_params
