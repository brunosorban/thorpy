import os
from typing import Union

from math_tools.data_handler_class import DataHandler
from trajectory_planning.coupled_pol_interpolation import *
from trajectory_planning.post_processing import *
from trajectory_planning.auxiliar_codes.plot_traj import *
from trajectory_planning.drift_checker_class import DriftChecker


class Trajectory:
    """Class meant to take the desired states and constraints and returns the trajectory parameters.

    Args:
        states (dict): Dictionary containing the desired states. The path points shall be in x, y and z lists, and the
            time points shall be in t list. The lists shall have the same length and the time points shall be equally
            spaced. Currently, the initial and final velocities and accelerations are 0.
        env_params (dict): Dictionary containing the environment parameters. Currently g is used.
        rocket_params (dict): Dictionary containing the rocket parameters. Currently m, l_tvc and J_z are used.
        controller_params (dict): Dictionary containing the controller parameters. Currently dt, thrust and delta_tvc
            constraints are being used.

    Returns:
        trajectory_parameters (dict): Dictionary containing the trajectory parameters.
    """

    def __init__(self, path: str = None):
        if path is not None:
            self.load_trajectory(path)

    def add_new_configuration(
        self, states: Union[dict, str], env_params: dict, rocket_params: dict, controller_params: dict
    ):
        if not isinstance(states, dict):
            raise TypeError("Invalid states type. A dictionary or a string is expected.")

        self.states = states
        self.env_params = env_params
        self.rocket_params = rocket_params
        self.controller_params = controller_params

    def generate_trajectory(self, total_time: float) -> dict:
        self.total_time = total_time
        n_states = len(self.states["x"])
        states_copy = self.states.copy()
        states_copy["t"] = np.linspace(0, total_time, n_states)
        Px_coeffs, Py_coeffs, Pz_coeffs, t = coupled_pol_interpolation(
            states_copy, self.rocket_params, self.controller_params, self.env_params
        )

        # calculate gamma using differential flatness
        trajectory_params = traj_post_processing(
            Px_coeffs, Py_coeffs, Pz_coeffs, t, self.env_params, self.rocket_params, self.controller_params
        )

        # save the states updated witht the time points used in the trajectory
        self.states = states_copy
        trajectory_params["ref_states"] = states_copy

        DriftChecker(self.env_params, self.rocket_params, trajectory_params)

        self.trajectory_params = trajectory_params

        return trajectory_params

    def plot_trajectory(self, title: str = "Trajectory"):
        plot_trajectory(self.states, self.trajectory_params, self.controller_params, title)

    def save_trajectory(self, file_name: str, folder_path: str = None):
        if folder_path is None:  # use current directory
            folder_path = os.getcwd()
        save_dict = {
            "total_time": self.total_time,
            "states": self.states,
            "env_params": self.env_params,
            "rocket_params": self.rocket_params,
            "controller_params": self.controller_params,
            "trajectory_params": self.trajectory_params,
        }
        DataHandler.save(folder_path, file_name, save_dict)

    def load_trajectory(self, path: str):
        load_dict = DataHandler.load(path)
        self.total_time = load_dict["total_time"]
        self.states = load_dict["states"]
        self.env_params = load_dict["env_params"]
        self.rocket_params = load_dict["rocket_params"]
        self.controller_params = load_dict["controller_params"]
        self.trajectory_params = load_dict["trajectory_params"]
