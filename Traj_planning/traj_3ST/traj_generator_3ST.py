import json
from Traj_planning.traj_3ST.pol_interpolation import *
from Traj_planning.traj_3ST.diff_flat import *
from Traj_planning.traj_3ST.MPC_traj import *
from Traj_planning.traj_3ST.plot_traj import *

def trajenerator_3ST(states, constraints, env_params, rocket_params, controller_params):
    # interpolate polinomials for minimum snap trajectory
    Px_coeffs, Py_coeffs, Pz_coeffs, t = min_snap_traj(states, constraints)
        
    # calculate gamma using differential flatness
    trajectory_params = diff_flat_traj(Px_coeffs, Py_coeffs, Pz_coeffs, t, env_params, rocket_params, controller_params)
    
    plot_trajectory(trajectory_params, "Diff flat trajectory")
    
    # # optimize again using MPC
    # new_trajectory_params = MPC_traj(env_params, rocket_params, controller_params, trajectory_params,)
    
    # plot_trajectory(new_trajectory_params, "3S trajectory")
    
    # # Save the dictionary as a JSON file
    # with open("new_trajectory_params", "w") as json_file:
    #     json.dump(new_trajectory_params, json_file)
        
    # return new_trajectory_params
    return trajectory_params