import sys
import numpy as np

sys.path.append("../")
sys.path.append("../Traj_planning")

# import Function as ft
from flight_class import *
import casadi as ca
from MPC_SE3_controller import MPC_controller
from animate import *
from simulation_parameters import *


########################## Trajectory generation ###############################

##### need to adapt to e1b, e2b beginning ##################
# from Traj_planning.hop import hopper_traj
# trajectory_params = hopper_traj()
# tf = 120 + 5 # 5 seconds after the landing

# from Traj_planning.hop_min_snap import hop_min_snap
# trajectory_params = hop_min_snap()
# tf = 120 + 5 # 5 seconds after the landing

# from Traj_planning.hop_ms_reduced import hop_ms_reduced
# trajectory_params = hop_ms_reduced()
# tf = 12 + 1  # 5 seconds after the landing

# from Traj_planning.backflip.backflip import traj_backflip
# trajectory_params = traj_backflip()
# tf = 152 + 5 # 5 seconds after the landing
##### need to adapt to e1b, e2b finish ##################

# from Traj_planning.backflip.circle import traj_circle
# trajectory_params = traj_circle()
# tf = 50  # 0 seconds after the landing

# from Traj_planning.backflip.simple_circ import traj_simple_circle
# trajectory_params = traj_simple_circle()
# tf = trajectory_params["t"][-1] + 2 # 0 seconds after the landing

# from Traj_planning.diff_flat.backflip_df import traj_points_df_circ
# trajectory_params = traj_points_df_circ(plot=True)
# tf = trajectory_params["t"][-1] + 0.1  # 0.1 second after the landing

from Traj_planning.traj_3ST.circle_3ST import traj_circle_3ST
trajectory_params = traj_circle_3ST()
tf = trajectory_params["t"][-1] + 0.1  # 0.1 second after the landing

######################### Creating the controller ##############################
controller = MPC_controller(
    env_params=env_params,
    rocket_params=rocket_params,
    controller_params=controller_params,
    trajectory_params=trajectory_params,
)


################################## Simulation ##################################
t, x, u, state_horizon_list, control_horizon_list = controller.simulate_inside(
    tf, plot_online=False
)


################################## Plotting ####################################
controller.plot_simulation(t, x, u)
controller.plot_tracking_results(t, x)


################################## Animation ###################################
animate(
    t,
    x=x[:, 0],
    y=x[:, 2],
    gamma=np.arctan2(x[:, 5], x[:, 4]),
    state_horizon_list=state_horizon_list,
    control_horizon_list=control_horizon_list,
    N=N,
    dt=T / N,
    target_goal=None,
    trajectory_params=trajectory_params,
    scale=1,
    matplotlib=False,
    save=False,
)
