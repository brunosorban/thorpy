import sys
import numpy as np

sys.path.append("../")
sys.path.append("../Traj_planning")

# import Function as ft
from flight_class import *
import casadi as ca
from MPC_controller import MPC_controller
from animate import *
from simulation_parameters import *


########################## Trajectory generation ###############################

from Traj_planning.hop import hopper_traj
trajectory_params = hopper_traj()
tf = 120 + 5 # 5 seconds after the landing

# from Traj_planning.backflip import traj_backflip
# trajectory_params = traj_backflip()
# tf = 152 + 5 # 5 seconds after the landing



######################### Creating the controller ##############################
controller = MPC_controller(
    env_params=env_params,
    rocket_params=rocket_params,
    controller_params=controller_params,
    normalization_params_x=normalization_params_x,
    normalization_params_u=normalization_params_u,
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
    gamma=x[:, 4],
    delta_tvc=x[:, 6],
    state_horizon_list=state_horizon_list,
    control_horizon_list=control_horizon_list, 
    N=N,
    dt=T / N,
    target_goal=[target[0], target[2]],
    trajectory_params=trajectory_params,
    scale=1,
    matplotlib=False,
    save=False,
)
