import sys
import numpy as np

sys.path.append("../")
sys.path.append("../Traj_planning")

# import Function as ft
# from flight_class import *
import casadi as ca
from MPC_SE3_controller import MPC_controller
from animate import *
from parameters import *


########################## Trajectory generation ###############################

# from Traj_planning.examples.go_up_3ST import traj_go_up_3ST
# trajectory_params = traj_go_up_3ST()
# tf = trajectory_params["t"][-1] + 0.1  # 0.1 second after the landing

from Traj_planning.examples.hopper_3ST import traj_hopper_3ST
trajectory_params = traj_hopper_3ST()
tf = trajectory_params["t"][-1] + 0.1  # 0.1 second after the landing

# from Traj_planning.examples.M_3ST import traj_M_3ST
# trajectory_params = traj_M_3ST()
# tf = trajectory_params["t"][-1] + 0.1  # 0.1 second after the landing

# from Traj_planning.examples.circle_3ST import traj_circle_3ST
# trajectory_params = traj_circle_3ST()
# tf = trajectory_params["t"][-1] + 0.1  # 0.1 second after the landing

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
# animate(
#     t,
#     x=x[:, 0],
#     y=x[:, 2],
#     gamma=np.arctan2(x[:, 5], x[:, 4]),
#     state_horizon_list=state_horizon_list,
#     control_horizon_list=control_horizon_list,
#     N=N,
#     dt=T / N,
#     target_goal=None,
#     trajectory_params=trajectory_params,
#     scale=1,
#     matplotlib=False,
#     save=True,
# )
