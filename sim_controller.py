import sys

sys.path.append("../")
sys.path.append("../Traj_planning")

from MPC import MPC_controller
from animate import *
from parameters import *
from Animation.animate_traj import animate_traj
from Traj_planning.examples.go_up_3ST import traj_go_up_3ST
from Traj_planning.examples.hopper_3ST import traj_hopper_3ST
from Traj_planning.examples.M_3ST import traj_M_3ST
from Traj_planning.examples.circle_3ST import traj_circle_3ST
from Traj_planning.examples.spiral_3ST import traj_spiral_3ST
from Traj_planning.examples.infinity_symbol_3ST import traj_infinity_symbol_3ST

########################## Trajectory generation ###############################

# trajectory_params = traj_go_up_3ST()
# tf = trajectory_params["t"][-1] + 0.1  # 0.1 second after the landing

# trajectory_params = traj_hopper_3ST()
# tf = trajectory_params["t"][-1] + 0.1  # 0.1 second after the landing

# trajectory_params = traj_M_3ST()
# tf = trajectory_params["t"][-1] + 0.1  # 0.1 second after the landing

# trajectory_params = traj_circle_3ST()
# tf = trajectory_params["t"][-1] + 0.1  # 0.1 second after the landing

# trajectory_params = traj_spiral_3ST()
# tf = trajectory_params["t"][-1]

trajectory_params = traj_infinity_symbol_3ST()
tf = trajectory_params["t"][-1]

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
# controller.plot_tracking_results(t, x)


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

animate_traj(
    x[:, 0], # x
    x[:, 2], # y
    x[:, 4], # z
    x[:, 6], # e1bx
    x[:, 7], # e1by
    x[:, 8], # e1bz
    x[:, 9], # e2bx
    x[:, 10], # e2by
    x[:, 11], # e2bz
    x[:, 12], # e3bx
    x[:, 13], # e3by
    x[:, 14], # e3bz
)