import sys

sys.path.append("../")
sys.path.append("../Traj_planning")

from MPC import MPC_controller
from parameters import *
from Animation.animate_traj import animate_traj
from Traj_planning.examples.go_up_3ST import traj_go_up_3ST
from Traj_planning.examples.hopper_3ST import traj_hopper_3ST
from Traj_planning.examples.M_3ST import traj_M_3ST
from Traj_planning.examples.circle_3ST import traj_circle_3ST
from Traj_planning.examples.spiral_3ST import traj_spiral_3ST
from Traj_planning.examples.infinity_symbol_3ST import traj_infinity_symbol_3ST
from Traj_planning.trajectory_class import DataHandler

########################## Trajectory generation ###############################

# trajectory_params = traj_go_up_3ST()
# tf = trajectory_params["t"][-1]  # 0.1 second after the landing

# trajectory_params = traj_hopper_3ST()
# tf = trajectory_params["t"][-1]  # 0.1 second after the landing

# trajectory_params = traj_M_3ST()
# tf = trajectory_params["t"][-1]  # 0.1 second after the landing

# trajectory_params = traj_circle_3ST() # currently not working
# tf = trajectory_params["t"][-1]  # 0.1 second after the landing

# trajectory_params = traj_spiral_3ST()
# tf = trajectory_params["t"][-1]

trajectory_params = traj_infinity_symbol_3ST()
tf = trajectory_params["t"][-1]

########################## Saving the trajectory ###############################
# folder_path = r"D:\Code\SA2023\rocket-control-framework\Traj_planning\examples\data"
# file_name = "go_up_3ST"
# file_name = "hopper_3ST"
# file_name = "infinity_symbol_3ST"
# file_name = "M_3ST"
# DataHandler.save(folder_path, file_name + ".json", trajectory_params)

######################### Creating the controller ##############################
controller = MPC_controller(
    env_params=env_params,
    rocket_params=rocket_params,
    controller_params=controller_params,
    trajectory_params=trajectory_params,
)


################################## Simulation ##################################
t, x, u, state_horizon_list, control_horizon_list = controller.simulate_inside(tf, plot_online=False)

########################## Saving the simulation ###############################
# sol_dict = {
#     "t": t,
#     "x": x,
#     "u": u,
#     "state_horizon_list": state_horizon_list,
#     "control_horizon_list": control_horizon_list,
# }

# DataHandler.save(folder_path, "sol-" + file_name + ".json", sol_dict)

################################## Plotting ####################################
controller.plot_simulation(t, x, u)

################################## Animation ###################################
# animate_traj(
#     t,
#     x[:, 0],  # x
#     x[:, 2],  # y
#     x[:, 4],  # z
#     x[:, 6],  # e1bx
#     x[:, 7],  # e1by
#     x[:, 8],  # e1bz
#     x[:, 9],  # e2bx
#     x[:, 10],  # e2by
#     x[:, 11],  # e2bz
#     x[:, 12],  # e3bx
#     x[:, 13],  # e3by
#     x[:, 14],  # e3bz
#     trajectory_params,
#     save=False,
# )
