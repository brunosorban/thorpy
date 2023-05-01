import sys
import numpy as np

sys.path.append("../")

import Function as ft
from flight_class import *
import casadi as ca
from MPC_controller import MPC_controller
from animate import *

# global variables
# universal constants
g = 9.81  # gravity

env_params = {
    "g": g,
}

# rocket variables
m = 150  # mass of the hopper
h = 2  # height of the hopper
radius = 0.25  # radius of the hopper
C = (
    1 / 12 * m * (h**2 + 3 * radius**2)
)  # moment of inertia of the hopper perpendicular to the main axis
x_tvc = 0.5  # distance from the center of mass to the TVC

# TVC parameters
K_tvc = 1  # TVC gain
T_tvc = 0.1  # TVC time constant

rocket_params = {
    "m": m,
    "h": h,
    "radius": radius,
    "C": C,
    "K_tvc": K_tvc,
    "T_tvc": T_tvc,
    "x_tvc": x_tvc,
}

# solution parameters
t0 = 0  # initial time
tf = 15  # final time
# state space: [x, x_dot, y, y_dot, gamma, gamma_dot, delta_tvc]
initial_state = [0, 0, 0, 0, np.deg2rad(90), 0, 0]
target = [30, 0, 30, 0, np.deg2rad(90), 0, 0]
max_step = 1e-4

# sol_params = {
#     "t0": t0,
#     "tf": tf,
#     "initial_state": initial_state,
#     "target": target,
#     "max_step": max_step,
# }

# controller parameters
T = 1  # time horizon
N = 40  # Number of control intervals

u_max_f = 10000
u_min_f = -10000
u_max_delta_tvc_c = np.deg2rad(12)
u_min_delta_tvc_c = -np.deg2rad(12)
gamma_max = np.deg2rad(15)
gamma_min = np.deg2rad(-15)

q1 = 5
q2 = 5
q3 = 10
q4 = 10
q5 = 15
q6 = 3
q7 = 100
Qf_gain = 10

r1 = 1e-4
r2 = 100

# initial state
t0_val = 0
x0_val = ca.vertcat(*initial_state)
x_target = ca.vertcat(*target)
Q = ca.diag([q1, q2, q3, q4, q5, q6, q7])
Qf = Qf_gain * Q
R = ca.diag([r1, r2])

u_bounds = [(u_min_f, u_max_f), (u_min_delta_tvc_c, u_max_delta_tvc_c)]

controller_params = {
    "T": T,
    "N": N,
    "dt": T / N,
    "u_bounds": u_bounds,
    "t0": t0_val,
    "x0": x0_val,
    "x_target": x_target,
    "Q": Q,
    "Qf": Qf,
    "R": R,
    "gamma_bounds": (gamma_min, gamma_max),
}


controller = MPC_controller(
    env_params=env_params,
    rocket_params=rocket_params,
    controller_params=controller_params,
)
print("target =", x_target)

t, x, u, state_horizon_list, control_horizon_list = controller.simulate_inside(tf)
controller.plot_simulation(t, x, u)
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
    scale=1,
    matplotlib=False,
    save=False,
)
