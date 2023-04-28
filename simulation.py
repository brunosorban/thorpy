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

# rocket variables
m = 150  # mass of the hopper
h = 2  # height of the hopper
radius = 0.25  # radius of the hopper
C = (
    1 / 12 * m * (h**2 + 3 * radius**2)
)  # moment of inertia of the hopper perpendicular to the main axis

# solution parameters
t0 = 0  # initial time
tf = 15  # final time
initial_state = [0, 0, 0, 0, np.deg2rad(90), 0]
target = [30, 0, 30, 0, np.deg2rad(90), 0]
max_step = 1e-4

# controller parameters
T = 1  # time horizon
N = 40  # Number of control intervals

u_max_f = 10
u_min_f = -10
u_max_tau = 10
u_min_tau = -10

q1 = 5
q2 = 5
q3 = 10
q4 = 10
q5 = 15
q6 = 3
Qf_gain = 10
r = 1e-4

# initial state
t0_val = 0
x0_val = ca.vertcat(*initial_state)
x_target = ca.vertcat(*target)
Q = ca.diag([q1, q2, q3, q4, q5, q6])
Qf = Qf_gain * Q

u_bounds = [(u_min_f, u_max_f), (u_min_tau, u_max_tau)]

controller = MPC_controller(
    m=m,
    C=C,
    g=g,
    T=T,
    N=N,
    u_bounds=u_bounds,
    t0=t0_val,
    x0=x0_val,
    x_target=x_target,
    Q=Q,
    Qf=Qf,
    r=r,
)
print("target =", x_target)

t, x, u, state_horizon_list, control_horizon_list = controller.simulate_inside(tf)
controller.plot_simulation(t, x, u)
animate(
    t,
    x=x[:, 0],
    y=x[:, 2],
    gamma=x[:, 4],
    state_horizon_list=state_horizon_list,
    control_horizon_list=control_horizon_list,
    N=N,
    dt=T / N,
    scale=1,
    matplotlib=False,
    save=False,
)
