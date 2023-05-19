import numpy as np
from traj_plan_MPC import *

# Example usage #################################################################
dt = 0.05
t0 = 0  # beginning
dt1 = 23  # first jump
dt2 = 50 #30  # second jump
dt3 = 3  # getting into inverted flight
dt4 = 10  # inverted flight upside down
dt5 = 15  # getting out of inverted flight
dt6 = 3  # returning to upside up
dt7 = 50  # slowing down
dt8 = 18  # landing

time_points = [
    t0, # beginning
    t0 + dt1, # first jump
    t0 + dt1 + dt2, # second jump
    t0 + dt1 + dt2 + dt3, # getting into inverted flight
    t0 + dt1 + dt2 + dt3 + dt4, # inverted flight upside down
    t0 + dt1 + dt2 + dt3 + dt4 + dt5, # getting out of inverted flight
    t0 + dt1 + dt2 + dt3 + dt4 + dt5 + dt6, # returning to upside up
    t0 + dt1 + dt2 + dt3 + dt4 + dt5 + dt6 + dt7, # slowing down
    t0 + dt1 + dt2 + dt3 + dt4 + dt5 + dt6 + dt7 + dt8, # landing
]
g = 9.81  # gravity

# controller input bounds
max_acc_y = g  # maxium thrust rate
min_acc_y = (
    -2 * max_acc_y
)  # 0.35 * max_acc_y  - g # should be 30% to 40% of the max thrust
max_acc_x = max_acc_y
min_acc_x = -max_acc_x

points = np.array(
    [
        [0, 0, 0],  # beginning
        [100, 2000, 0],  # first jump
        [850, 9200, 0],  # second jump
        [900, 9500, 0],  # getting into inverted flight
        [1000, 10000, 0],  # inverted flight upside down
        [900, 9000, 0],  # getting out of inverted flight
        [850, 8650, 0],  # returning to upside up
        [500, 500, 0],  # slowing down
        [500, 0, 0],  # landing
    ]
)

gamma_points_deg = np.array(
    [
        90,  # beginning
        90,  # first jump
        90,  # second jump
        270,  # getting into inverted flight
        270,  # inverted flight upside down
        270,  # getting out of inverted flight
        90,  # returning to upside up
        90,  # slowing down
        90,  # landing
    ]
)

velocities = np.array(
    [
        [0, 0, None],  # beginning
        [None, None, None],  # first jump
        [None, 122, None],  # second jump
        [None, None, None],  # getting into inverted flight
        [0, 0, None],  # inverted flight upside down
        [None, None, None],  # getting out of inverted flight
        [None, None, None],  # returning to upside up
        [0, -30, None],  # slowing down
        [0, 0, 0],  # landing
    ]
)

accelerations = np.array(
    [
        [0, 0, None],  # beginning
        [None, None, None],  # first jump
        [None, None, None],  # second jump
        [None, None, None],  # getting into inverted flight
        [0, 0, None],  # inverted flight upside down
        [None, None, None],  # getting out of inverted flight
        [None, None, None],  # returning to upside up
        [0, None, None],  # slowing down
        [0, None, None],  # landing
    ]
)
############################ User defined parameters ###########################
# environment variables
g = 9.81  # gravity

# rocket variables
m = 100  # mass of the hopper
# mf = 50  # final mass of the hopper
h = 2  # height of the hopper
radius = 0.25  # radius of the hopper
l_tvc = 0.5  # distance from the center of mass to the TVC

# TVC parameters
K_tvc = 1  # TVC gain
T_tvc = 0.5  # TVC time constant

# Thrust parameters
K_thrust = 1  # Thrust gain
T_thrust = 1  # Thrust time constant

# state space: [x, x_dot, y, y_dot, gamma, gamma_dot, thrust, delta_tvc]
initial_state = [0, 0, 0, 0, np.deg2rad(90), 0, m * g, 0]

# controller input bounds
u_max_f = 2 * m * g  # maxium thrust rate
u_min_f = 0.35 * u_max_f  # should be 30% to 40% of the max thrust
u_max_delta_tvc_c = np.deg2rad(10)  # maxium thrust vector angle
u_min_delta_tvc_c = -np.deg2rad(10)  # minium thrust vector angle

stab_time_criterion = 3  # corresponds to the system reach 95% of the final value

u_max_f_deriv = u_max_f / (
    stab_time_criterion * T_thrust
)  # u_max_f / T_thrust # maxium thrust rate
u_min_f_deriv = -u_max_f_deriv  # should be 30% to 40% of the max thrust
u_max_delta_tvc_c_deriv = np.deg2rad(10) / (
    stab_time_criterion * T_tvc
)  # maxium thrust vector angle
u_min_delta_tvc_c_deriv = -np.deg2rad(10) / (
    stab_time_criterion * T_tvc
)  # minium thrust vector angle

q1 = 15  # position in x cost penalty
q2 = 20  # velocity in x cost penalty
q3 = 15  # position in y cost penalty
q4 = 20  # velocity in y cost penalty
q5 = 15 * 100  # yaw angle cost penalty
q6 = 3 * 100  # yaw rate cost penalty
q7 = 1e-15  # thrust cost penalty
q8 = 1e-15  # thrust vector angle cost penalty
Qf_gain = 10  # gain of the final cost

r1 = 1e-3  # thrust cost penalty
r2 = 200  # thrust vector angle cost penalty


###################### Calculated and casadi varibles ##########################
C = (
    1 / 12 * m * (h**2 + 3 * radius**2)
)  # moment of inertia of the hopper perpendicular to the main axis

# initial state
t0_val = 0  # initial time
x0_val = ca.vertcat(*initial_state)  # initial state in casadi varible

Q = ca.diag([q1, q2, q3, q4, q5, q6, q7, q8])  # cost matrix
Qf = Qf_gain * Q  # final cost matrix
R = ca.diag([r1, r2])  # control cost matrix

# control bounds
u_bounds = [
    (u_min_f_deriv, u_max_f_deriv),
    (u_min_delta_tvc_c_deriv, u_max_delta_tvc_c_deriv),
]

states = {
    "t": time_points,
    "x": points[:, 0],
    "y": points[:, 1],
    "z": points[:, 2],
    "vx": velocities[:, 0],
    "vy": velocities[:, 1],
    "vz": velocities[:, 2],
    "ax": accelerations[:, 0],
    "ay": accelerations[:, 1],
    "az": accelerations[:, 2],
    "gamma": np.deg2rad(gamma_points_deg),
    "gamma_dot": np.zeros(len(points[:, 0])),
    "gamma_ddot": np.zeros(len(points[:, 0])),
}

constraints = {
    "vx0": 0,
    "vy0": 0,
    "gamma_dot0": 0,
    "min_acc_x": min_acc_x,
    "min_acc_y": min_acc_y,
    "max_acc_y": max_acc_y,
    "max_acc_x": max_acc_x,
    "offset": 200,
    "angle_offset": np.deg2rad(45),
}

# environment parameters
env_params = {
    "g": g,
}

# Rocket parameters
rocket_params = {
    "m": m,
    "h": h,
    "radius": radius,
    "C": C,
    "K_tvc": K_tvc,
    "T_tvc": T_tvc,
    "l_tvc": l_tvc,
    "K_thrust": K_thrust,
    "T_thrust": T_thrust,
}

# Controller parameters
controller_params = {
    "dt": dt,
    "u_bounds": u_bounds,
    "t0": t0_val,
    "x0": x0_val,
    "Q": Q,
    "Qf": Qf,
    "R": R,
    "thrust_bounds": (u_min_f, u_max_f),
    "delta_tvc_bounds": (u_min_delta_tvc_c, u_max_delta_tvc_c),
}

x, u = mpc_trajectory_planning(
    env_params, rocket_params, controller_params, states, constraints
)

plot_simulation(x, u, controller_params, states)

t = np.linspace(0, len(x[0]) * dt, len(x[0]))
# animate(
#     t,
#     x=x[:, 0],
#     y=x[:, 2],
#     gamma=x[:, 4],
#     delta_tvc=x[:, 6],
#     state_horizon_list=None,
#     control_horizon_list=None,
#     trajectory_params=states,
#     scale=1,
#     matplotlib=False,
#     save=True,
# )

import json

# Define the variable names
variable_names = ["x", "x_dot", "y", "y_dot", "gamma", "gamma_dot"]

# Create a dictionary with the variable names and their values
data = {}
for i in range(len(variable_names)):
    data[variable_names[i]] = x[i]

# Write the dictionary to a JSON file
with open("variables.json", "w") as file:
    json.dump(data, file, indent=4)