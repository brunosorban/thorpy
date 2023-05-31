import numpy as np
import casadi as ca


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

# solution parameters
t0 = 0  # initial time
tf = 60  # final time

# state space: [x, x_dot, y, y_dot, gamma, gamma_dot, thrust, delta_tvc]
# initial_state = [0, 0, 0, 0, np.deg2rad(90), 0, m * g, 0]
initial_state = [200, 0, 0, 35, 0, 1, -1, 0, 0, 1, -1, 0, 0, m * g]
# target = [30, 0, 30, 0, np.deg2rad(90), 0, 0, 0]
x_target = None
max_step = 1e-4

# controller parameters
T = 3  # time horizon
N = 60  # Number of control intervals

# controller input bounds
u_max_f = 2 * m * g  # maxium thrust rate
u_min_f = 0.35 * u_max_f  # should be 30% to 40% of the max thrust
u_max_delta_tvc_c = np.deg2rad(10)  # maxium thrust vector angle
u_min_delta_tvc_c = -np.deg2rad(10)  # minium thrust vector angle

# state bounds
gamma_max = np.deg2rad(30) + initial_state[4]  # maxium yaw angle
gamma_min = np.deg2rad(-30) + initial_state[4]  # minium yaw angle

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

# normalizing parameters (not being used)
pos_norm = 1  # position normalization
vel_norm = 1  # velocity normalization
angle_norm = 1  # np.deg2rad(15)  # angle normalization
angle_rate_norm = 1  # np.deg2rad(5)  # angle rate normalization
thrust_norm = 1  # thrust normalization

###################### Calculated and casadi varibles ##########################
C = (
    1 / 12 * m * (h**2 + 3 * radius**2)
)  # moment of inertia of the hopper perpendicular to the main axis

# initial state
t0_val = 0  # initial time
x0_val = ca.vertcat(*initial_state)  # initial state in casadi varible
# x_target = ca.vertcat(*target)  # target state in casadi varible

Q = ca.diag([q1, q2, q3, q4, q5, q6, q7, q8])  # cost matrix
Qf = Qf_gain * Q  # final cost matrix
R = ca.diag([r1, r2])  # control cost matrix

# control bounds
u_bounds = [
    (u_min_f_deriv, u_max_f_deriv),
    (u_min_delta_tvc_c_deriv, u_max_delta_tvc_c_deriv),
]


######################### Creating the dictionaries ############################
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
    "thrust_bounds": (u_min_f, u_max_f),
    "delta_tvc_bounds": (u_min_delta_tvc_c, u_max_delta_tvc_c),
    "omega_control_bounds": (1, 1),
}

normalization_params_x = [
    1 / pos_norm,
    1 / vel_norm,
    1 / pos_norm,
    1 / vel_norm,
    1 / angle_norm,
    1 / angle_rate_norm,
    1 / thrust_norm,
    1 / angle_norm,
]

normalization_params_u = [thrust_norm, angle_norm]

# Solution parameters
# sol_params = {
#     "t0": t0,
#     "tf": tf,
#     "initial_state": initial_state,
#     "target": target,
#     "max_step": max_step,
# }
