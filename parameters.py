import numpy as np
import casadi as ca


############################ User defined parameters ###########################
# environment variables
g = 9.81  # gravity

# rocket variables
m = 100  # mass of the hopper
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

J_z = (
    1 / 12 * m * (h**2 + 3 * radius**2)
)  # moment of inertia of the hopper perpendicular to the main axis

# initial state of the oscillation point (the trajectory generator will return to the CM)
initial_state = [0, 0, -J_z / (m * l_tvc), 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, m * g]
x_target = None

# controller parameters
T = 3  # time horizon
freq = 60  # frequency of the controller
N = int(T * freq)  # Number of control intervals

# controller input bounds
max_thrust = 2 * m * g  # maxium thrust rate
min_thrust = 0.25 * max_thrust  # should be 30% to 40% of the max thrust
max_delta_tvc_c = np.deg2rad(15)  # maxium thrust vector angle
min_delta_tvc_c = -np.deg2rad(15)  # minium thrust vector angle

stab_time_criterion = 3  # corresponds to the system reach 95% of the final value
max_thrust_deriv = max_thrust / (
    stab_time_criterion * T_thrust
)  # max_thrust / T_thrust # maxium thrust rate
min_thrust_deriv = -max_thrust_deriv  # should be 30% to 40% of the max thrust

max_delta_tvc_c_deriv = np.deg2rad(30)  # 30 deg/s
min_delta_tvc_c_deriv = -max_delta_tvc_c_deriv

q1 = 15  # position in x cost penalty
q2 = 20  # velocity in x cost penalty
q3 = 15  # position in y cost penalty
q4 = 20  # velocity in y cost penalty
q5 = 150  # yaw angle cost penalty
q6 = 30  # yaw rate cost penalty
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

# Trajectory generator parameters
max_drift = 5 / 100  # maximum drift of the trajectory (1 equals to 100%)
max_angular_drift = np.deg2rad(5)  # maximum angular drift of the trajectory in radians
safety_factor_num_int = 1.1  # safety margin for the numerical integrator. The constraint will be the maximum value of the numerical integrator divided this factor

###################### Calculated and casadi varibles ##########################
# initial state
t0_val = 0  # initial time
x0_val = ca.vertcat(*initial_state)  # initial state in casadi varible

Q = ca.diag([q1, q2, q3, q4, q5, q6, q7, q8])  # cost matrix
Qf = Qf_gain * Q  # final cost matrix
R = ca.diag([r1, r2])  # control cost matrix


######################### Creating the dictionaries ############################
# environment parameters
env_params = {
    "g": g,
    "max_drift": max_drift,
    "max_angular_drift": max_angular_drift,
    "safety_factor_num_int": safety_factor_num_int,
}

# Rocket parameters
rocket_params = {
    "m": m,
    "h": h,
    "radius": radius,
    "J_z": J_z,
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
    "t0": t0_val,
    "x0": x0_val,
    "x_target": x_target,
    "Q": Q,
    "Qf": Qf,
    "R": R,
    "thrust_bounds": (min_thrust, max_thrust),
    "thrust_dot_bounds": (min_thrust_deriv, max_thrust_deriv),
    "delta_tvc_bounds": (min_delta_tvc_c, max_delta_tvc_c),
    "delta_tvc_dot_bounds": (min_delta_tvc_c_deriv, max_delta_tvc_c_deriv),
}
