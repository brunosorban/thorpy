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

J_3 = (
    1 / 2 * m * radius**2
)  # moment of inertia of the hopper perpendicular to the main axis
J_1 = (
    1 / 12 * m * (h**2 + 3 * radius**2)
)  # moment of inertia of the hopper perpendicular to the main axis
J_2 = (
    1 / 12 * m * (h**2 + 3 * radius**2)
)  # moment of inertia of the hopper perpendicular to the main axis

I = J_3
J = J_1

# initial state of the oscillation point (the trajectory generator will return to the CM)
initial_state = [
    0,  # x
    0,  # x_dot
    0,  # y
    0,  # y_dot
    -J / (m * l_tvc),  # z
    0,  # z_dot
    1,  # e1bx
    0,  # e1by
    0,  # e1bz
    0,  # e2bx
    1,  # e2by
    0,  # e2bz
    0,  # e3bx
    0,  # e3by
    1,  # e3bz
    1,  # e1tx
    0,  # e1ty
    0,  # e1tz
    0,  # e2tx
    1,  # e2ty
    0,  # e2tz
    0,  # e3tx
    0,  # e3ty
    1,  # e3tz
    0,  # omega_x
    0,  # omega_y
    0,  # omega_z
    m * g,  # thrust
]  # initial state of the center of mass (oscillation point displacement)
x_target = None

# controller parameters
T = 3  # time horizon
freq = 60  # frequency of the controller
N = int(T * freq)  # Number of control intervals

# controller input bounds
max_thrust = 2 * m * g  # maxium thrust rate
min_thrust = 0.25 * max_thrust  # should be 30% to 40% of the max thrust
max_delta_tvc_y = np.deg2rad(15)  # maxium thrust vector angle
min_delta_tvc_y = -np.deg2rad(15)  # minium thrust vector angle
max_delta_tvc_z = np.deg2rad(15)  # maxium thrust vector angle
min_delta_tvc_z = -np.deg2rad(15)  # minium thrust vector angle

stab_time_criterion = 3  # corresponds to the system reach 95% of the final value
max_thrust_deriv = max_thrust / (
    stab_time_criterion * T_thrust
)  # max_thrust / T_thrust # maxium thrust rate
min_thrust_deriv = -max_thrust_deriv  # should be 30% to 40% of the max thrust

max_delta_tvc_y_deriv = np.deg2rad(30)  # 30 deg/s
min_delta_tvc_y_deriv = -max_delta_tvc_y_deriv
max_delta_tvc_z_deriv = np.deg2rad(30)  # 30 deg/s
min_delta_tvc_z_deriv = -max_delta_tvc_z_deriv

q1 = 15  # position in x cost penalty
q2 = 20  # velocity in x cost penalty
q3 = 15  # position in y cost penalty
q4 = 20  # velocity in y cost penalty
q5 = 15  # position in z cost penalty
q6 = 20  # velocity in z cost penalty
q7 = 150  # roll angle cost penalty
q8 = 30  # roll rate cost penalty
q9 = 150  # pitch angle cost penalty
q10 = 30  # pitch rate cost penalty
q11 = 150  # yaw angle cost penalty
q12 = 30  # yaw rate cost penalty
q13 = 1e-15  # thrust cost penalty
q14 = 1e-15  # thrust vector angle cost penalty
q15 = 1e-15  # thrust vector angle cost penalty
Qf_gain = 10  # gain of the final cost

r1 = 1e-3  # thrust cost penalty
r2 = 200  # thrust vector angle cost penalty
r3 = 200  # thrust vector angle cost penalty

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

Q = ca.diag(
    [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12, q13, q14, q15]
)  # cost matrix
Qf = Qf_gain * Q  # final cost matrix
R = ca.diag([r1, r2, r3])  # control cost matrix


######################### Creating the dictionaries ############################
# environment parameters
env_params = {
    "g": g,
    "max_drift": max_drift,
    "max_angular_drift": max_angular_drift,
}

# Rocket parameters
rocket_params = {
    "m": m,
    "h": h,
    "radius": radius,
    "J_1": J_1,
    "J_2": J_2,
    "J_3": J_3,
    "I": I,  # used during traj gen to leverage diferential flatness properties
    "J": J,  # used during traj gen to leverage diferential flatness properties
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
    "delta_tvc_bounds": (min_delta_tvc_y, max_delta_tvc_y),
    "delta_tvc_bounds": (min_delta_tvc_z, max_delta_tvc_z),
    "delta_tvc_dot_bounds": (min_delta_tvc_y_deriv, max_delta_tvc_y_deriv),
    "delta_tvc_dot_bounds": (min_delta_tvc_z_deriv, max_delta_tvc_z_deriv),
    "safety_factor_num_int": safety_factor_num_int,
}
