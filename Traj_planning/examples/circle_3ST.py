import sys
import numpy as np
from Traj_planning.traj_3ST.traj_generator_3ST import *

sys.path.append("../")
sys.path.append("../Traj_planning")

from parameters import *
from Traj_planning.examples.simple_circ import *

v = 10  # 40 works
r = 100  # with 100

# constraints
max_vx = 150
min_vx = -150
max_vy = 150
min_vy = -150
max_vz = 150
min_vz = -150

max_ax = 30
min_ax = -30
max_ay = 30
min_ay = -30
max_az = 30
min_az = -30

x, y, vx, vy, t, e1bx, e1by = calculate_traj_params(v, r, m, g)
z = np.zeros_like(x)

constraints = {
    "min_vx": min_vx,
    "max_vx": max_vx,
    "min_vy": min_vy,
    "max_vy": max_vy,
    "min_vz": min_vz,
    "max_vz": max_vz,
    "min_ax": min_ax,
    "max_ax": max_ax,
    "min_ay": min_ay,
    "max_ay": max_ay,
    "min_az": min_az,
    "max_az": max_az,
    "g": g,
}

none_vec = np.array([None] * len(x))
none_vec[0] = 0
none_vec[-1] = 0

states = {
    "t": t,
    "x": x,
    "y": y,
    "z": z,
    "vx": vx,
    "vy": vy,
    "vz": none_vec,
    "ax": none_vec,
    "ay": none_vec,
    "az": none_vec,
    "gamma_dot": none_vec,
}


def traj_circle_3ST():
    traj_params = trajenerator_3ST(
        states, constraints, env_params, rocket_params, controller_params
    )
    return traj_params
